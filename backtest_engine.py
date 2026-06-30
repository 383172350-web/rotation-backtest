"""
轮动策略回测引擎
核心逻辑：定期排名 → 条件卖出 → 冷冻期+条件买入
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from indicators import compute_all_indicators, compute_indicators_for_df
from expression_parser import evaluate_condition, evaluate_score, _fast_evaluate_simple
from precompiler import precompute_strategy, has_special_var


class Position:
    """单个持仓"""
    def __init__(self, code: str, name: str, shares: float, cost_price: float,
                 entry_date: str, exit_date: Optional[str] = None):
        self.code = code
        self.name = name
        self.shares = shares
        self.cost_price = cost_price
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.current_price = cost_price

    @property
    def market_value(self):
        return self.shares * self.current_price

    def update_price(self, price: float):
        self.current_price = price

    @property
    def profit_pct(self):
        """当前收益率"""
        if self.cost_price == 0:
            return 0
        return (self.current_price - self.cost_price) / self.cost_price

    @property
    def hold_days(self):
        if not hasattr(self, '_hold_days'):
            self._hold_days = 0
        return self._hold_days

    @hold_days.setter
    def hold_days(self, val):
        self._hold_days = val


class BacktestEngine:
    """轮动策略回测引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.strategy = config['strategy']

        bt = self.strategy['backtest']
        self.initial_capital = bt['initial_capital']
        self.commission = bt.get('commission', 0.0001)
        self.slippage = bt.get('slippage', 0.0)
        self.start_date = bt['start_date']
        self.end_date = bt.get('end_date', datetime.now().strftime("%Y-%m-%d"))

        pos = self.strategy['position']
        self.max_count = pos['max_count']
        self.pos_mode = pos.get('mode', 'adaptive')
        # fixed模式：自动根据max_count计算比例（5只=20%，4只=25%）
        # 如果配置了fixed_ratio则优先使用配置值
        if 'fixed_ratio' in pos:
            self.fixed_ratio = pos['fixed_ratio']
        else:
            self.fixed_ratio = 1.0 / self.max_count

        rebal = self.strategy['rebalance']
        self.rebalance_freq = rebal.get('frequency', 'weekly')
        self.rebalance_weekday = rebal.get('weekday', 5)
        self.rebalance_day = rebal.get('day_of_month', 1)
        self.rebalance_interval = rebal.get('interval', 1)  # 每N个交易日轮动一次

        self.alternative_asset = self.strategy.get('alternative_asset')  # 替代资产

        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[dict] = []
        self.daily_values: List[dict] = []
        self.daily_positions: List[dict] = []  # 每日持仓明细
        self.rebalance_plans: List[dict] = []  # 调仓计划记录

        self.all_data: Dict[str, pd.DataFrame] = {}
        self.all_dates: List = []

        # T+1模式：待执行订单队列
        self.pending_sells: List[dict] = []   # [{'code':, 'reason':}, ...]
        self.pending_buys: List[dict] = []    # [{'code':, 'name':, 'score':}, ...]

    def load_data(self, data: Dict[str, pd.DataFrame]):
        """加载预处理好的数据"""
        self.all_data = {}
        all_dates_set = None

        for code, df in data.items():
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            # 按需计算指标：只计算策略公式中用到的指标，大幅提速
            expressions = [self.strategy.get('rank_formula', '')]
            for rule in self.strategy.get('buy_rules', []):
                expressions.append(rule.get('condition', '') if isinstance(rule, dict) else str(rule))
            for rule in self.strategy.get('sell_rules', []):
                expressions.append(rule.get('condition', '') if isinstance(rule, dict) else str(rule))
            df = compute_indicators_for_df(df, expressions)
            # 预计算策略表达式（不涉及特殊变量的）
            df = precompute_strategy(df, self.strategy)
            self.all_data[code] = df
            # 只保留股票池标的（排除替代资产、基准等）的交易日交集
            alt_code = self.alternative_asset.get('code', '__alt__') if self.alternative_asset else '__alt__'
            if not code.startswith('__') and code != alt_code:
                if all_dates_set is None:
                    all_dates_set = set(df.index.tolist())
                else:
                    all_dates_set = all_dates_set.intersection(set(df.index.tolist()))

        if all_dates_set is None:
            all_dates_set = set()
        self.all_dates = sorted(all_dates_set)
        print(f"已加载 {len(self.all_data)} 只标的，共 {len(self.all_dates)} 个交易日（交易日交集）")

    def run(self) -> dict:
        """执行回测"""
        print(f"\n{'='*60}")
        print(f"开始回测: {self.strategy['name']}")
        print(f"期间: {self.start_date} ~ {self.end_date}")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"{'='*60}\n")

        universe = {item['code']: item['name'] for item in self.strategy['universe']}
        alt_code_str = self.alternative_asset['code'] if self.alternative_asset else ''

        start_dt = pd.Timestamp(self.start_date)
        end_dt = pd.Timestamp(self.end_date)

        for date in self.all_dates:
            # 跳过回测区间外的日期
            if date < start_dt or date > end_dt:
                continue
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]

            # === T+1执行：执行T日积累的待执行订单（用T+1日open价）===
            self._execute_pending_orders(date, date_str, universe)

            self._update_positions(date)

            for pos in self.positions.values():
                pos.hold_days += 1

            # === T日信号：每天检查卖出条件（用收盘价判断，次日执行）===
            self._check_sell_conditions(date, date_str, universe, alt_code_str)

            # === T日信号：只在轮动日执行买入逻辑（用收盘价判断，次日执行）===
            if self._is_rebalance_date(date):
                self._rebalance_buy(date, date_str, universe)

            total_value = self.cash + sum(p.market_value for p in self.positions.values())
            self.daily_values.append({
                'date': date_str,
                'total_value': total_value,
                'cash': self.cash,
                'positions_value': total_value - self.cash,
                'num_positions': len(self.positions)
            })

            # 记录每日持仓明细
            for code, pos in self.positions.items():
                self.daily_positions.append({
                    'date': date_str,
                    'code': code,
                    'name': pos.name,
                    'shares': pos.shares,
                    'cost_price': round(pos.cost_price, 4),
                    'current_price': round(pos.current_price, 4),
                    'market_value': round(pos.market_value, 2),
                    'profit_pct': round(pos.profit_pct, 4),
                    'hold_days': pos.hold_days
                })
            # 如果当日无持仓，记录空仓状态
            if not self.positions:
                self.daily_positions.append({
                    'date': date_str,
                    'code': '',
                    'name': '空仓',
                    'shares': 0,
                    'cost_price': 0,
                    'current_price': 0,
                    'market_value': 0,
                    'profit_pct': 0,
                    'hold_days': 0
                })

        return self._generate_results()

    def _execute_pending_orders(self, date, date_str: str, universe: dict):
        """
        T+1执行：执行T日积累的待执行订单，使用T+1日的open价格。
        替代资产就是cash，不需要单独买卖。
        """
        # 1. 先执行卖出（释放资金给买入用）
        for order in self.pending_sells:
            code = order['code']
            reason = order['reason']
            self._execute_sell(code, date_str, reason, date)
        self.pending_sells = []

        # 2. 再执行买入
        for order in self.pending_buys:
            code = order['code']
            name = order['name']
            score = order['score']
            self._execute_buy(code, name, date, score)
        self.pending_buys = []

    def _update_positions(self, date):
        for code, pos in self.positions.items():
            if code in self.all_data:
                df = self.all_data[code]
                if date in df.index:
                    price = df.loc[date, 'close']
                    pos.update_price(price)

    def _is_rebalance_date(self, date) -> bool:
        dt = date if isinstance(date, datetime) else pd.Timestamp(date)

        # 支持 interval 模式：每N个交易日轮动一次
        if self.rebalance_freq == 'interval':
            if not hasattr(self, '_trading_day_count'):
                self._trading_day_count = 0
            self._trading_day_count += 1
            if self._trading_day_count >= self.rebalance_interval:
                self._trading_day_count = 0
                return True
            return False
        elif self.rebalance_freq == 'daily':
            return True
        elif self.rebalance_freq == 'weekly':
            return dt.weekday() == self.rebalance_weekday - 1
        elif self.rebalance_freq == 'monthly':
            return dt.day == self.rebalance_day
        return False

    def _check_sell_conditions(self, date, date_str: str, universe: dict, alt_code_str: str):
        """
        T日信号：每天检查卖出条件（用收盘价判断），满足则加入待执行队列，次日open价执行。
        优化：预计算列直接读取；涉及特殊变量的简单条件用 fast_evaluate，不传整df。
        """
        all_rankings = self._rank_candidates(date, universe)
        rank_map = {code: rank for rank, (code, _) in enumerate(all_rankings, 1)}

        sell_rules = self.strategy.get('sell_rules', [])
        sell_mode = self.strategy.get('sell_match_mode', 'any')
        # 检查哪些规则有预计算列
        sample_df = next(iter(self.all_data.values())) if self.all_data else None
        has_precols = [f'__sell_{i}' in sample_df.columns for i in range(len(sell_rules))] if sample_df is not None else [False]*len(sell_rules)

        for code, pos in self.positions.items():
            if code == alt_code_str:
                continue
            if code not in self.all_data:
                continue

            df = self.all_data[code]
            if date not in df.index:
                continue

            current_rank = rank_map.get(code, 999)
            extra_vars = {
                'profit': pos.profit_pct,
                'hold_days': pos.hold_days,
                'rank': current_rank
            }

            sell_results = []
            sell_reason = ""

            for i, rule in enumerate(sell_rules):
                condition = rule['condition']
                met = False
                try:
                    if has_precols[i] and not has_special_var(condition):
                        # 预计算列直接读取
                        val = df.loc[date, f'__sell_{i}']
                        met = bool(val)
                    else:
                        # 先尝试简单条件快速求值（不涉及df操作）
                        fast_result = _fast_evaluate_simple(condition, extra_vars)
                        if fast_result is not None:
                            met = fast_result
                        else:
                            # 回退到 ExpressionParser（复杂表达式），传入整df不切片
                            result = evaluate_condition(condition, df, extra_vars)
                            if isinstance(result, pd.Series):
                                met = bool(result.iloc[-1])
                            else:
                                met = bool(result)
                    sell_results.append(met)
                    if met:
                        sell_reason = rule.get('description', condition)
                except Exception:
                    sell_results.append(False)

            should_sell = all(sell_results) if sell_mode == 'all' else any(sell_results)
            if should_sell:
                self.pending_sells.append({'code': code, 'reason': sell_reason})
                # 记录调仓计划（卖出）
                self.rebalance_plans.append({
                    'date': date_str,
                    'plan_type': 'SELL_PLAN',
                    'code': code,
                    'name': pos.name,
                    'detail': sell_reason,
                    'profit_pct': round(pos.profit_pct, 4),
                    'hold_days': pos.hold_days
                })

    def _rebalance_buy(self, date, date_str: str, universe: dict):
        """
        只在轮动日执行：排名 + 条件买入 + 替代资产管理。
        优化：预计算列直接读取；涉及特殊变量的简单条件用 fast_evaluate。
        """
        alt_code_str = self.alternative_asset['code'] if self.alternative_asset else ''

        # 1. 全量排名（已持仓 + 未持仓一起排）
        rankings = self._rank_candidates(date, universe)
        rank_map_for_buy = {code: rank for rank, (code, _) in enumerate(rankings, 1)}

        # 2. 计算可用仓位（替代资产不算）
        equity_count = sum(1 for c in self.positions if c != alt_code_str)
        available_slots = self.max_count - equity_count

        if available_slots <= 0:
            return

        buy_rules = self.strategy.get('buy_rules', [])
        buy_mode = self.strategy.get('buy_match_mode', 'all')
        # 检查哪些规则有预计算列
        sample_df = next(iter(self.all_data.values())) if self.all_data else None
        has_precols = [f'__buy_{i}' in sample_df.columns for i in range(len(buy_rules))] if sample_df is not None else [False]*len(buy_rules)

        planned_buys = []
        # 3. 检查买入条件：从排名靠前的未持仓品种中选入
        for code, score in rankings:
            if code in self.positions:
                continue
            if code not in self.all_data:
                continue

            df = self.all_data[code]
            if date not in df.index:
                continue

            row_idx = df.index.get_loc(date)
            if row_idx < 1:
                continue

            current_rank = rank_map_for_buy.get(code, 999)
            extra_vars = {'rank': current_rank}
            buy_results = []

            for i, rule in enumerate(buy_rules):
                condition = rule['condition']
                met = False
                try:
                    if has_precols[i] and not has_special_var(condition):
                        # 预计算列直接读取
                        val = df.loc[date, f'__buy_{i}']
                        met = bool(val)
                    else:
                        # 先尝试简单条件快速求值
                        fast_result = _fast_evaluate_simple(condition, extra_vars)
                        if fast_result is not None:
                            met = fast_result
                        else:
                            # 回退到 ExpressionParser，传入整df不切片
                            result = evaluate_condition(condition, df, extra_vars)
                            if isinstance(result, pd.Series):
                                met = bool(result.iloc[-1])
                            else:
                                met = bool(result)
                    buy_results.append(met)
                except Exception:
                    buy_results.append(False)

            all_conditions_met = all(buy_results) if buy_mode == 'all' else any(buy_results)

            if all_conditions_met:
                planned_buys.append({'code': code, 'name': universe.get(code, code), 'score': score})
                self.pending_buys.append({'code': code, 'name': universe.get(code, code), 'score': score})
                equity_count += 1
                if equity_count >= self.max_count:
                    break

        # 记录调仓计划（买入）
        for plan in planned_buys:
            self.rebalance_plans.append({
                'date': date_str,
                'plan_type': 'BUY_PLAN',
                'code': plan['code'],
                'name': plan['name'],
                'detail': f"排名得分: {plan['score']:.2f}",
                'profit_pct': 0,
                'hold_days': 0
            })

    def _rank_candidates(self, date, universe: dict) -> List[Tuple[str, float]]:
        rank_formula = self.strategy.get('rank_formula', 'returns(20)')
        direction = self.strategy.get('rank_direction', 'desc')
        # 检查是否有预计算列
        sample_df = next(iter(self.all_data.values())) if self.all_data else None
        has_precomputed = '__rank_score' in sample_df.columns if sample_df is not None else False
        needs_dynamic = has_special_var(rank_formula)

        scores = []
        for code in universe:
            if code not in self.all_data:
                continue
            df = self.all_data[code]
            if date not in df.index:
                continue

            row_idx = df.index.get_loc(date)
            if row_idx < 60:
                continue

            try:
                if has_precomputed and not needs_dynamic:
                    # 直接读取预计算列
                    score = df.loc[date, '__rank_score']
                else:
                    # 回退到动态解析，传入整df不切片
                    score = evaluate_score(rank_formula, df)
                    if isinstance(score, pd.Series):
                        score = score.iloc[-1]
                if not np.isnan(score):
                    scores.append((code, float(score)))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=(direction == 'desc'))
        return scores

    def _execute_buy(self, code: str, name: str, date, score: float):
        """T+1执行买入：使用T+1日的open价格。"""
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]

        df = self.all_data[code]
        if date not in df.index:
            return
        price = df.loc[date, 'open'] * (1 + self.slippage)

        total_value = self.cash + sum(p.market_value for p in self.positions.values())

        if self.pos_mode == 'fixed':
            target_value = total_value * self.fixed_ratio
        else:
            target_count = min(self.max_count, len(self.positions) + 1)
            target_value = total_value / target_count

        target_value = min(target_value, self.cash * 0.99)

        if target_value < 100:
            return

        shares = int(target_value / price / 100) * 100
        if shares <= 0:
            shares = int(target_value / price)
        if shares <= 0:
            return

        cost = shares * price * (1 + self.commission)
        if cost > self.cash:
            shares = int(self.cash / price / (1 + self.commission))
            if shares <= 0:
                return
            cost = shares * price * (1 + self.commission)

        self.cash -= cost
        pos = Position(code, name, shares, price, date_str)
        pos.current_price = price
        self.positions[code] = pos

        self.trade_log.append({
            'date': date_str,
            'action': 'BUY',
            'code': code,
            'name': name,
            'price': round(price, 4),
            'shares': shares,
            'amount': round(cost, 2),
            'reason': f'排名得分: {score:.2f}'
        })

        print(f"  [BUY] {date_str} 买入 {name}({code}) {shares}股 @ {price:.4f}, 金额={cost:.0f}")

    def _execute_sell(self, code: str, date_str: str, reason: str, date):
        """T+1执行卖出：使用T+1日的open价格。"""
        if code not in self.positions:
            return

        pos = self.positions[code]
        if code in self.all_data and date in self.all_data[code].index:
            price = self.all_data[code].loc[date, 'open'] * (1 - self.slippage)
        else:
            price = pos.current_price * (1 - self.slippage)
        amount = pos.shares * price * (1 - self.commission)

        profit_pct = pos.profit_pct
        profit_amount = amount - pos.shares * pos.cost_price

        self.cash += amount

        self.trade_log.append({
            'date': date_str,
            'action': 'SELL',
            'code': code,
            'name': pos.name,
            'price': round(price, 4),
            'shares': pos.shares,
            'amount': round(amount, 2),
            'profit_pct': f"{profit_pct:.2%}",
            'profit_amount': round(profit_amount, 2),
            'reason': reason
        })

        print(f"  [SELL] {date_str} 卖出 {pos.name}({code}) {pos.shares}股 @ {price:.4f}, "
              f"收益={profit_pct:.2%}({profit_amount:.0f}), 原因: {reason}")

        del self.positions[code]

    def _generate_results(self) -> dict:
        if not self.daily_values:
            return {'error': '无回测数据'}

        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        df['nav'] = df['total_value'] / self.initial_capital
        df['daily_return'] = df['nav'].pct_change()
        df['cummax'] = df['nav'].cummax()
        df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']

        # 从第一笔买入日开始计算收益（排除纯现金的预热期）
        trade_df_raw = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        if not trade_df_raw.empty:
            first_trade_date = pd.to_datetime(trade_df_raw['date'].min())
            df_calc = df[df.index >= first_trade_date]
        else:
            df_calc = df

        total_return = (df['total_value'].iloc[-1] / self.initial_capital) - 1
        days = (df.index[-1] - df_calc.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        max_drawdown = df['drawdown'].min()

        if df['daily_return'].std() > 0:
            sharpe = (df['daily_return'].mean() * 252 - 0.03) / (df['daily_return'].std() * np.sqrt(252))
        else:
            sharpe = 0

        trade_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        sell_trades = trade_df[trade_df['action'] == 'SELL'] if not trade_df.empty else pd.DataFrame()

        if not sell_trades.empty and 'profit_amount' in sell_trades.columns:
            sell_trades_copy = sell_trades.copy()
            sell_trades_copy['profit_amount_num'] = pd.to_numeric(sell_trades_copy['profit_amount'], errors='coerce')
            win_trades = sell_trades_copy[sell_trades_copy['profit_amount_num'] > 0]
            win_rate = len(win_trades) / len(sell_trades_copy)
        else:
            win_rate = 0

        results = {
            'strategy_name': self.strategy['name'],
            'period': f"{self.start_date} ~ {self.end_date}",
            'initial_capital': self.initial_capital,
            'final_value': round(df['total_value'].iloc[-1], 2),
            'total_return': f"{total_return:.2%}",
            'annual_return': f"{annual_return:.2%}",
            'max_drawdown': f"{max_drawdown:.2%}",
            'sharpe_ratio': round(sharpe, 2),
            'total_trades': len(self.trade_log),
            'win_rate': f"{win_rate:.2%}",
            'daily_values': df,
            'trade_log': trade_df,
            'daily_positions': pd.DataFrame(self.daily_positions) if self.daily_positions else pd.DataFrame(),
            'rebalance_plans': pd.DataFrame(self.rebalance_plans) if self.rebalance_plans else pd.DataFrame()
        }

        print(f"\n{'='*60}")
        print(f"回测完成: {self.strategy['name']}")
        print(f"{'='*60}")
        print(f"期间: {results['period']}")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"最终市值: {results['final_value']:,.0f}")
        print(f"总收益率: {results['total_return']}")
        print(f"年化收益: {results['annual_return']}")
        print(f"最大回撤: {results['max_drawdown']}")
        print(f"夏普比率: {results['sharpe_ratio']}")
        print(f"总交易次数: {results['total_trades']}")
        print(f"胜率: {results['win_rate']}")
        print(f"{'='*60}")

        return results
