"""
持仓状态查询 & 调仓计划模块
功能：
1. 当前持仓（品种、仓位比例、成本、浮动盈亏）
2. 历史持仓（曾持有但已卖出的品种列表）
3. 调仓计划（下一个交易日需要执行的操作）
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


def get_portfolio_status(config: dict, trade_log_path: str = 'output/trade_log.csv',
                        data_dir: str = '.') -> dict:
    """
    获取完整的持仓状态报告

    Returns:
        dict: 包含 current_positions, history_positions, rebalance_plan
    """
    import yaml
    from data_fetcher import fetch_kline
    from indicators import compute_all_indicators
    from expression_parser import ExpressionParser, evaluate_condition

    strategy = config['strategy']
    universe = strategy['universe']
    code_name_map = {item['code']: item['name'] for item in universe}
    alt = strategy.get('alternative_asset')
    alt_code = alt['code'] if alt else ''

    # 读取交易日志
    trade_df = pd.read_csv(trade_log_path) if trade_log_path else pd.DataFrame()
    trade_df['date'] = pd.to_datetime(trade_df['date'])

    if trade_df.empty:
        return {'error': '无交易日志'}

    # 获取最新数据
    all_data = {}
    for item in universe:
        code = item['code']
        try:
            df = fetch_kline(code, '2024-01-01', datetime.now().strftime('%Y-%m-%d'))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df = compute_all_indicators(df)
                all_data[code] = df
        except:
            pass

    if alt:
        try:
            df = fetch_kline(alt['code'], '2024-01-01', datetime.now().strftime('%Y-%m-%d'))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df = compute_all_indicators(df)
                all_data[alt['code']] = df
        except:
            pass

    # 找最新交易日
    latest = None
    for code, df in all_data.items():
        d = pd.to_datetime(df['date']).max()
        if latest is None or d < latest:
            latest = d

    if latest is None:
        return {'error': '无法获取最新数据'}

    # ============================================================
    # 1. 推断当前持仓
    # ============================================================
    # T+1模式: 交易日志中的日期 = 信号产生日的前一天的开盘执行日
    # 也就是: 6月26日的BUY记录 = 6月25日收盘信号 → 6月26日开盘执行
    # 最后一条记录是6月26日执行的操作，6月26日收盘的信号还没执行
    positions = {}  # code -> {'shares': x, 'buy_price': x, 'buy_date': x, 'name': x}

    for _, row in trade_df.iterrows():
        code = row['code']
        if row['action'] == 'BUY':
            positions[code] = {
                'shares': row['shares'],
                'buy_price': row['price'],
                'buy_date': row['date'],
                'name': row['name'],
            }
        elif row['action'] == 'SELL':
            if code in positions:
                del positions[code]

    # 计算当前市值和盈亏
    total_value = 0
    current_positions = []
    for code, pos in positions.items():
        name = pos['name']
        shares = pos['shares']
        buy_price = pos['buy_price']
        buy_date = pd.to_datetime(pos['buy_date']).strftime('%Y-%m-%d')

        # 获取最新价格
        current_price = buy_price
        if code in all_data:
            df = all_data[code]
            last_close = df['close'].iloc[-1]
            if not np.isnan(last_close):
                current_price = last_close

        market_value = shares * current_price
        cost = shares * buy_price
        profit = market_value - cost
        profit_pct = (current_price / buy_price - 1) if buy_price > 0 else 0
        total_value += market_value

        # 计算持仓天数
        hold_days = 0
        if code in all_data:
            df = all_data[code]
            df_d = df.set_index('date') if 'date' in df.columns else df
            buy_dt = pd.to_datetime(buy_date)
            recent = df_d[df_d.index >= buy_dt]
            hold_days = len(recent) - 1

        is_alt = code == alt_code
        current_positions.append({
            'code': code,
            'name': name,
            'shares': shares,
            'buy_price': buy_price,
            'current_price': current_price,
            'market_value': market_value,
            'cost': cost,
            'profit': profit,
            'profit_pct': profit_pct,
            'hold_days': hold_days,
            'buy_date': buy_date,
            'is_alternative': is_alt,
        })

    # cash即替代资产，加入total_value
    # 从trade_log推断：卖出回笼的资金 - 买入花费 = cash
    # 简化：total_value = 持仓市值 + cash，cash用初始资金+总卖出-总买入推算
    trade_df_cash = trade_df.copy()
    cash_flow = 0
    for _, row in trade_df_cash.iterrows():
        if row['action'] == 'SELL':
            cash_flow += row['amount']
        elif row['action'] == 'BUY':
            cash_flow -= row['amount']
    cash = config.get('backtest', config).get('initial_capital', 100000) + cash_flow
    total_value = total_value + cash

    # 加入替代资产（cash）到持仓列表
    alt_value = cash
    equity_value = total_value - alt_value
    if alt and cash > 100:
        current_positions.append({
            'code': alt_code,
            'name': alt['name'],
            'shares': 0,
            'buy_price': 0,
            'current_price': 0,
            'market_value': cash,
            'cost': cash,
            'profit': 0,
            'profit_pct': 0,
            'hold_days': 0,
            'buy_date': '',
            'is_alternative': True,
        })

    for p in current_positions:
        p['weight'] = p['market_value'] / total_value if total_value > 0 else 0

    # 按市值降序
    current_positions.sort(key=lambda x: x['market_value'], reverse=True)

    # ============================================================
    # 2. 历史持仓（曾经持有但已全部卖出的品种）
    # ============================================================
    all_codes_in_positions = set()
    history_positions = []
    for _, row in trade_df.iterrows():
        code = row['code']
        if row['action'] == 'SELL' and code not in positions:
            all_codes_in_positions.add(code)

    for code in all_codes_in_positions:
        name = code_name_map.get(code, code)
        # 获取该品种的所有交易记录
        code_trades = trade_df[trade_df['code'] == code].sort_values('date')
        total_profit = 0
        trade_count = 0
        last_sell_date = None
        for _, row in code_trades.iterrows():
            if row['action'] == 'SELL':
                total_profit += row.get('profit_amount', 0)
                trade_count += 1
                last_sell_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')

        history_positions.append({
            'code': code,
            'name': name,
            'total_profit': total_profit,
            'trade_count': trade_count,
            'last_sell_date': last_sell_date,
        })

    history_positions.sort(key=lambda x: x['last_sell_date'] or '', reverse=True)

    # ============================================================
    # 3. 调仓计划（下一个交易日需要执行的操作）
    # ============================================================
    rebalance_plan = []

    # 3a. 计算DIFv排名
    rank_formula = strategy.get('rank_formula', '(MACD_DIF / ATR_26) * 100')
    rank_direction = strategy.get('rank_direction', 'desc')
    rankings = {}

    for code in code_name_map:
        if code not in all_data:
            continue
        df = all_data[code].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        if latest not in df.index:
            continue
        row_idx = df.index.get_loc(latest)
        if row_idx < 60:
            continue
        snapshot = df.iloc[:row_idx+1].copy()
        try:
            score = ExpressionParser(snapshot).evaluate(rank_formula)
            if isinstance(score, pd.Series):
                score = score.iloc[-1]
            if not np.isnan(score):
                rankings[code] = score
        except:
            pass

    # 排名
    sorted_codes = sorted(rankings.keys(), key=lambda c: rankings[c],
                          reverse=(rank_direction == 'desc'))
    rank_map = {code: i+1 for i, code in enumerate(sorted_codes)}

    # 3b. 卖出检查
    sell_rules = strategy.get('sell_rules', [])
    sell_mode = strategy.get('sell_match_mode', 'any')

    for code, pos in list(positions.items()):
        if code == alt_code:
            continue
        if code not in all_data:
            continue

        name = pos['name']
        current_rank = rank_map.get(code, 999)

        df = all_data[code].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        if latest not in df.index:
            continue
        row_idx = df.index.get_loc(latest)
        snapshot = df.iloc[:row_idx+1].copy()

        close_val = snapshot['close'].iloc[-1]
        ret1 = (close_val / snapshot['close'].iloc[-2] - 1) if row_idx >= 1 else 0
        ret20 = (close_val / snapshot['close'].iloc[-20] - 1) if row_idx >= 19 else 0

        extra_vars = {
            'rank': current_rank,
            'profit': pos['shares'] * close_val / (pos['shares'] * pos['buy_price']) - 1 if pos['buy_price'] > 0 else 0,
            'hold_days': pos.get('hold_days', 0),
            'buy_price': pos['buy_price'],
        }

        sell_results = []
        sell_reasons = []
        for rule in sell_rules:
            condition = rule['condition']
            desc = rule.get('description', condition)
            try:
                result = evaluate_condition(condition, snapshot, extra_vars)
                if isinstance(result, pd.Series):
                    met = bool(result.iloc[-1])
                else:
                    met = bool(result)
                sell_results.append(met)
                if met:
                    sell_reasons.append(desc)
            except:
                sell_results.append(False)

        should_sell = all(sell_results) if sell_mode == 'all' else any(sell_results)

        if should_sell:
            rebalance_plan.append({
                'action': 'SELL',
                'code': code,
                'name': name,
                'shares': pos['shares'],
                'current_price': close_val,
                'current_weight': next((p['weight'] for p in current_positions if p['code'] == code), 0),
                'reason': '; '.join(sell_reasons),
                'rank': current_rank,
            })

    # 3c. 买入检查（需要排除已持仓的和将被卖出的）
    buy_rules = strategy.get('buy_rules', [])
    buy_mode = strategy.get('buy_match_mode', 'all')
    max_count = strategy.get('position', {}).get('max_count', 5)

    codes_to_sell = {p['code'] for p in rebalance_plan if p['action'] == 'SELL'}
    active_positions = {code for code in positions if code not in codes_to_sell and code != alt_code}
    equity_count = len(active_positions)
    pos_mode = strategy.get('position', {}).get('mode', 'adaptive')

    for i, code in enumerate(sorted_codes):
        if code in active_positions or code in codes_to_sell:
            continue
        if code not in all_data:
            continue
        if equity_count >= max_count:
            break

        name = code_name_map.get(code, code)
        rank = i + 1

        df = all_data[code].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        if latest not in df.index:
            continue
        row_idx = df.index.get_loc(latest)
        snapshot = df.iloc[:row_idx+1].copy()

        extra_vars = {'rank': rank}
        buy_results = []
        for rule in buy_rules:
            condition = rule['condition']
            try:
                result = evaluate_condition(condition, snapshot, extra_vars)
                if isinstance(result, pd.Series):
                    met = bool(result.iloc[-1])
                else:
                    met = bool(result)
                buy_results.append(met)
            except:
                buy_results.append(False)

        all_met = all(buy_results) if buy_mode == 'all' else any(buy_results)

        if all_met:
            close_val = snapshot['close'].iloc[-1]
            # 计算目标仓位
            if pos_mode == 'fixed':
                target_weight = 1.0 / max_count
            else:
                target_weight = 1.0 / (len(active_positions) + 1)

            rebalance_plan.append({
                'action': 'BUY',
                'code': code,
                'name': name,
                'rank': rank,
                'current_price': close_val,
                'target_weight': target_weight,
                'reason': f'排名#{rank}, 满足全部买入条件',
            })
            equity_count += 1

    # 判断下一个交易日（从数据中推断：找任意标的数据中比latest_date大的最早日期）
    next_trade_date = None
    for code, df in all_data.items():
        df_dates = pd.to_datetime(df['date'])
        future_dates = df_dates[df_dates > latest]
        if not future_dates.empty:
            candidate = future_dates.min()
            if next_trade_date is None or candidate < next_trade_date:
                next_trade_date = candidate

    if next_trade_date is None:
        # 数据中没有未来日期，用日历推算跳过周末
        next_trade_date = latest + timedelta(days=1)
        while next_trade_date.weekday() >= 5:
            next_trade_date += timedelta(days=1)

    return {
        'latest_data_date': latest.strftime('%Y-%m-%d'),
        'next_trade_date': next_trade_date.strftime('%Y-%m-%d'),
        'total_value': total_value,
        'equity_value': equity_value,
        'alt_value': alt_value,
        'equity_weight': equity_value / total_value if total_value > 0 else 0,
        'alt_weight': alt_value / total_value if total_value > 0 else 0,
        'current_positions': current_positions,
        'history_positions': history_positions,
        'rebalance_plan': rebalance_plan,
        'rankings': {code: {'rank': rank_map[code], 'score': score, 'name': code_name_map.get(code, code)}
                     for code, score in rankings.items()
                     if code in rank_map},
        'has_plan': len(rebalance_plan) > 0,
    }


def print_portfolio_status(status: dict):
    """打印格式化的持仓状态报告"""
    print('=' * 65)
    print(f'  持仓状态报告')
    print(f'  数据截止: {status.get("latest_data_date", "N/A")}')
    print(f'  下一个交易日: {status.get("next_trade_date", "N/A")}')
    print(f'  总市值: {status.get("total_value", 0):,.0f}')
    print('=' * 65)

    # 1. 当前持仓
    print()
    print('-' * 65)
    print('  当前持仓')
    print('-' * 65)
    print(f'  {"品种":18s} {"仓位":>7s} {"份额":>8s} {"成本":>8s} {"现价":>8s} {"盈亏":>8s} {"收益率":>8s} {"持仓天数":>6s}')
    print('  ' + '-' * 60)

    positions = status.get('current_positions', [])
    for p in positions:
        tag = ' [替代]' if p.get('is_alternative') else ''
        print(f'  {p["name"]:16s}{tag:4s} {p["weight"]:>6.2%} {p["shares"]:>8.0f} '
              f'{p["buy_price"]:>8.4f} {p["current_price"]:>8.4f} '
              f'{p["profit"]:>8.0f} {p["profit_pct"]:>7.2%} {p["hold_days"]:>6d}')

    eq_w = status.get('equity_weight', 0)
    alt_w = status.get('alt_weight', 0)
    print(f'  {"合计":20s} {1.0:>6.2%}')
    print(f'  其中: 股票型={eq_w:.2%}, 货币型={alt_w:.2%}')

    # 2. 历史持仓
    history = status.get('history_positions', [])
    if history:
        print()
        print('-' * 65)
        print('  历史持仓（已卖出）')
        print('-' * 65)
        print(f'  {"品种":18s} {"交易次数":>8s} {"累计盈亏":>10s} {"最后卖出":>12s}')
        print('  ' + '-' * 50)
        for h in history:
            profit_color = '+' if h['total_profit'] >= 0 else ''
            print(f'  {h["name"]:16s} {h["name"]:>2s} {h["trade_count"]:>8d} '
                  f'{profit_color}{h["total_profit"]:>9.0f} {h["last_sell_date"] or "N/A":>12s}')

    # 3. DIFv排名
    print()
    print('-' * 65)
    print('  DIFv排名')
    print('-' * 65)
    rankings = status.get('rankings', {})
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1]['rank'])
    for code, info in sorted_rankings:
        in_pos = code in {p['code'] for p in positions if not p.get('is_alternative')}
        mark = '[持]' if in_pos else '    '
        print(f'  {mark} #{info["rank"]:>2d} {info["name"]:18s} DIFv={info["score"]:>8.2f}')

    # 4. 调仓计划
    plan = status.get('rebalance_plan', [])
    print()
    print('-' * 65)
    if plan:
        print(f'  调仓计划 ({status.get("next_trade_date", "N/A")} 开盘执行)')
        print('-' * 65)
        for p in plan:
            if p['action'] == 'SELL':
                print(f'  卖出: {p["name"]:16s} ({p["code"]}) '
                      f'{p["current_weight"]:.2%} → 0.00%  原因: {p["reason"]}')
            else:
                print(f'  买入: {p["name"]:16s} ({p["code"]}) '
                      f'0.00% → {p["target_weight"]:.2%}  原因: {p["reason"]}')
    else:
        print('  调仓计划: 无操作，维持当前持仓')

    print()
    print('=' * 65)
