# -*- coding: utf-8 -*-
"""
Streamlit Config → QMT Generator Config 适配器
将 backtest_engine 的 config 格式转换为 qmt_generator 的格式
"""
import re


def _convert_code(code):
    """sh512100 → 512100.SH, sz159915 → 159915.SZ"""
    if code.startswith('sh'):
        return code[2:] + '.SH'
    elif code.startswith('sz'):
        return code[2:] + '.SZ'
    return code


def _parse_rank_formula(rank_formula):
    """
    解析排序公式，返回 sort 配置
    支持: difv, return_n, RSRS_zscore, RSRS_right_zscore, wdm_momentum, std_momentum, logbias
    """
    f = rank_formula.strip()
    
    # DIFv: (MACD_DIF(12,26,9) / ATR(26)) * 100
    if 'MACD_DIF' in f and 'ATR' in f:
        # 提取参数
        ema_short = 12
        ema_long = 26
        atr_period = 26
        m = re.search(r'MACD_DIF\((\d+),(\d+),\d+\)', f)
        if m:
            ema_short = int(m.group(1))
            ema_long = int(m.group(2))
        m2 = re.search(r'ATR\((\d+)\)', f)
        if m2:
            atr_period = int(m2.group(1))
        return {
            'indicator': 'difv',
            'direction': 'desc',
            'ema_short': ema_short,
            'ema_long': ema_long,
            'atr_period': atr_period,
        }
    
    # returns(n)
    m = re.match(r'^returns\((\d+)\)$', f)
    if m:
        return {
            'indicator': 'return_n',
            'direction': 'desc',
            'window': int(m.group(1)),
        }
    
    # RSRS_zscore / RSRS_right_zscore
    m = re.match(r'^RSRS_zscore\((\d+)\)$', f)
    if m:
        return {
            'indicator': 'rsrs_zscore',
            'direction': 'desc',
            'rsrs_period': int(m.group(1)),
        }
    m = re.match(r'^RSRS_right_zscore\((\d+)\)$', f)
    if m:
        return {
            'indicator': 'rsrs_right_zscore',
            'direction': 'desc',
            'rsrs_period': int(m.group(1)),
        }
    
    # 五斗米动量: close / MA(12) * 100 - 100 或类似
    if 'wdm_momentum' in f or ('close' in f and 'MA(' in f and '100' in f):
        return {
            'indicator': 'wdm_momentum',
            'direction': 'desc',
            'shift': 12,
            'smooth': 3,
        }
    
    # 标准动量: (close - MA(20)) / std(20) 或类似
    if 'std_momentum' in f or ('MA(' in f and 'std' in f):
        return {
            'indicator': 'std_momentum',
            'direction': 'desc',
            'window': 20,
        }
    
    # logbias
    if 'logbias' in f:
        return {
            'indicator': 'logbias',
            'direction': 'desc',
            'ema_period': 20,
            'multiplier': 100,
        }
    
    # 默认: return_20
    return {
        'indicator': 'return_n',
        'direction': 'desc',
        'window': 20,
    }


def _parse_condition(condition_str):
    """
    解析单个条件表达式，返回 qmt_generator 的 indicator/value 格式
    例如: "close > MA(5)" → {'indicator': 'close', 'op': '>', 'value': 'ma5'}
    """
    # 去除首尾空格
    c = condition_str.strip()
    
    # 匹配比较运算
    m = re.match(r'^(.*?)\s*(>=|<=|!=|==|>|<)\s*(.*)$', c)
    if not m:
        return None
    
    left = m.group(1).strip()
    op = m.group(2)
    right = m.group(3).strip()
    
    # 处理左边（indicator）
    ind = _expr_to_indicator(left)
    val = _expr_to_indicator(right)
    
    # 数值常量直接返回数值
    try:
        val = float(val)
    except ValueError:
        pass
    
    return {'indicator': ind, 'op': op, 'value': val}


def _expr_to_indicator(expr):
    """将表达式转为 qmt_generator 的指标名"""
    e = expr.strip()
    
    # 基础字段
    if e in ('close', 'open', 'high', 'low', 'volume'):
        return e
    
    # MA(n)
    m = re.match(r'^MA\((\d+)\)$', e)
    if m:
        return f'ma{int(m.group(1))}'
    
    # EMA(n)
    m = re.match(r'^EMA\((\d+)\)$', e)
    if m:
        return f'ema{int(m.group(1))}'
    
    # returns(n)
    m = re.match(r'^returns\((\d+)\)$', e)
    if m:
        return f'return_{int(m.group(1))}'
    
    # ATR(n)
    m = re.match(r'^ATR\((\d+)\)$', e)
    if m:
        return f'atr{int(m.group(1))}'
    
    # MACD_DIF(...)
    if 'MACD_DIF' in e:
        return 'difv'
    
    # BOLL_upper / BOLL_lower
    m = re.match(r'^BOLL_upper\((\d+),(\d+)\)$', e)
    if m:
        return 'boll_upper'
    m = re.match(r'^BOLL_lower\((\d+),(\d+)\)$', e)
    if m:
        return 'boll_lower'
    
    # RSRS
    if 'RSRS_zscore' in e:
        return 'rsrs_zscore'
    if 'RSRS_right_zscore' in e:
        return 'rsrs_right_zscore'
    if 'RSRS_slope' in e:
        return 'rsrs_slope'
    
    # volatility
    m = re.match(r'^volatility\((\d+)\)$', e)
    if m:
        return f'std{int(m.group(1))}'
    
    # rank / profit / hold_days
    if e in ('rank', 'profit', 'hold_days', 'buy_price'):
        return e
    
    # 复杂表达式：尝试提取
    # (MACD_DIF(12,26,9) / ATR(26)) * 100
    if 'MACD_DIF' in e and 'ATR' in e:
        return 'difv'
    
    # (close - MA(20)) / volatility(20)
    if 'close' in e and 'MA(' in e and 'volatility' in e:
        return 'std_momentum'
    
    # close / MA(12) * 100 - 100
    if 'close' in e and 'MA(' in e and '100' in e:
        return 'wdm_momentum'
    
    # 默认返回原表达式
    return e


def build_qmt_config(st_config):
    """
    将 Streamlit 的 config 转换为 qmt_generator 的 config
    
    st_config 格式: {"strategy": {universe, rank_formula, buy_rules, sell_rules, ...}}
    """
    strategy = st_config.get('strategy', {})
    
    # 标的池
    universe = strategy.get('universe', [])
    stock_tickers = [_convert_code(item['code']) for item in universe]
    
    # 替代资产
    alt = strategy.get('alternative_asset')
    bond_ticker = _convert_code(alt['code']) if alt else None
    
    # 回测参数
    bt = strategy.get('backtest', {})
    initial_capital = bt.get('initial_capital', 100000)
    fee_rate = bt.get('commission', 0.0001)
    start_date = bt.get('start_date', '2020-01-01')
    
    # 排序
    rank_formula = strategy.get('rank_formula', 'returns(20)')
    rank_direction = strategy.get('rank_direction', 'desc')
    sort_cfg = _parse_rank_formula(rank_formula)
    sort_cfg['direction'] = rank_direction
    
    # 检查是否有惩罚项
    if 'penalty(' in rank_formula:
        # 提取惩罚参数
        m = re.search(r'penalty\((\d+),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', rank_formula)
        if m:
            sort_cfg['drop_penalty'] = True
            sort_cfg['drop_threshold'] = abs(float(m.group(2)))
    
    # 买入条件
    buy_rules = strategy.get('buy_rules', [])
    buy_conditions = []
    for rule in buy_rules:
        cond = rule.get('condition', '') if isinstance(rule, dict) else str(rule)
        parsed = _parse_condition(cond)
        if parsed:
            buy_conditions.append(parsed)
    
    buy_match_mode = strategy.get('buy_match_mode', 'all')
    # qmt_generator 的 buy_mode: 'switch' = 替换模式（AND，所有条件必须满足）
    # 'free' = 自由组合（group内部AND，group之间OR）
    # 当 buy_match_mode='all' 时：所有条件AND，用 switch 模式
    # 当 buy_match_mode='any' 时：任一条件满足，用 free 模式，每个条件一个group
    if buy_match_mode == 'all':
        buy_cfg = {
            'mode': 'switch',
            'conditions': buy_conditions,
            'condition_groups': [],
        }
    else:
        buy_cfg = {
            'mode': 'free',
            'conditions': [],
            'condition_groups': [{'logic': 'AND', 'rules': [c]} for c in buy_conditions],
        }
    
    # 卖出条件
    sell_rules = strategy.get('sell_rules', [])
    sell_conditions = []
    for rule in sell_rules:
        cond = rule.get('condition', '') if isinstance(rule, dict) else str(rule)
        parsed = _parse_condition(cond)
        if parsed:
            sell_conditions.append(parsed)
    
    # 检测是否有止损条件（在sell_cfg之前计算）
    stop_loss = 0
    for cond in sell_conditions:
        if cond.get('indicator') == 'profit' and cond.get('op') == '<' and isinstance(cond.get('value'), float):
            stop_loss = abs(cond['value'])
    
    sell_match_mode = strategy.get('sell_match_mode', 'any')
    # 卖出逻辑：'all' = AND（switch模式），'any' = OR（free模式，每个条件一个group）
    if sell_match_mode == 'all':
        sell_cfg = {
            'mode': 'switch',
            'conditions': sell_conditions,
            'condition_groups': [],
            'stop_loss': stop_loss,
            'sell_if_buy_fails': False,
        }
    else:
        sell_cfg = {
            'mode': 'free',
            'conditions': [],
            'condition_groups': [{'logic': 'AND', 'rules': [c]} for c in sell_conditions],
            'stop_loss': stop_loss,
            'sell_if_buy_fails': False,
        }
    
    # 持仓
    pos = strategy.get('position', {})
    max_count = pos.get('max_count', 5)
    position_mode = pos.get('mode', 'fixed')
    qmt_position_mode = 'equal_weight' if position_mode == 'fixed' else 'adaptive'
    
    rebalance = strategy.get('rebalance', {})
    rebalance_days = rebalance.get('interval', 2)
    
    pos_cfg = {
        'mode': qmt_position_mode,
        'max_holdings': max_count,
        'position_pct': 1.0 / max_count if position_mode == 'fixed' else 0.2,
        'rebalance_days': rebalance_days,
        'new_rank_limit': 0,
    }
    
    return {
        'stock_tickers': stock_tickers,
        'bond_ticker': bond_ticker,
        'initial_capital': initial_capital,
        'fee_rate': fee_rate,
        'start_date': start_date,
        'sort': sort_cfg,
        'buy': buy_cfg,
        'sell': sell_cfg,
        'position': pos_cfg,
    }
