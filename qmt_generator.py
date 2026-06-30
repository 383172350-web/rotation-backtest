# -*- coding: utf-8 -*-
"""
QMT实盘文件生成器
=================
根据策略config字典，生成完整的QMT双模式py文件。
生成的文件完全自包含，不依赖engine模块或streamlit。
"""
from __future__ import print_function, division

import os
import re
import pprint
import datetime

# ETF中文名称映射（与engine/data_loader.py的ETF_NAMES一致）
ETF_NAMES = {
    # ---- 宽基 ----
    "510050.SH": "上证50",
    "510300.SH": "沪深300",
    "510500.SH": "中证500",
    "512100.SH": "中证1000",
    "159915.SZ": "创业板",
    "159949.SZ": "创业板50",
    "159958.SZ": "创业板ETF",
    "159922.SZ": "中证500ETF",
    "588000.SH": "科创50",
    "588220.SH": "科创100",
    "562500.SH": "中证2000",
    "159531.SZ": "中证2000ETF",
    "588250.SH": "科创医药",
    "159905.SZ": "深证红利",
    "510880.SH": "红利ETF",
    "512890.SH": "红利低波",
    # ---- 商品 / 另类 ----
    "159980.SZ": "有色ETF",
    "159981.SZ": "能源化工",
    "159985.SZ": "豆粕ETF",
    "501018.SH": "南方原油",
    "518880.SH": "黄金ETF",
    # ---- 债券 / 货币 ----
    "511880.SH": "银华日利",
    "511260.SH": "30年国债",
    "511010.SH": "国债ETF",
    "511090.SH": "30年国债ETF",
    # ---- 跨境 ----
    "513030.SH": "德国ETF",
    "513050.SH": "中概互联",
    "513100.SH": "纳指ETF",
    "513500.SH": "标普500",
    "513520.SH": "日经ETF",
    "513080.SH": "法国ETF",
    "513060.SH": "恒生ETF",
    "159740.SZ": "恒生科技",
    "513310.SH": "中韩半导体",
    # ---- 医药 ----
    "159938.SZ": "医药ETF",
    "512010.SH": "医药ETF",
    "159992.SZ": "创新药",
    "515120.SH": "创新药",
    "159898.SZ": "医疗器械",
    "159883.SZ": "医疗器械",
    # ---- 科技 / 半导体 ----
    "159509.SZ": "科创芯片",
    "588200.SH": "科创芯片",
    "512480.SH": "半导体",
    "159995.SZ": "半导体",
    "515070.SH": "人工智能",
    "159819.SZ": "人工智能",
    "515880.SH": "通信ETF",
    "515050.SH": "通信ETF国泰",
    "515000.SH": "科技ETF",
    "159611.SZ": "云计算",
    "516510.SH": "云计算",
    "515990.SH": "软件ETF",
    "159852.SZ": "软件ETF",
    "159766.SZ": "消费电子",
    "159732.SZ": "电子ETF",
    "159997.SZ": "电子ETF",
    # ---- 游戏 / 娱乐 / VR ----
    "159869.SZ": "游戏ETF",
    "516010.SH": "游戏ETF",
    "159786.SZ": "虚拟现实",
    # ---- 机器人 / 智造 ----
    "159551.SZ": "机器人",
    "516800.SH": "智能制造",
    "159380.SZ": "智能制造",
    # ---- 军工 ----
    "512660.SH": "军工ETF",
    "512560.SH": "军工ETF",
    # ---- 工业 ----
    "159967.SZ": "工业母机",
    "159667.SZ": "工业母机",
    "560913.SH": "工程机械",
    "159542.SZ": "工程机械",
    # ---- 新材料 ----
    "588010.SH": "科创新材料",
    "159871.SZ": "科创新材料",
    # ---- 新能源 ----
    "515790.SH": "光伏ETF",
    "159806.SZ": "新能源车",
    "515700.SH": "新能源车",
    "159755.SZ": "电池ETF",
    "159566.SZ": "储能电池",
    # ---- 信息 / 数据 / 电信 ----
    "515400.SH": "大数据",
    "560200.SH": "电信ETF",
    "563010.SH": "电信ETF",
    # ---- 消费 / 白酒 ----
    "159928.SZ": "消费ETF",
    "512690.SH": "白酒ETF",
    # ---- 金融 ----
    "512070.SH": "非银ETF",
    # ---- LOF ----
    "163402.SZ": "兴全趋势LOF",
    "163417.SZ": "兴全合宜LOF",
    "161903.SZ": "万家行业优选LOF",
    "162703.SZ": "广发小盘LOF",
    "161005.SZ": "富国天惠LOF",
    # ---- 补充 ----
    "159605.SZ": "中概互联ETF",
}


# ============================================================
#  分析config引用的指标
# ============================================================

_BASE_INDICATORS = {'close', 'open', 'high', 'low', 'volume'}
_KNOWN_INDICATOR_PREFIXES = (
    'ma', 'ema', 'atr', 'std', 'boll_', 'bias', 'above_ma',
    'return_', 'daily_return', 'volume_ratio', 'difv', 'dif',
    'sort_value', 'momentum_score', 'rsrs_', 'wdm_momentum',
    'big_drop_penalty', 'logbias',
)


def _collect_required_indicators(config):
    """分析config中buy/sell/sort引用的所有指标，返回需要计算的指标集合"""
    required = set()

    # 排序指标
    sort_cfg = config.get('sort', {})
    sort_indicator = sort_cfg.get('indicator', '')
    if sort_indicator:
        required.add(sort_indicator)

    # 大跌惩罚
    if sort_cfg.get('drop_penalty', False):
        required.add('big_drop_penalty')
        required.add('daily_return')

    def _extract_from_conditions(conditions):
        for cond in conditions:
            ind = cond.get('indicator', '')
            val = cond.get('value', '')
            if ind and ind not in _BASE_INDICATORS and ind != 'rank':
                required.add(ind)
            if isinstance(val, str) and val not in _BASE_INDICATORS:
                if any(val.startswith(p) for p in _KNOWN_INDICATOR_PREFIXES):
                    required.add(val)

    buy_cfg = config.get('buy', {})
    if buy_cfg.get('mode') == 'switch':
        _extract_from_conditions(buy_cfg.get('conditions', []))
    elif buy_cfg.get('mode') == 'free':
        for group in buy_cfg.get('condition_groups', []):
            _extract_from_conditions(group.get('rules', []))

    sell_cfg = config.get('sell', {})
    if sell_cfg.get('mode') == 'switch':
        _extract_from_conditions(sell_cfg.get('conditions', []))
    elif sell_cfg.get('mode') == 'free':
        for group in sell_cfg.get('condition_groups', []):
            _extract_from_conditions(group.get('rules', []))

    required.discard('')
    required -= _BASE_INDICATORS
    return required


def _build_etf_config(stock_tickers, bond_ticker):
    """构建ETF_CONFIG字典"""
    etf_config = {}
    all_tickers = list(stock_tickers)
    if bond_ticker:
        all_tickers.append(bond_ticker)
    for thscode in all_tickers:
        code, suffix = thscode.split('.')
        name_cn = ETF_NAMES.get(thscode, code)
        etf_config[code] = {"suffix": suffix, "thscode": thscode, "name_cn": name_cn}
    return etf_config


def _build_colors_fund(etf_config, bond_ticker):
    """构建颜色配置"""
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#c49c94',
    ]
    colors = {}
    idx = 0
    for code, cfg in etf_config.items():
        if cfg['thscode'] == bond_ticker:
            colors[cfg['name_cn']] = '#c49c94'
        else:
            colors[cfg['name_cn']] = default_colors[idx % len(default_colors)]
            idx += 1
    return colors


# ============================================================
#  生成_calc_signals函数体
# ============================================================

def _generate_calc_signals_code(config, required):
    """根据config生成_calc_signals函数中的指标计算代码"""
    sort_cfg = config.get('sort', {})
    sort_indicator = sort_cfg.get('indicator', '')
    direction = sort_cfg.get('direction', 'desc')
    lines = []

    # 排序指标计算
    if sort_indicator == 'difv':
        ema_short = sort_cfg.get('ema_short', 12)
        ema_long = sort_cfg.get('ema_long', 26)
        atr_period = sort_cfg.get('atr_period', 26)
        lines.append(f"        df['ema{ema_short}'] = c.ewm(span={ema_short}, adjust=False).mean()")
        lines.append(f"        df['ema{ema_long}'] = c.ewm(span={ema_long}, adjust=False).mean()")
        lines.append(f"        df['dif'] = _macd_dif(c, {ema_short}, {ema_long})")
        lines.append(f"        df['atr{atr_period}'] = _atr(df['high'], df['low'], c, {atr_period})")
        lines.append(f"        df['difv'] = (df['dif'] / df['atr{atr_period}'] * 100).replace([np.inf, -np.inf], np.nan)")
        lines.append("        df['sort_value'] = df['difv']")

    elif sort_indicator == 'return_n':
        window = sort_cfg.get('window', 20)
        lines.append(f"        df['return_{window}'] = c.pct_change({window})")
        lines.append(f"        df['sort_value'] = df['return_{window}']")

    elif sort_indicator == 'momentum_score':
        window = sort_cfg.get('window', 20)
        lines.append(f"        df = _calc_momentum_score(df, window={window})")
        lines.append("        df['sort_value'] = df['momentum_score']")

    elif sort_indicator == 'logbias':
        ema_period = sort_cfg.get('ema_period', 20)
        multiplier = sort_cfg.get('multiplier', 100)
        lines.append(f"        df = _calc_logbias(df, ema_period={ema_period}, multiplier={multiplier})")
        lines.append("        df['sort_value'] = df['logbias']")

    elif sort_indicator == 'std_momentum':
        window = sort_cfg.get('window', 20)
        lines.append(f"        df['ma{window}'] = c.rolling({window}).mean()")
        lines.append(f"        df['std{window}'] = c.rolling({window}).std()")
        lines.append(f"        df['std_momentum'] = ((c - df['ma{window}']) / df['std{window}']).replace([np.inf, -np.inf], np.nan)")
        lines.append("        df['sort_value'] = df['std_momentum']")

    elif sort_indicator == 'wdm_momentum':
        shift = sort_cfg.get('shift', 12)
        smooth = sort_cfg.get('smooth', 3)
        lines.append(f"        ref_avg = c.shift({shift}).rolling(window={smooth}).mean()")
        lines.append(f"        df['wdm_momentum'] = (c / ref_avg * 100 - 100).replace([np.inf, -np.inf], np.nan)")
        lines.append("        df['sort_value'] = df['wdm_momentum']")

    # 大跌惩罚
    if sort_cfg.get('drop_penalty', False):
        threshold = sort_cfg.get('drop_threshold', 0.03)
        lines.append(f"        # 大跌惩罚")
        lines.append(f"        df['big_drop_penalty'] = (")
        lines.append(f"            (df['daily_return'] < -{threshold})")
        lines.append(f"            .rolling(window=3, min_periods=1)")
        lines.append(f"            .max().fillna(0).astype(bool)")
        lines.append(f"        )")
        lines.append(f"        df.loc[df['big_drop_penalty'], 'sort_value'] = -300")

    # 其他需要的基础指标（排除已被排序指标计算过的）
    computed = set()
    if sort_indicator == 'difv':
        computed.update({'ema12', 'ema26', 'atr26', 'dif', 'difv'})
    elif sort_indicator == 'std_momentum':
        computed.update({'std_momentum'})

    # daily_return
    if 'daily_return' in required and 'daily_return' not in computed:
        lines.append("        df['daily_return'] = c.pct_change()")
        computed.add('daily_return')

    # MA系列
    for period in [5, 10, 20, 60, 120, 250]:
        name = f'ma{period}'
        if name in required and name not in computed:
            lines.append(f"        df['ma{period}'] = c.rolling({period}).mean()")
            computed.add(name)
        # above_ma需要先计算ma
        if f'above_ma{period}' in required and name not in computed:
            lines.append(f"        df['ma{period}'] = c.rolling({period}).mean()")
            computed.add(name)
        # bias需要先计算ma
        if f'bias{period}' in required and name not in computed:
            lines.append(f"        df['ma{period}'] = c.rolling({period}).mean()")
            computed.add(name)

    # EMA系列
    for period in [12, 26, 20]:
        name = f'ema{period}'
        if name in required and name not in computed:
            lines.append(f"        df['ema{period}'] = c.ewm(span={period}, adjust=False).mean()")
            computed.add(name)

    # ATR系列
    for period in [26]:
        name = f'atr{period}'
        if name in required and name not in computed:
            lines.append(f"        df['atr{period}'] = _atr(df['high'], df['low'], c, {period})")
            computed.add(name)

    # return_N系列
    for ind in required:
        m = re.match(r'^return_(\d+)d?$', ind)
        if m and ind not in computed:
            n = int(m.group(1))
            lines.append(f"        df['{ind}'] = c.pct_change({n})")
            computed.add(ind)

    # volume_ratio
    if 'volume_ratio' in required and 'volume_ratio' not in computed:
        lines.append("        df['volume_ratio'] = (df['volume'] / df['volume'].rolling(7).mean()).replace([np.inf, -np.inf], np.nan)")
        computed.add('volume_ratio')

    # std系列
    for period in [20, 60]:
        name = f'std{period}'
        if name in required and name not in computed:
            lines.append(f"        df['std{period}'] = c.rolling({period}).std()")
            computed.add(name)

    # boll
    if any(k in required for k in ('boll_upper', 'boll_mid', 'boll_lower')):
        lines.append("        boll_period = 17")
        lines.append("        boll_std_n = 2")
        lines.append("        boll_ma = c.rolling(boll_period).mean()")
        lines.append("        boll_s = c.rolling(boll_period).std()")
        lines.append("        df['boll_mid'] = boll_ma")
        lines.append("        df['boll_upper'] = boll_ma + boll_std_n * boll_s")
        lines.append("        df['boll_lower'] = boll_ma - boll_std_n * boll_s")

    # bias系列
    for period in [5, 10, 20, 60]:
        name = f'bias{period}'
        if name in required and name not in computed:
            ma_name = f'ma{period}'
            if ma_name not in computed:
                lines.append(f"        df['ma{period}'] = c.rolling({period}).mean()")
                computed.add(ma_name)
            lines.append(f"        df['bias{period}'] = c / df['ma{period}'] - 1")
            computed.add(name)

    # above_ma系列
    for period in [5, 10, 20]:
        name = f'above_ma{period}'
        if name in required and name not in computed:
            if period == 10:
                lines.append(f"        df['above_ma{period}'] = (c >= df['ma{period}']).fillna(False)")
            else:
                lines.append(f"        df['above_ma{period}'] = (c > df['ma{period}']).fillna(False)")
            computed.add(name)

    # RSRS
    if 'rsrs_strength' in required or 'rsrs_pass' in required:
        lines.append("        df = _calc_rsrs(df)")
        computed.update({'rsrs_strength', 'rsrs_pass'})

    # momentum_score（非排序场景）
    if 'momentum_score' in required and 'momentum_score' not in computed:
        lines.append("        df = _calc_momentum_score(df)")
        computed.add('momentum_score')

    # big_drop_penalty（非排序场景）
    if 'big_drop_penalty' in required and 'big_drop_penalty' not in computed:
        if 'daily_return' not in computed:
            lines.append("        df['daily_return'] = c.pct_change()")
            computed.add('daily_return')
        lines.append("        df['big_drop_penalty'] = (")
        lines.append("            (df['daily_return'] < -0.03)")
        lines.append("            .rolling(window=3, min_periods=1)")
        lines.append("            .max().fillna(0).astype(bool)")
        lines.append("        )")
        computed.add('big_drop_penalty')

    # logbias（非排序场景）
    if 'logbias' in required and 'logbias' not in computed:
        lines.append("        df = _calc_logbias(df)")
        computed.add('logbias')

    return '\n'.join(lines)


# ============================================================
#  生成辅助函数代码（需要包含在生成文件中）
# ============================================================

_MOMENTUM_SCORE_FUNC = r'''
def _calc_momentum_score(df, window=20):
    """对数价格加权回归斜率年化 * R²"""
    close = df['close']
    n = len(df)
    momentum_scores = np.full(n, np.nan)
    log_prices = np.log(close.values)
    x_vals = np.arange(window, dtype=float)
    weights = np.linspace(1, 2, window)
    w_mean = np.average(x_vals, weights=weights)
    w_std = np.sqrt(np.average((x_vals - w_mean) ** 2, weights=weights))
    w_x_centered = (x_vals - w_mean) / w_std if w_std > 0 else x_vals - w_mean
    for i in range(window - 1, n):
        y = log_prices[i - window + 1:i + 1]
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            continue
        y_mean = np.average(y, weights=weights)
        y_centered = y - y_mean
        slope = (np.sum(weights * w_x_centered * y_centered)
                 / np.sum(weights * w_x_centered ** 2) / w_std) if w_std > 0 else 0
        intercept = y_mean - slope * w_mean
        predicted = slope * x_vals + intercept
        ss_res = np.sum(weights * (y - predicted) ** 2)
        ss_tot = np.sum(weights * (y - y_mean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        annualized_return = np.exp(slope * 250) - 1
        ms = annualized_return * r2
        prices = close.values[i - window + 1:i + 1]
        if len(prices) >= 4:
            if min(prices[-1] / prices[-2],
                   prices[-2] / prices[-3],
                   prices[-3] / prices[-4]) < 0.95:
                ms = -8
        momentum_scores[i] = ms
    df['momentum_score'] = momentum_scores
    return df
'''.strip()

_LOGBIAS_FUNC = r'''
def _calc_logbias(df, ema_period=20, multiplier=100):
    """logbias: (ln(close) - EMA(ln(close), N)) * multiplier"""
    close = df['close']
    log_close = np.log(close)
    ema_log = log_close.ewm(span=ema_period, adjust=False).mean()
    df['logbias'] = (log_close - ema_log) * multiplier
    return df
'''.strip()

_RSRS_FUNC = r'''
def _calc_rsrs(df, rsrs_days=18, rsrs_window=20, lookback_days=250):
    """RSRS标准化强度指标"""
    n = len(df)
    low_vals = df['low'].values
    high_vals = df['high'].values
    all_slopes = np.full(n, np.nan)
    for i in range(rsrs_window - 1, n):
        lv = low_vals[i - rsrs_window + 1:i + 1]
        hv = high_vals[i - rsrs_window + 1:i + 1]
        if np.std(lv) == 0 or np.std(hv) == 0:
            continue
        try:
            all_slopes[i] = np.polyfit(lv, hv, 1)[0]
        except Exception:
            pass
    rsrs_strengths = np.full(n, np.nan)
    rsrs_pass_list = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < rsrs_days - 1:
            continue
        cur_slope = all_slopes[i]
        if np.isnan(cur_slope):
            continue
        start_j = max(rsrs_window - 1, i - lookback_days + 1)
        slope_window = all_slopes[start_j:i + 1]
        valid_slopes = slope_window[~np.isnan(slope_window)]
        if len(valid_slopes) < 2:
            continue
        mean_slope = np.mean(valid_slopes)
        std_slope = np.std(valid_slopes)
        beta = mean_slope - 2 * std_slope
        strength = (cur_slope - beta) / abs(beta) if beta != 0 else 0
        rsrs_pass = cur_slope > beta
        rsrs_pass_list[i] = rsrs_pass
        if rsrs_pass:
            rsrs_strengths[i] = strength
    df['rsrs_strength'] = rsrs_strengths
    df['rsrs_pass'] = rsrs_pass_list
    return df
'''.strip()


# ============================================================
#  需要包含的辅助函数
# ============================================================

def _get_helper_funcs(required, sort_indicator):
    """返回需要包含在生成文件中的辅助函数代码列表"""
    funcs = []
    # _macd_dif和_atr总是包含（基础辅助）
    funcs.append("""\
def _macd_dif(close, fast=12, slow=26):
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()

def _atr(high, low, close, period=26):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()""")

    if 'momentum_score' in required or sort_indicator == 'momentum_score':
        funcs.append(_MOMENTUM_SCORE_FUNC)

    if 'logbias' in required or sort_indicator == 'logbias':
        funcs.append(_LOGBIAS_FUNC)

    if 'rsrs_strength' in required or 'rsrs_pass' in required:
        funcs.append(_RSRS_FUNC)

    return '\n\n'.join(funcs)


# ============================================================
#  主生成函数
# ============================================================

def generate_qmt_file(config, output_path, strategy_name,
                      account_id='520000249836', real_capital=10000.0,
                      script_dir=None, run_mode='live'):
    """
    根据策略config生成QMT实盘文件

    Parameters
    ----------
    config : dict
        策略配置，格式与streamlit_app.py的collect_current_config()输出一致
    output_path : str
        输出文件路径
    strategy_name : str
        策略名称，用于文件头和日志
    account_id : str
        QMT账号ID
    real_capital : float
        实盘资金
    script_dir : str or None
        脚本目录，None则自动生成
    run_mode : str
        运行模式 "live" 或 "backtest"
    """
    # ---- 解析config ----
    stock_tickers = config.get('stock_tickers', [])
    bond_ticker = config.get('bond_ticker')
    initial_capital = config.get('initial_capital', 1000000)
    fee_rate = config.get('fee_rate', 0.0001)
    start_date = config.get('start_date', '2020-01-02')

    sort_cfg = config.get('sort', {})
    sort_indicator = sort_cfg.get('indicator', 'difv')
    sort_direction = sort_cfg.get('direction', 'desc')

    buy_cfg = config.get('buy', {})
    buy_mode = buy_cfg.get('mode', 'switch')
    buy_conditions = buy_cfg.get('conditions', [])
    buy_condition_groups = buy_cfg.get('condition_groups', [])

    sell_cfg = config.get('sell', {})
    sell_mode = sell_cfg.get('mode', 'switch')
    sell_conditions = sell_cfg.get('conditions', [])
    sell_condition_groups = sell_cfg.get('condition_groups', [])
    stop_loss = sell_cfg.get('stop_loss', 0)
    sell_if_buy_fails = sell_cfg.get('sell_if_buy_fails', False)

    pos_cfg = config.get('position', {})
    position_mode = pos_cfg.get('mode', 'equal_weight')
    max_holdings = pos_cfg.get('max_holdings', 5)
    position_pct = pos_cfg.get('position_pct', 0.20)
    rebalance_days = pos_cfg.get('rebalance_days', 2)
    new_rank_limit = pos_cfg.get('new_rank_limit', 0)

    # ---- 计算衍生值 ----
    required = _collect_required_indicators(config)
    etf_config = _build_etf_config(stock_tickers, bond_ticker)
    colors_fund = _build_colors_fund(etf_config, bond_ticker)

    if script_dir is None:
        script_dir = f"C:\\{strategy_name}"

    # ---- 生成文件内容 ----
    lines = []

    # === 1. 文件头 ===
    lines.append('# -*- coding: utf-8 -*-')
    lines.append('"""')
    lines.append(f'============================================================')
    lines.append(f'{strategy_name} - 全自动交易机器人')
    lines.append(f'策略引擎：数据加载/回测/计划生成')
    lines.append(f'下单执行：QMT下单/定时任务')
    lines.append(f'自动生成时间：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'============================================================')
    lines.append('"""')
    lines.append('from __future__ import print_function, division')
    lines.append('import pandas as pd')
    lines.append('import numpy as np')
    lines.append('import os')
    lines.append('import csv')
    lines.append('import math')
    lines.append('import time')
    lines.append('import builtins as _builtins')
    lines.append('import datetime')
    lines.append('import codecs')
    lines.append('import warnings')
    lines.append('import traceback')
    lines.append('import re')

    # === 2. matplotlib配置 ===
    lines.append(r'''
# ====== matplotlib 导入（无头后端，QMT兼容）======
try:
    import matplotlib as _mpl
    _mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.font_manager import FontProperties
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    _mpl_available = True
except ImportError:
    _mpl_available = False

if _mpl_available:
    try:
        import matplotlib.font_manager as _fm
        _font_candidates = ['Microsoft YaHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS']
        _available_fonts = [f.name for f in _fm.fontManager.ttflist]
        _selected_font = None
        for fc in _font_candidates:
            if fc in _available_fonts:
                _selected_font = fc
                break
        if _selected_font:
            _mpl.rcParams['font.family'] = [_selected_font, 'sans-serif']
        else:
            _mpl.rcParams['font.family'] = ['sans-serif']
        _mpl.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    fp_title = FontProperties(family=_mpl.rcParams['font.family'], size=12, weight='bold')
    fp_subtitle = FontProperties(family=_mpl.rcParams['font.family'], size=11, weight='bold')
    fp_label = FontProperties(family=_mpl.rcParams['font.family'], size=10)
    fp_banner = FontProperties(family=_mpl.rcParams['font.family'], size=10, weight='bold')
    fp_legend = FontProperties(family=_mpl.rcParams['font.family'], size=7)
    fp_legend_title = FontProperties(family=_mpl.rcParams['font.family'], size=7, weight='bold')
    fp_tick = FontProperties(family=_mpl.rcParams['font.family'], size=9)
    fp_bar = FontProperties(family=_mpl.rcParams['font.family'], size=9, weight='bold')
''')
    lines.append("warnings.filterwarnings('ignore')")

    # === 3. 配置区 ===
    lines.append('')
    lines.append('# ========================================')
    lines.append('#  配置区')
    lines.append('# ========================================')
    # 在生成代码前，确保路径中的特殊字符被正确转义
    import json
    safe_script_dir = json.dumps(script_dir)  # 自动转义反斜杠等
    lines.append(f'SCRIPT_DIR = {safe_script_dir}')
    lines.append('')
    lines.append(f'DATA_ROOT = r"D:\\qmt_data\\ETF\\1d"')
    lines.append('')
    # 文件路径使用strategy_name
    safe_name = re.sub(r'[^\w]', '_', strategy_name)
    lines.append(f'TRADE_PLAN_CSV = os.path.join(SCRIPT_DIR, "{safe_name}_trade_plan.csv")')
    lines.append(f'SIGNAL_REPORT_PATH = os.path.join(SCRIPT_DIR, "{safe_name}_signal_report.txt")')
    lines.append(f'LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "auto_trader.log")')
    lines.append(f'TRADE_RECORDS_PATH = os.path.join(SCRIPT_DIR, "{safe_name}_trade_records.csv")')
    lines.append(f'DASHBOARD_PATH = os.path.join(SCRIPT_DIR, "{strategy_name}_监控面板.png")')
    lines.append('')
    lines.append(f"ACCOUNT_ID = '{account_id}'")
    lines.append(f'REAL_CAPITAL = {real_capital}')
    lines.append(f'INITIAL_CAPITAL = {float(initial_capital)}')
    lines.append('')
    lines.append(f'RUN_MODE = "{run_mode}"')
    lines.append('')
    lines.append('TASK_PRE_MARKET = "09:25:00"')
    lines.append('TASK_EXEC_TIME = "09:30:00"')
    lines.append('TASK_INTRADAY_TIME = "14:50:00"')
    lines.append('MARKET_OPEN = "09:30:00"')
    lines.append('MARKET_CLOSE = "15:00:00"')
    lines.append('')
    lines.append(f'MAX_HOLDINGS = {max_holdings}')
    lines.append(f'POSITION_PCT = {position_pct}')
    lines.append(f'REBALANCE_DAYS = {rebalance_days}')
    lines.append(f'NEW_RANK_LIMIT = {new_rank_limit}')
    lines.append(f'STRATEGY_START = "{start_date}"')
    lines.append(f'FEE_RATE = {fee_rate}')
    lines.append(f'POSITION_MODE = "{position_mode}"')
    lines.append(f'SORT_INDICATOR = "{sort_indicator}"')
    lines.append(f'SORT_DIRECTION = "{sort_direction}"')
    lines.append('')

    # colors_fund
    lines.append('# 颜色配置')
    lines.append(f'colors_fund = {pprint.pformat(colors_fund, width=120)}')
    lines.append('')

    lines.append('ENABLE_SCHEDULED_MODE = True')
    lines.append('ENABLE_INTRADAY_MODE = False')
    lines.append('')

    # ETF_CONFIG
    lines.append('ETF_CONFIG = {')
    for code, cfg in etf_config.items():
        lines.append(f'    "{code}": {{"suffix": "{cfg["suffix"]}", "thscode": "{cfg["thscode"]}", "name_cn": "{cfg["name_cn"]}"}},')
    lines.append('}')
    lines.append('')

    # BOND_TICKER
    lines.append(f'BOND_TICKER = "{bond_ticker}"' if bond_ticker else 'BOND_TICKER = None')
    lines.append(f'STOCK_TICKERS = [v["thscode"] for k, v in ETF_CONFIG.items() if v["thscode"] != BOND_TICKER]')
    lines.append('ALL_TICKERS = [v["thscode"] for v in ETF_CONFIG.values()]')
    lines.append('TICKER_NAMES = {v["thscode"]: v["name_cn"] for v in ETF_CONFIG.values()}')
    lines.append('')

    lines.append('BUY_ORDER_TYPE = 23')
    lines.append('SELL_ORDER_TYPE = 24')
    lines.append('ACCOUNT_TYPE = 1101')
    lines.append(f'BUY_TAG = "{safe_name}_buy"')
    lines.append(f'SELL_TAG = "{safe_name}_sell"')
    lines.append('PRICE_TYPE = 14')
    lines.append('ETF_MIN_UNIT = 100')
    lines.append('')

    # === 4. 买入/卖出条件常量 ===
    lines.append('# ========================================')
    lines.append('#  买入/卖出条件常量')
    lines.append('# ========================================')
    lines.append('')
    lines.append(f'BUY_MODE = "{buy_mode}"')
    lines.append(f'BUY_CONDITIONS = {pprint.pformat(buy_conditions, width=120)}')
    lines.append(f'BUY_CONDITION_GROUPS = {pprint.pformat(buy_condition_groups, width=120)}')
    lines.append('')
    lines.append(f'SELL_MODE = "{sell_mode}"')
    lines.append(f'SELL_CONDITIONS = {pprint.pformat(sell_conditions, width=120)}')
    lines.append(f'SELL_CONDITION_GROUPS = {pprint.pformat(sell_condition_groups, width=120)}')
    lines.append(f'SELL_STOP_LOSS = {stop_loss}')
    lines.append(f'SELL_IF_BUY_FAILS = {sell_if_buy_fails}')
    lines.append('')

    # sort config参数
    lines.append('# 排序参数')
    for key, val in sort_cfg.items():
        if isinstance(val, str):
            lines.append(f'SORT_{key.upper()} = "{val}"')
        else:
            lines.append(f'SORT_{key.upper()} = {val}')
    lines.append('')

    lines.append('DEBUG = True')
    lines.append('ECHO_TO_CONSOLE = True')
    lines.append('LOG_TO_FILE = True')
    lines.append('')
    lines.append('_log_file = None')
    lines.append('_last_trading_day = None')
    lines.append('_intraday_done = False')
    lines.append('_pre_market_done = False')
    lines.append('_exec_done = False')
    lines.append('')

    # === 5. 日志系统 ===
    lines.append(r'''
# ========================================
#  日志系统
# ========================================

def _open_log():
    global _log_file
    if _log_file is None and LOG_TO_FILE and LOG_FILE_PATH:
        try:
            d = os.path.dirname(LOG_FILE_PATH)
            if d and not os.path.exists(d):
                os.makedirs(d)
            _log_file = codecs.open(LOG_FILE_PATH, "a", encoding="gbk", errors="replace")
        except Exception:
            pass

def _close_log():
    global _log_file
    if _log_file:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None

def _write_log(text):
    _open_log()
    if _log_file:
        try:
            _log_file.write(text)
            _log_file.flush()
        except Exception:
            pass

def print(*args, **kwargs):
    if ECHO_TO_CONSOLE:
        _builtins.print(*args, **kwargs)
    try:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        line = sep.join([str(x) for x in args]) + end
        _write_log(line)
    except Exception:
        pass

def log_critical(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] [CRITICAL] {}".format(ts, msg))

def log_info(msg):
    if DEBUG:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[{}] [INFO] {}".format(ts, msg))

def log_error(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] [ERROR] {}".format(ts, msg))
''')

    # === 6. 工具函数 ===
    lines.append(r'''
# ========================================
#  工具函数
# ========================================

def _normalize_code(code):
    s = str(code).strip()
    if not s: return s
    if "." in s: return s
    if s.startswith(("5", "6", "9", "11")): return s + ".SH"
    return s + ".SZ"

def _today_str():
    return datetime.datetime.now().strftime("%Y%m%d")

def _now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

def _is_trading_time():
    return MARKET_OPEN <= _now_str() <= MARKET_CLOSE
''')

    # === 7. 辅助函数 ===
    lines.append('')
    lines.append('# ========================================')
    lines.append('#  辅助函数')
    lines.append('# ========================================')
    lines.append('')
    helper_code = _get_helper_funcs(required, sort_indicator)
    lines.append(helper_code)
    lines.append('')

    # === 8. 数据模块 ===
    lines.append(r'''
# ========================================
#  数据模块
# ========================================

def _detect_pkl_dir():
    candidates = [
        r"D:\qmt_data\ETF\1d", r"C:\qmt_data\ETF\1d", DATA_ROOT,
        r"D:\国金QMT投研端\userdata_mini\ETF\1d",
        r"C:\国金QMT投研端\userdata_mini\ETF\1d",
        r"D:\XtQuant\data\ETF\1d", r"C:\XtQuant\data\ETF\1d",
    ]
    for path in candidates:
        if os.path.exists(path):
            pkls = [f for f in os.listdir(path) if f.endswith("_1d.pkl")]
            if len(pkls) > 3:
                log_info("检测到pkl目录: {} ({}个pkl)".format(path, len(pkls)))
                return path
    search_roots = ["C:\\", "D:\\"]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) > 10:
                del dirnames[:]
                continue
            for f in filenames:
                if f.endswith("_1d.pkl") and ("510300" in f or "518880" in f):
                    log_info("搜索发现pkl目录: {}".format(dirpath))
                    return dirpath
    log_info("未找到pkl目录，使用默认: {}".format(DATA_ROOT))
    return DATA_ROOT

def _load_data_from_pkl(pkl_dir):
    log_info("从pkl加载数据: {}".format(pkl_dir))
    dfs = []
    for code, cfg in ETF_CONFIG.items():
        pkl_file = os.path.join(pkl_dir, "{}_{}_1d.pkl".format(code, cfg['suffix']))
        if not os.path.exists(pkl_file):
            log_info("  {}: 文件不存在 {}".format(code, pkl_file))
            continue
        try:
            df = pd.read_pickle(pkl_file).reset_index()
            df['time'] = pd.to_datetime(df['stime'].astype(str), format='%Y%m%d')
            df['date'] = df['time'].dt.strftime('%Y-%m-%d')
            df['thscode'] = cfg['thscode']
            df['thsname_cn'] = cfg['name_cn']
            df = df[['open', 'high', 'low', 'close', 'volume', 'thscode', 'time', 'thsname_cn', 'date']]
            dfs.append(df)
            log_info("  {}: {}条".format(code, len(df)))
        except Exception as e:
            log_error("  {}: 读取失败 - {}".format(code, e))
    if not dfs:
        return None
    result = pd.concat(dfs, ignore_index=True).sort_values(['thscode', 'date']).reset_index(drop=True)
    log_info("pkl加载完成: 共{}条".format(len(result)))
    return result

def _download_and_save_daily_data(ContextInfo):
    log_info("=" * 60)
    log_info("下载并保存日线数据（1d）到指定路径")
    log_info("=" * 60)
    start_date = "20150101"
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    for code, cfg in ETF_CONFIG.items():
        full_code = code + "." + cfg['suffix']
        try:
            download_history_data(full_code, "1d", start_date, end_date)
            log_info("  {}: 下载到缓存完成".format(full_code))
            try:
                data = None
                if hasattr(ContextInfo, 'get_market_data_ex'):
                    data = ContextInfo.get_market_data_ex(
                        ['open', 'high', 'low', 'close', 'volume'], [full_code],
                        period='1d', start_time=start_date + '000000', end_time=end_date + '150000',
                        count=-1, dividend_type='front', fill_data=True)
                elif 'get_market_data_ex' in dir():
                    data = get_market_data_ex(
                        ['open', 'high', 'low', 'close', 'volume'], [full_code],
                        period='1d', start_time=start_date + '000000', end_time=end_date + '150000',
                        count=-1, dividend_type='front', fill_data=True)
                else:
                    log_info("  {}: get_market_data_ex 不可用，跳过保存".format(full_code))
                    continue
                if data and full_code in data:
                    df = data[full_code]
                    if df is not None and not df.empty:
                        d = DATA_ROOT
                        if not os.path.exists(d):
                            os.makedirs(d)
                        file_name = full_code.replace('.', '_') + "_1d.pkl"
                        file_path = os.path.join(d, file_name)
                        df.to_pickle(file_path)
                        log_info("  {}: 已保存pkl -> {} ({}条)".format(full_code, file_path, str(len(df))))
                    else:
                        log_info("  {}: 缓存数据为空".format(full_code))
                else:
                    log_info("  {}: 无法从缓存读取".format(full_code))
            except Exception as e2:
                log_info("  {}: 读取缓存失败 - {}".format(full_code, str(e2)))
        except Exception as e:
            log_error("  {}: 下载失败 - {}".format(full_code, str(e)))
    log_info("日线数据下载并保存完成")

def _build_data_dict(data_df):
    d = {}
    for t in data_df['thscode'].unique():
        df = data_df[data_df['thscode'] == t].copy().set_index('time').sort_index()
        d[t] = df[['open', 'high', 'low', 'close', 'volume']]
    return d

def _align_dates(data_dict):
    common = None
    for t, df in data_dict.items():
        common = set(df.index) if common is None else common.intersection(set(df.index))
    common = sorted(list(common))
    for t in data_dict:
        data_dict[t] = data_dict[t].loc[data_dict[t].index.isin(common)]
    return data_dict, common
''')

    # === 9. _calc_signals ===
    lines.append('')
    lines.append('# ========================================')
    lines.append('#  策略模块')
    lines.append('# ========================================')
    lines.append('')

    calc_signals_body = _generate_calc_signals_code(config, required)
    lines.append(f'''def _calc_signals(data_dict, tickers):
    sig = {{}}
    for t in tickers:
        df = data_dict[t].copy()
        c = df['close']
{calc_signals_body}
        sig[t] = df
    return sig
''')

    # === 10. 通用条件评估函数 ===
    lines.append(r'''
# ========================================
#  通用条件评估函数
# ========================================

def _eval_condition(row, indicator, op, value, rank_map=None, ticker=None):
    """通用条件评估"""
    if indicator == 'rank':
        if rank_map is None or ticker is None:
            return True
        val = rank_map.get(ticker, 999)
    else:
        val = row.get(indicator, np.nan)
    if pd.isna(val):
        return True  # NaN跳过
    val = float(val)

    if op == 'is_true':
        return bool(val)
    if op == 'is_false':
        return not bool(val)

    # 解析value
    if isinstance(value, str) and hasattr(row, 'index') and value in row.index:
        rhs = row.get(value, np.nan)
        if pd.isna(rhs):
            return True
        rhs = float(rhs)
    elif isinstance(value, str) and isinstance(row, dict) and value in row:
        rhs = row.get(value, np.nan)
        if pd.isna(rhs):
            return True
        rhs = float(rhs)
    else:
        try:
            rhs = float(value)
        except (ValueError, TypeError):
            return True

    if op == '>': return val > rhs
    if op == '>=': return val >= rhs
    if op == '<': return val < rhs
    if op == '<=': return val <= rhs
    if op == '==': return abs(val - rhs) < 1e-9
    if op == '!=': return abs(val - rhs) >= 1e-9
    return False

def _check_buy_conditions(row, rank_map=None, ticker=None):
    """买入条件检查"""
    if BUY_MODE == "switch":
        # 排除条件
        for cond in BUY_CONDITIONS:
            if not cond.get('enabled', True):
                continue
            if cond.get('exclude', False):
                if _eval_condition(row, cond['indicator'], cond['op'], cond['value'], rank_map, ticker):
                    return False
        # 普通条件（AND）
        for cond in BUY_CONDITIONS:
            if not cond.get('enabled', True) or cond.get('exclude', False):
                continue
            if not _eval_condition(row, cond['indicator'], cond['op'], cond['value'], rank_map, ticker):
                return False
        return True
    else:  # free模式
        for group in BUY_CONDITION_GROUPS:
            logic = group.get('logic', 'AND').upper()
            if logic == 'ALL':
                logic = 'AND'
            rules = group.get('rules', [])
            results = [_eval_condition(row, r['indicator'], r['op'], r['value'], rank_map, ticker) for r in rules]
            if logic == 'AND' and all(results):
                return True
            if logic == 'OR' and any(results):
                return True
        return False

def _check_sell_conditions(row, rank_map=None, ticker=None, buy_price=None, is_rebalance_day=False):
    """卖出条件检查，返回 (should_sell, reason)"""
    # 止损
    if SELL_STOP_LOSS > 0 and buy_price is not None and buy_price > 0:
        current = row.get('close', np.nan)
        if pd.notna(current) and (current - buy_price) / buy_price < -SELL_STOP_LOSS:
            return True, "止损"
    # sell_if_buy_fails (仅轮动日检查)
    if SELL_IF_BUY_FAILS and is_rebalance_day:
        if not _check_buy_conditions(row, rank_map, ticker):
            return True, "不满足买入条件"
    # sell conditions
    if SELL_MODE == "switch":
        for cond in SELL_CONDITIONS:
            if not cond.get('enabled', True):
                continue
            if _eval_condition(row, cond['indicator'], cond['op'], cond['value'], rank_map, ticker):
                name = cond.get('name', "{} {} {}".format(cond['indicator'], cond['op'], cond['value']))
                return True, name
    else:  # free模式
        for group in SELL_CONDITION_GROUPS:
            logic = group.get('logic', 'AND').upper()
            if logic == 'ALL':
                logic = 'AND'
            rules = group.get('rules', [])
            results = [_eval_condition(row, r['indicator'], r['op'], r['value'], rank_map, ticker) for r in rules]
            if (logic == 'AND' and all(results)) or (logic == 'OR' and any(results)):
                return True, "条件触发"
    return False, None
''')

    # === 11. _run_backtest ===
    lines.append(r'''
# ========================================
#  回测引擎
# ========================================

def _calc_rank_map(signals, stock_tickers, date):
    """计算某日排名，返回 {ticker: rank}"""
    descending = (SORT_DIRECTION == 'desc')
    sort_values = {}
    for t in stock_tickers:
        if t not in signals:
            continue
        df = signals[t]
        if date not in df.index:
            continue
        v = df.loc[date, 'sort_value'] if 'sort_value' in df.columns else np.nan
        if pd.notna(v):
            sort_values[t] = v
    ranked = sorted(sort_values.items(), key=lambda x: x[1], reverse=descending)
    rank_map = {t: i + 1 for i, (t, _) in enumerate(ranked)}
    return rank_map, sort_values

def _run_backtest(signals, stock_tickers, bond_ticker, all_tickers, dates, initial_capital, start_date=None):
    if start_date:
        start_date = pd.Timestamp(start_date)
        dates = [d for d in dates if d >= start_date]
    if not dates:
        raise ValueError("起始日期后无数据")

    # 对齐：从所有标的close>0的第一天开始
    valid_start = None
    for d in dates:
        all_positive = True
        for t in stock_tickers:
            if t in signals and d in signals[t].index:
                c = signals[t].loc[d, 'close']
                if pd.isna(c) or c <= 0:
                    all_positive = False
                    break
            else:
                all_positive = False
                break
        if all_positive:
            valid_start = d
            break
    if valid_start:
        dates = [d for d in dates if d >= valid_start]
        log_info("回测起始日(所有标的close>0): {}".format(valid_start.date()))
    if not dates:
        raise ValueError("无有效起始日期")

    cash = float(initial_capital)
    holdings = {}
    nav_history = []
    trade_log = []
    hold_history = []
    rebalance_dates = set([dates[i] for i in range(0, len(dates), REBALANCE_DAYS)])

    for i, date in enumerate(dates):
        nav = cash
        for t, pos in holdings.items():
            if t in signals and date in signals[t].index:
                c = signals[t].loc[date, 'close']
                nav += pos['shares'] * (c if c > 0 else 0)
        nav_history.append({'date': date, 'nav': nav})
        active = [t for t in holdings if t != bond_ticker]
        hold_history.append({'date': date, 'holdings': len(active), 'hold_tickers': active})

        if i + 1 >= len(dates):
            continue
        next_date = dates[i + 1]

        # 计算排名
        rank_map, sort_values = _calc_rank_map(signals, stock_tickers, date)

        # ---- 每日检查卖出 ----
        sell_list = []
        for t in list(holdings.keys()):
            if t == bond_ticker:
                continue
            if t not in signals or date not in signals[t].index:
                continue
            s = signals[t].loc[date]
            bp = holdings[t]['cost']
            rank = rank_map.get(t, 99)
            row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
            row_dict['rank'] = rank
            row_dict['buy_price'] = bp
            should_sell, reason = _check_sell_conditions(row_dict, rank_map, t, bp, is_rebalance_day=(date in rebalance_dates))
            if should_sell:
                sell_list.append((t, reason))

        # single模式换仓：轮动日找到满足买入条件的最佳候选，若与当前持仓不同则换仓
        if POSITION_MODE == 'single' and date in rebalance_dates:
            sold_tickers = set(t for t, _ in sell_list)
            holding_tickers = [t for t in holdings if t != bond_ticker and t not in sold_tickers]
            if holding_tickers:
                holding_ticker = holding_tickers[0]
                # 找满足买入条件中排名最优的候选（包含当前持仓）
                best_candidate = None
                best_rank = 999
                for t in stock_tickers:
                    if t not in signals or date not in signals[t].index:
                        continue
                    s = signals[t].loc[date]
                    row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
                    t_rank = rank_map.get(t, 99)
                    row_dict['rank'] = t_rank
                    buy_ok = _check_buy_conditions(row_dict, rank_map, t)
                    if NEW_RANK_LIMIT > 0 and t_rank > NEW_RANK_LIMIT:
                        buy_ok = False
                    if buy_ok and t_rank < best_rank:
                        best_rank = t_rank
                        best_candidate = t
                # 如果最优候选不是当前持仓，才换仓
                if best_candidate is not None and best_candidate != holding_ticker:
                    sell_list.append((holding_ticker, "轮动换仓"))

        for t, reason in sell_list:
            if t not in holdings:
                continue
            if t not in signals or next_date not in signals[t].index:
                continue
            o = signals[t].loc[next_date, 'open']
            if o <= 0:
                continue
            pos = holdings[t]
            sv = pos['shares'] * o
            fee = sv * FEE_RATE if t != bond_ticker else 0.0
            cash += (sv - fee)
            bp = pos['cost']
            pnl = (o - bp) / bp * 100 if bp > 0 else 0
            hd = (date - pos['buy_date']).days
            trade_log.append({
                'date': next_date, 'ticker': t, 'action': 'SELL', 'price': o,
                'shares': pos['shares'], 'value': sv, 'fee': fee, 'pnl_pct': pnl,
                'hold_days': hd, 'reason': reason
            })
            del holdings[t]

        # ---- 非轮动日 ----
        if date not in rebalance_dates:
            if bond_ticker and cash > 1e-6:
                if bond_ticker in signals and next_date in signals[bond_ticker].index:
                    o = signals[bond_ticker].loc[next_date, 'open']
                    if o > 0:
                        if bond_ticker in holdings:
                            old = holdings[bond_ticker]
                            add = cash / o
                            old['shares'] += add
                            old['cost'] = (old['shares'] * old['cost'] + cash) / old['shares'] if old['shares'] > 0 else old['cost']
                            trade_log.append({'date': next_date, 'ticker': bond_ticker, 'action': 'ADD_BOND',
                                              'price': o, 'shares': add, 'value': cash, 'fee': 0, 'pnl_pct': 0,
                                              'hold_days': (next_date - old['buy_date']).days, 'reason': '非轮动日资金归集'})
                        else:
                            shares = cash / o
                            holdings[bond_ticker] = {'shares': shares, 'cost': o, 'buy_date': next_date}
                            trade_log.append({'date': next_date, 'ticker': bond_ticker, 'action': 'BUY_BOND',
                                              'price': o, 'shares': shares, 'value': cash, 'fee': 0, 'pnl_pct': 0,
                                              'hold_days': 0, 'reason': '非轮动日空仓替代'})
                        cash = 0.0
            continue

        # ---- 轮动日 ----
        target_stocks = [t for t in holdings if t != bond_ticker]

        # 找新的买入候选
        candidates = []
        for t in stock_tickers:
            if t in target_stocks:
                continue
            if t not in signals or date not in signals[t].index:
                continue
            s = signals[t].loc[date]
            rank = rank_map.get(t, 99)
            row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
            row_dict['rank'] = rank
            buy_ok = _check_buy_conditions(row_dict, rank_map, t)
            # 排名限制
            if NEW_RANK_LIMIT > 0:
                rank_ok = rank <= NEW_RANK_LIMIT
            else:
                rank_ok = True
            if buy_ok and rank_ok:
                sort_val = sort_values.get(t, np.nan)
                if pd.notna(sort_val):
                    candidates.append((t, sort_val, rank))

        descending = (SORT_DIRECTION == 'desc')
        candidates.sort(key=lambda x: x[1], reverse=descending)
        slots = MAX_HOLDINGS - len(target_stocks)

        # incremental模式满仓换仓：卖最弱买最强
        if POSITION_MODE == 'incremental' and slots <= 0 and candidates:
            # 找最弱持仓（sort_value最差的）
            weakest_ticker = None
            if descending:
                weakest_sv = float('inf')
            else:
                weakest_sv = float('-inf')
            for t in target_stocks:
                if t in signals and date in signals[t].index:
                    sv = signals[t].loc[date, 'sort_value']
                    if pd.notna(sv):
                        if (descending and sv < weakest_sv) or \
                           (not descending and sv > weakest_sv):
                            weakest_sv = sv
                            weakest_ticker = t
            # 找最强候选（已按sort_value排序，candidates[0]最强）
            best_ticker, best_sv, best_rank = candidates[0]
            # 比较最强候选是否优于最弱持仓
            is_better = (best_sv > weakest_sv) if descending else (best_sv < weakest_sv)
            if weakest_ticker and pd.notna(best_sv) and is_better:
                # 卖最弱持仓
                if weakest_ticker in holdings:
                    if weakest_ticker in signals and next_date in signals[weakest_ticker].index:
                        open_price = signals[weakest_ticker].loc[next_date, 'open']
                        if open_price > 0:
                            pos = holdings[weakest_ticker]
                            sell_value = pos['shares'] * open_price
                            fee = sell_value * FEE_RATE
                            cash += (sell_value - fee)
                            bp = pos['cost']
                            pnl_pct = (open_price - bp) / bp * 100 if bp > 0 else 0
                            hd = (date - pos['buy_date']).days
                            trade_log.append({
                                'date': next_date, 'ticker': weakest_ticker, 'action': 'SELL',
                                'price': open_price, 'shares': pos['shares'], 'value': sell_value,
                                'fee': fee, 'pnl_pct': pnl_pct, 'hold_days': hd, 'reason': '增量换仓'
                            })
                            del holdings[weakest_ticker]
                target_stocks.remove(weakest_ticker)
                slots = 1  # 腾出一个位置

        for t, _, _ in candidates[:slots]:
            if t not in target_stocks:
                target_stocks.append(t)

        need_rebal = any(t not in holdings for t in target_stocks)

        if need_rebal:
            # 全量再平衡
            for t in list(holdings.keys()):
                if t in signals and next_date in signals[t].index:
                    p = signals[t].loc[next_date, 'open']
                    if p > 0:
                        cash += holdings[t]['shares'] * p
                del holdings[t]
            total = cash

            if POSITION_MODE == 'equal_weight':
                for t in target_stocks:
                    tv = total * POSITION_PCT
                    fee = tv * FEE_RATE
                    if t not in signals or next_date not in signals[t].index:
                        continue
                    o = signals[t].loc[next_date, 'open']
                    if o <= 0:
                        continue
                    sh = tv / o
                    bv = sh * o
                    cash -= (bv + fee)
                    holdings[t] = {'shares': sh, 'cost': o, 'buy_date': next_date}
                    trade_log.append({'date': next_date, 'ticker': t, 'action': 'BUY', 'price': o,
                                      'shares': sh, 'value': bv, 'fee': fee, 'pnl_pct': 0, 'hold_days': 0,
                                      'reason': '建仓/再平衡'})

            elif POSITION_MODE == 'single':
                if target_stocks:
                    t = target_stocks[0]
                    if t in signals and next_date in signals[t].index:
                        tv = total
                        fee = tv * FEE_RATE
                        o = signals[t].loc[next_date, 'open']
                        if o > 0:
                            sh = tv / o
                            bv = sh * o
                            cash -= (bv + fee)
                            holdings[t] = {'shares': sh, 'cost': o, 'buy_date': next_date}
                            trade_log.append({'date': next_date, 'ticker': t, 'action': 'BUY', 'price': o,
                                              'shares': sh, 'value': bv, 'fee': fee, 'pnl_pct': 0, 'hold_days': 0,
                                              'reason': '单标的建仓'})

            elif POSITION_MODE == 'incremental':
                for t in target_stocks:
                    if cash < total * POSITION_PCT * 0.5:
                        break
                    tv = total * POSITION_PCT
                    if t not in signals or next_date not in signals[t].index:
                        continue
                    fee = tv * FEE_RATE
                    o = signals[t].loc[next_date, 'open']
                    if o <= 0:
                        continue
                    sh = tv / o
                    bv = sh * o
                    if cash < bv + fee:
                        bv = cash - fee
                        if bv <= 0:
                            continue
                        sh = bv / o
                    cash -= (bv + fee)
                    holdings[t] = {'shares': sh, 'cost': o, 'buy_date': next_date}
                    trade_log.append({'date': next_date, 'ticker': t, 'action': 'BUY', 'price': o,
                                      'shares': sh, 'value': bv, 'fee': fee, 'pnl_pct': 0, 'hold_days': 0,
                                      'reason': '增量建仓'})

            # 余款买债券替代
            if bond_ticker and cash > 1e-6:
                if bond_ticker in signals and next_date in signals[bond_ticker].index:
                    o = signals[bond_ticker].loc[next_date, 'open']
                    if o > 0:
                        sh = cash / o
                        holdings[bond_ticker] = {'shares': sh, 'cost': o, 'buy_date': next_date}
                        trade_log.append({'date': next_date, 'ticker': bond_ticker, 'action': 'BUY_BOND',
                                          'price': o, 'shares': sh, 'value': cash, 'fee': 0, 'pnl_pct': 0,
                                          'hold_days': 0, 'reason': '空仓替代'})
                        cash = 0.0
        else:
            # 不需要再平衡：清理非目标持仓
            for t in list(holdings.keys()):
                if t != bond_ticker and t not in target_stocks:
                    if t not in signals or next_date not in signals[t].index:
                        continue
                    o = signals[t].loc[next_date, 'open']
                    if o <= 0:
                        continue
                    sv = holdings[t]['shares'] * o
                    fee = sv * FEE_RATE if t != bond_ticker else 0.0
                    cash += (sv - fee)
                    trade_log.append({'date': next_date, 'ticker': t, 'action': 'SELL_CLEAR',
                                      'price': o, 'shares': holdings[t]['shares'], 'value': sv, 'fee': fee,
                                      'pnl_pct': (o - holdings[t]['cost']) / holdings[t]['cost'] * 100 if holdings[t]['cost'] > 0 else 0,
                                      'hold_days': (next_date - holdings[t]['buy_date']).days, 'reason': '轮动调出'})
                    del holdings[t]
            # 剩余资金转债券替代
            if bond_ticker and cash > 1e-6:
                if bond_ticker in signals and next_date in signals[bond_ticker].index:
                    o = signals[bond_ticker].loc[next_date, 'open']
                    if o > 0:
                        if bond_ticker in holdings:
                            old = holdings[bond_ticker]
                            add = cash / o
                            old['shares'] += add
                            old['cost'] = (old['shares'] * old['cost'] + cash) / old['shares'] if old['shares'] > 0 else old['cost']
                            trade_log.append({'date': next_date, 'ticker': bond_ticker, 'action': 'ADD_BOND',
                                              'price': o, 'shares': add, 'value': cash, 'fee': 0, 'pnl_pct': 0,
                                              'hold_days': (next_date - old['buy_date']).days, 'reason': '轮动归集'})
                        else:
                            sh = cash / o
                            holdings[bond_ticker] = {'shares': sh, 'cost': o, 'buy_date': next_date}
                            trade_log.append({'date': next_date, 'ticker': bond_ticker, 'action': 'BUY_BOND',
                                              'price': o, 'shares': sh, 'value': cash, 'fee': 0, 'pnl_pct': 0,
                                              'hold_days': 0, 'reason': '空仓替代'})
                        cash = 0.0

    return pd.DataFrame(nav_history).set_index('date'), pd.DataFrame(trade_log), \
           pd.DataFrame(hold_history).set_index('date'), holdings, cash
''')

    # === 12. 绩效计算 ===
    lines.append(r'''
# ========================================
#  绩效计算
# ========================================

def _compute_performance(nav_df, trade_df, initial_capital):
    """计算策略绩效指标"""
    total_return = (nav_df['nav'].iloc[-1] / initial_capital - 1) * 100
    # 使用自然日天数，从第一笔交易开始（与Streamlit回测引擎一致）
    if not trade_df.empty:
        first_trade_date = pd.to_datetime(trade_df['date'].min())
        df_calc = nav_df[nav_df.index >= first_trade_date]
    else:
        df_calc = nav_df
    days = (nav_df.index[-1] - df_calc.index[0]).days
    if days > 0:
        annual_return = ((nav_df['nav'].iloc[-1] / initial_capital) ** (365 / days) - 1) * 100
    else:
        annual_return = 0

    dd_series = (nav_df['nav'] / nav_df['nav'].cummax()) - 1
    max_dd = dd_series.min() * 100
    max_dd_date = dd_series.idxmin()

    daily_ret = nav_df['nav'].pct_change().dropna()
    sharpe = (daily_ret.mean() * 252 - 0.03) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    buy_trades = trade_df[trade_df['action'] == 'BUY']
    sell_trades = trade_df[trade_df['action'] == 'SELL']
    win_trades = sell_trades[sell_trades['pnl_pct'] > 0]
    loss_trades = sell_trades[sell_trades['pnl_pct'] <= 0]
    win_rate = len(win_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
    avg_win = win_trades['pnl_pct'].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades['pnl_pct'].mean() if len(loss_trades) > 0 else 0
    avg_hold = sell_trades['hold_days'].mean() if len(sell_trades) > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'max_dd_date': max_dd_date,
        'sharpe': sharpe,
        'buy_count': len(buy_trades),
        'sell_count': len(sell_trades),
        'rebalance_count': len(trade_df[trade_df['action'].isin(['BUY_BOND','ADD_BOND','SELL_CLEAR'])]),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_hold': avg_hold,
    }
''')

    # === 13. _generate_trade_plan ===
    lines.append(r'''
# ========================================
#  交易计划生成
# ========================================

def _generate_trade_plan(signals, holdings, cash, stock_tickers, last_date, next_date, real_capital, common_dates=None):
    log_info("生成交易计划CSV（金额+股数双保留）...")

    rank_map, sort_values = _calc_rank_map(signals, stock_tickers, last_date)

    # 判断轮动日
    if common_dates is not None and last_date in common_dates:
        idx = common_dates.index(last_date)
        next_idx = idx + 1
        is_rebal = (next_idx % REBALANCE_DAYS == 0)
    else:
        all_dates = list(signals[stock_tickers[0]].index)
        is_rebal = False
        if last_date in all_dates:
            idx = all_dates.index(last_date)
            next_idx = idx + 1
            is_rebal = (next_idx % REBALANCE_DAYS == 0)

    try:
        nav_val = sum(pos['shares'] * signals[t].loc[last_date, 'close'] for t, pos in holdings.items() if t in signals and last_date in signals[t].index) + cash
        current_nav = float(nav_val) / INITIAL_CAPITAL
    except Exception as e:
        log_error("净值计算异常: {}".format(e))
        current_nav = 1.0

    real_total = real_capital * current_nav

    if not np.isfinite(real_total) or real_total <= 0:
        real_total = float(real_capital)

    try:
        nav_val = sum(pos['shares'] * signals[t].loc[last_date, 'close'] for t, pos in holdings.items() if t in signals and last_date in signals[t].index) + cash
    except Exception:
        nav_val = 0
    scale = (real_total / nav_val) if nav_val > 0 else 0
    log_info("  映射比例: scale={:.6f} (虚拟净值{:.0f} -> 实盘{:.0f})".format(scale, nav_val, real_total))

    current_values = {}
    for t, pos in holdings.items():
        if t in signals and last_date in signals[t].index:
            price = signals[t].loc[last_date, 'close']
            current_values[t] = pos['shares'] * price * scale
    cash_real = cash * scale

    is_rebalance_day = is_rebal

    # 卖出检查
    sell_tickers = []
    keep_tickers = []
    for t in holdings:
        if t == BOND_TICKER:
            continue
        if t not in signals or last_date not in signals[t].index:
            continue
        s = signals[t].loc[last_date]
        rank = rank_map.get(t, 99)
        row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
        row_dict['rank'] = rank
        row_dict['buy_price'] = holdings[t]['cost']
        should_sell, reason = _check_sell_conditions(row_dict, rank_map, t, holdings[t]['cost'], is_rebalance_day=is_rebal)
        if should_sell:
            sell_tickers.append(t)
            log_info("{} 卖出: {}".format(t, reason))
        else:
            keep_tickers.append(t)

    target_stocks = keep_tickers[:]
    need_rebal = False
    if is_rebal:
        candidates = []
        for t in stock_tickers:
            if t in keep_tickers:
                continue
            if t not in signals or last_date not in signals[t].index:
                continue
            s = signals[t].loc[last_date]
            rank = rank_map.get(t, 99)
            row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
            row_dict['rank'] = rank
            buy_ok = _check_buy_conditions(row_dict, rank_map, t)
            if NEW_RANK_LIMIT > 0:
                rank_ok = rank <= NEW_RANK_LIMIT
            else:
                rank_ok = True
            if buy_ok and rank_ok:
                candidates.append((t, rank_map[t]))
                log_info("候选: {} (排名{})".format(t, rank_map[t]))
        candidates.sort(key=lambda x: x[1])
        slots = MAX_HOLDINGS - len(keep_tickers)
        for t, _ in candidates[:slots]:
            if t not in target_stocks:
                target_stocks.append(t)
        need_rebal = any(t not in holdings for t in target_stocks)

    t1_amounts = {}
    if POSITION_MODE == 'equal_weight':
        target_amount = real_total * POSITION_PCT
    elif POSITION_MODE == 'single':
        target_amount = real_total
    elif POSITION_MODE == 'incremental':
        target_amount = real_total * POSITION_PCT
    else:
        target_amount = real_total * POSITION_PCT

    if is_rebal and need_rebal:
        log_info("轮动日+再平衡")
        if POSITION_MODE == 'equal_weight':
            for t in ALL_TICKERS:
                if t in target_stocks and t != BOND_TICKER:
                    t1_amounts[t] = target_amount
                elif t == BOND_TICKER and t in target_stocks:
                    sc = len([x for x in target_stocks if x != BOND_TICKER])
                    t1_amounts[t] = max(0, real_total - sc * target_amount)
                else:
                    t1_amounts[t] = 0.0
        elif POSITION_MODE == 'single':
            for t in ALL_TICKERS:
                if t == target_stocks[0] if target_stocks else False:
                    t1_amounts[t] = target_amount
                else:
                    t1_amounts[t] = 0.0
        elif POSITION_MODE == 'incremental':
            for t in ALL_TICKERS:
                if t in target_stocks and t != BOND_TICKER:
                    t1_amounts[t] = target_amount
                elif t == BOND_TICKER:
                    sc = len([x for x in target_stocks if x != BOND_TICKER])
                    t1_amounts[t] = max(0, real_total - sc * target_amount)
                else:
                    t1_amounts[t] = 0.0
    else:
        if is_rebal and not need_rebal:
            extra = [t for t in keep_tickers if t not in target_stocks]
            if extra:
                sell_tickers = list(set(sell_tickers + extra))
        sv = sum(current_values.get(t, 0) for t in sell_tickers)
        for t in ALL_TICKERS:
            if t == BOND_TICKER:
                t1_amounts[t] = current_values.get(BOND_TICKER, 0) + sv + cash_real
            elif t in sell_tickers:
                t1_amounts[t] = 0.0
            elif t in keep_tickers and t not in sell_tickers:
                t1_amounts[t] = current_values.get(t, 0)
            else:
                t1_amounts[t] = 0.0

    rows = []
    for t in ALL_TICKERS:
        sec_id = t.replace('.SH', '').replace('.SZ', '')
        amount = float(t1_amounts.get(t, 0))
        close_price = signals[t].loc[last_date, 'close'] if t in signals and last_date in signals[t].index else 0
        if amount > 0 and close_price > 0:
            shares = int(round(amount / close_price / ETF_MIN_UNIT)) * ETF_MIN_UNIT
        else:
            shares = 0
        rows.append({
            'sec_id': sec_id,
            'weight': round(amount, 2),
            'shares': shares,
            'name': TICKER_NAMES.get(t, t)
        })

    d = SCRIPT_DIR
    if not os.path.exists(d):
        os.makedirs(d)
    with codecs.open(TRADE_PLAN_CSV, 'w', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['sec_id', 'weight', 'shares', 'name'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log_critical("交易计划已保存: {} (金额+股数双保留)".format(TRADE_PLAN_CSV))
    log_info("  计算日期: {}, 交易日期: {}".format(last_date.date(), next_date.date()))
    log_info("  净值: {:.4f}, 总资金: {:.0f}, 目标金额: {:.0f}".format(current_nav, real_total, target_amount))
    log_info("  轮动日: {}, 再平衡: {}".format(is_rebal, is_rebal and need_rebal))
    for r in rows:
        if r['shares'] > 0:
            log_info("    {} ({}): {:.0f}元 / {}股".format(r['sec_id'], r['name'], r['weight'], r['shares']))
    return rows
''')

    # === 14. _generate_signal_report ===
    lines.append(r'''
# ========================================
#  信号报告
# ========================================

def _generate_signal_report(signals, holdings, stock_tickers, last_date, next_date, real_capital, common_dates=None):
    rank_map, sort_values = _calc_rank_map(signals, stock_tickers, last_date)
    hold_tickers = list(holdings.keys())

    if common_dates is not None and last_date in common_dates:
        idx = common_dates.index(last_date)
        next_idx = idx + 1
        is_rebal = (next_idx % REBALANCE_DAYS == 0)
    else:
        all_dates = list(signals[stock_tickers[0]].index)
        is_rebal = False
        if last_date in all_dates:
            idx = all_dates.index(last_date)
            next_idx = idx + 1
            is_rebal = (next_idx % REBALANCE_DAYS == 0)

    sell_list = []
    for t in hold_tickers:
        if t == BOND_TICKER:
            continue
        if t not in signals or last_date not in signals[t].index:
            continue
        rank = rank_map.get(t, 99)
        s = signals[t].loc[last_date]
        row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
        row_dict['rank'] = rank
        row_dict['buy_price'] = holdings[t]['cost']
        should_sell, reason = _check_sell_conditions(row_dict, rank_map, t, holdings[t]['cost'], is_rebalance_day=is_rebal)
        if should_sell:
            sell_list.append((t, reason))

    sell_names = [TICKER_NAMES.get(t, t) for t, _ in sell_list]
    keep_tickers = [t for t in hold_tickers if t != BOND_TICKER and t not in [st for st, _ in sell_list]]
    buy_list = []
    if is_rebal:
        candidates = []
        for t in stock_tickers:
            if t in keep_tickers:
                continue
            if t not in signals or last_date not in signals[t].index:
                continue
            s = signals[t].loc[last_date]
            rank = rank_map.get(t, 99)
            row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
            row_dict['rank'] = rank
            buy_ok = _check_buy_conditions(row_dict, rank_map, t)
            if NEW_RANK_LIMIT > 0:
                rank_ok = rank <= NEW_RANK_LIMIT
            else:
                rank_ok = True
            if buy_ok and rank_ok:
                candidates.append(t)
        slots = MAX_HOLDINGS - len(keep_tickers)
        buy_list = candidates[:slots]
    buy_names = [TICKER_NAMES.get(t, t) for t in buy_list]

    signal_parts = []
    if sell_names:
        signal_parts.append("卖出: {}".format(', '.join(sell_names)))
    if buy_names:
        signal_parts.append("买入: {}".format(', '.join(buy_names)))
    if not signal_parts:
        next_signal = "持仓不动"
    else:
        next_signal = " | ".join(signal_parts)

    target_stocks = keep_tickers + buy_list
    target_str = ', '.join([TICKER_NAMES.get(t, t) for t in target_stocks]) if target_stocks else '空仓(银华日利)'

    hold_str = ', '.join([TICKER_NAMES.get(t, t) for t in hold_tickers]) if hold_tickers else '空仓'
    descending = (SORT_DIRECTION == 'desc')
    ranked_list = sorted(sort_values.items(), key=lambda x: x[1], reverse=descending)
    rank_str = ' / '.join(["{}({:+.1f})".format(TICKER_NAMES.get(t, t), v) for t, v in ranked_list[:5]])

    try:
        nav_val = sum(pos['shares'] * signals[t].loc[last_date, 'close'] for t, pos in holdings.items() if t in signals and last_date in signals[t].index)
        current_nav = float(nav_val) / INITIAL_CAPITAL
    except Exception:
        current_nav = 1.0
    real_total = real_capital * current_nav

    r = []
    r.append("=" * 70)
    r.append("{} - 交易信号报告".format(strategy_name))
    r.append("=" * 70)
    r.append("计算日期: {} | 交易日期: {} 开盘".format(last_date.date(), next_date.date()))
    r.append("实盘资金: {:.0f}元 | 净值: {:.4f} | 总资金: {:.0f}元".format(real_capital, current_nav, real_total))
    r.append("轮动日: {} | 明日信号: {}".format("是" if is_rebal else "否", next_signal))
    r.append("当前持仓({}只): {}".format(len([t for t in hold_tickers if t != BOND_TICKER]), hold_str))
    r.append("明日目标({}只): {}".format(len(target_stocks), target_str))
    r.append("{}排名: {}".format(SORT_INDICATOR, rank_str))
    r.append("")
    r.append("=" * 70)
    r.append("{:<4} {:<12} {:<10} {:<10} {:<8}".format("排名", "代码", "名称", SORT_INDICATOR, "信号"))
    r.append("-" * 70)
    for i, (t, v) in enumerate(ranked_list):
        s = signals[t].loc[last_date]
        row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
        rank_i = i + 1
        row_dict['rank'] = rank_i
        buy_ok = _check_buy_conditions(row_dict, rank_map, t)
        if NEW_RANK_LIMIT > 0:
            rank_ok = rank_i <= NEW_RANK_LIMIT
        else:
            rank_ok = True
        ok = buy_ok and rank_ok
        marker = " << 持仓" if t in hold_tickers else ""
        r.append("{:<4} {:<12} {:<10} {:>+8.1f}   {:<8}{}".format(
            i+1, t, TICKER_NAMES.get(t, t), v, "可买" if ok else "--", marker))
    r.append("=" * 70)
    if sell_list:
        r.append("")
        r.append("卖出检查:")
        for t, reason in sell_list:
            r.append("  {} ({}): {}".format(t, TICKER_NAMES.get(t), reason))
    return "\\n".join(r)
'''.replace('strategy_name', f'"{strategy_name}"'))

    # === 15. 监控面板 ===
    lines.append(r'''
# ========================================
#  监控面板
# ========================================

def _generate_dashboard(nav_df, trade_df, hold_df, holdings, signals, all_tickers, stock_tickers, perf, cash, real_capital, last_date=None):
    """生成完整5张子图监控面板（含实盘金额）"""
    if not _mpl_available:
        log_info("matplotlib不可用，跳过监控面板生成")
        return None

    try:
        if last_date is None:
            last_date = nav_df.index[-1]

        rank_map, sort_values = _calc_rank_map(signals, stock_tickers, last_date)
        hold_tickers = list(holdings.keys())
        descending = (SORT_DIRECTION == 'desc')
        ranked = sorted(sort_values.items(), key=lambda x: x[1], reverse=descending)

        # 检查明日信号
        next_signal = None
        for ticker in hold_tickers:
            if ticker == BOND_TICKER:
                continue
            if ticker not in signals or last_date not in signals[ticker].index:
                continue
            rank = rank_map.get(ticker, 99)
            s = signals[ticker].loc[last_date]
            row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
            row_dict['rank'] = rank
            row_dict['buy_price'] = holdings[ticker]['cost']
            should_sell, reason = _check_sell_conditions(row_dict, rank_map, ticker, holdings[ticker]['cost'])
            if should_sell:
                next_signal = "卖出: {}".format(ticker)
                break

        if not next_signal and len([t for t in hold_tickers if t != BOND_TICKER]) < MAX_HOLDINGS:
            for ticker, sort_val in ranked:
                if ticker not in hold_tickers:
                    if ticker not in signals or last_date not in signals[ticker].index:
                        continue
                    s = signals[ticker].loc[last_date]
                    row_dict = s.to_dict() if isinstance(s, pd.Series) else dict(s)
                    row_dict['rank'] = rank_map.get(ticker, 99)
                    buy_ok = _check_buy_conditions(row_dict, rank_map, ticker)
                    if buy_ok:
                        next_signal = "轮动: {}".format(ticker)
                        break

        if not next_signal:
            next_signal = "持仓不动"

        hold_names_str = ', '.join([TICKER_NAMES.get(t, t) for t in hold_tickers]) if hold_tickers else '空仓'
        rank_strs = ["{}({:+.1f})".format(TICKER_NAMES.get(t, t), v) for t, v in ranked[:5]]

        tomorrow = last_date + pd.Timedelta(days=1)
        while tomorrow.weekday() >= 5:
            tomorrow += pd.Timedelta(days=1)

        try:
            nav_val = nav_df.loc[last_date, 'nav']
            if isinstance(nav_val, pd.Series):
                nav_val = nav_val.iloc[0]
            current_nav = float(nav_val) / INITIAL_CAPITAL
        except Exception:
            current_nav = 1.0
        real_total = real_capital * current_nav

        banner_text = ("明日交易信号 | 计算: {} | 交易: {} | {} | 持仓: {} | 排名: {} | 实盘: {:,.0f}元".format(
            last_date.date(), tomorrow.date(), next_signal, hold_names_str, '/'.join(rank_strs), real_total))

        fig = plt.figure(figsize=(20, 22))
        gs = GridSpec(5, 1, figure=fig, height_ratios=[0.35, 2.2, 1, 1.3, 1.3], hspace=0.30,
                      left=0.06, right=0.98, top=0.97, bottom=0.03)

        nav_normalized = nav_df['nav'] / INITIAL_CAPITAL
        dd = (nav_df['nav'] / nav_df['nav'].cummax() - 1) * 100
        max_dd_val = dd.min()
        max_dd_idx = dd.idxmin()
        start_date = nav_df.index[0]
        end_date = nav_df.index[-1]
        plot_xlim = (start_date, end_date)

        # 第0层：顶部信号横幅
        ax_banner = fig.add_subplot(gs[0])
        ax_banner.axis('off')
        ax_banner.text(0.5, 0.5, banner_text, transform=ax_banner.transAxes,
                       fontproperties=fp_banner, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFD700', edgecolor='black', alpha=0.9))
        ax_banner.set_xlim(0, 1)
        ax_banner.set_ylim(0, 1)

        # 子图1: 策略净值曲线
        ax1 = fig.add_subplot(gs[1])
        for ticker in all_tickers:
            name = TICKER_NAMES.get(ticker, ticker)
            valid_after_start = signals[ticker]['close'][
                (signals[ticker]['close'] > 0.5) & (signals[ticker].index >= start_date)]
            if len(valid_after_start) > 0:
                base = valid_after_start.iloc[0]
                nav_t = signals[ticker]['close'] / base
                nav_t = nav_t[nav_t.index >= start_date]
                ax1.plot(nav_t.index, nav_t, label=name,
                        color=colors_fund.get(name, 'gray'), linewidth=0.5, alpha=0.25, linestyle='--')

        daily_df_sorted = hold_df.reset_index().sort_values('date').reset_index(drop=True)
        for i in range(len(daily_df_sorted) - 1):
            date1 = daily_df_sorted['date'].iloc[i]
            date2 = daily_df_sorted['date'].iloc[i + 1]
            val1 = nav_df.loc[date1, 'nav'] / INITIAL_CAPITAL
            val2 = nav_df.loc[date2, 'nav'] / INITIAL_CAPITAL
            hold_list = daily_df_sorted['hold_tickers'].iloc[i]
            if hold_list and len(hold_list) > 0:
                main_hold = hold_list[0]
                color = colors_fund.get(TICKER_NAMES.get(main_hold, ''), '#d62728')
            else:
                color = '#95A5A6'
            ax1.plot([date1, date2], [val1, val2], color=color, linewidth=2.2, alpha=0.9, solid_capstyle='round')

        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.set_title('策略净值 | 现价{:.4f} | 收益{:.1f}% | 年化{:.1f}% | 回撤{:.1f}% | 夏普{:.2f} | 实盘{:,.0f}元'.format(
            nav_normalized.iloc[-1], perf['total_return'], perf['annual_return'], perf['max_dd'], perf['sharpe'], real_total),
            fontproperties=fp_title, pad=10)
        ax1.set_ylabel('净值', fontproperties=fp_label)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_xlim(plot_xlim)
        for label in ax1.get_xticklabels(): label.set_fontproperties(fp_tick)
        for label in ax1.get_yticklabels(): label.set_fontproperties(fp_tick)

        legend_elements = [Patch(facecolor=colors_fund[name], edgecolor='none', label=name)
                           for name in colors_fund if name != TICKER_NAMES.get(BOND_TICKER, '')]
        legend_elements.append(Patch(facecolor='#95A5A6', edgecolor='none', label='空仓/银华日利'))
        leg = ax1.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize=7,
                   framealpha=0.8, title='持仓颜色')
        leg.get_title().set_fontproperties(fp_legend_title)
        for text in leg.get_texts(): text.set_fontproperties(fp_legend)

        # 子图2: 策略回撤
        ax2 = fig.add_subplot(gs[2])
        ax2.fill_between(dd.index, dd, 0, color='#3498DB', alpha=0.4)
        ax2.plot(dd.index, dd, color='#3498DB', linewidth=0.8)
        ax2.axhline(y=max_dd_val, color='red', linestyle='--', alpha=0.7, linewidth=1.2,
                    label='最大回撤 {:.2f}%'.format(max_dd_val))
        ax2.scatter([max_dd_idx], [max_dd_val], color='red', s=60, zorder=5)
        ax2.annotate('{:.1f}%'.format(max_dd_val), xy=(max_dd_idx, max_dd_val),
                     xytext=(10, -15), textcoords='offset points',
                     fontsize=9, color='red', fontweight='bold')
        current_dd = dd.iloc[-1]
        ax2.axhline(y=current_dd, color='#FF8C00', linestyle='--', alpha=0.7, linewidth=1.2,
                    label='当前回撤 {:.2f}%'.format(current_dd))
        ax2.set_title('策略回撤 | 最大回撤: {:.2f}% | 当前回撤: {:.2f}%'.format(max_dd_val, current_dd),
                      fontproperties=fp_subtitle)
        ax2.set_ylabel('回撤 (%)', fontproperties=fp_label)
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.set_xlim(plot_xlim)
        for label in ax2.get_xticklabels(): label.set_fontproperties(fp_tick)
        for label in ax2.get_yticklabels(): label.set_fontproperties(fp_tick)

        # 子图3: 各标的净值走势
        ax3 = fig.add_subplot(gs[3])
        y_min_all, y_max_all = float('inf'), float('-inf')
        legend_items = []
        for ticker in all_tickers:
            name = TICKER_NAMES.get(ticker, ticker)
            s_t = signals[ticker]
            start_idx = s_t.index.get_indexer([start_date], method='nearest')[0]
            base_price = s_t['close'].iloc[start_idx] if start_idx >= 0 else None
            if base_price is None or base_price <= 0 or pd.isna(base_price):
                valid_prices = s_t['close'][((s_t['close'] > 0) & (s_t.index >= start_date))]
                if len(valid_prices) > 0:
                    base_price = valid_prices.iloc[0]
                else:
                    continue
            nav_t = s_t['close'] / base_price
            nav_t = nav_t[nav_t.index >= start_date]
            if len(nav_t) > 0 and not nav_t.isna().all():
                y_min_all = min(y_min_all, nav_t.min())
                y_max_all = max(y_max_all, nav_t.max())
                line, = ax3.plot(nav_t.index, nav_t,
                        color=colors_fund.get(name, 'gray'), linewidth=1.0, alpha=0.8)
                latest_nav = nav_t.iloc[-1] if len(nav_t) > 0 else 0
                legend_items.append((latest_nav, name, colors_fund.get(name, 'gray'), line))
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        if y_min_all < float('inf'):
            ax3.set_ylim(y_min_all * 0.9, y_max_all * 1.1)
        ax3.set_title('各标的净值走势（归一化）', fontproperties=fp_subtitle)
        ax3.set_ylabel('净值', fontproperties=fp_label)
        legend_items.sort(key=lambda x: x[0], reverse=True)
        sorted_handles = [item[3] for item in legend_items]
        sorted_labels = ["{} ({:.2f})".format(item[1], item[0]) for item in legend_items]
        leg3 = ax3.legend(sorted_handles, sorted_labels, loc='upper left', ncol=2, fontsize=7, framealpha=0.8)
        for text in leg3.get_texts(): text.set_fontproperties(fp_legend)
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.set_xlim(plot_xlim)
        for label in ax3.get_xticklabels(): label.set_fontproperties(fp_tick)
        for label in ax3.get_yticklabels(): label.set_fontproperties(fp_tick)

        # 子图4: 各标的最新排名
        ax4 = fig.add_subplot(gs[4])
        ax4.axis('off')

        left_pos = [0.08, 0.02, 0.88, 0.22]
        ax_left = fig.add_axes(left_pos)

        all_vals = [(t, v) for t, v in sort_values.items()]
        all_vals.sort(key=lambda x: x[1], reverse=descending)
        names = [TICKER_NAMES.get(t, t) for t, _ in all_vals]
        values = [v for _, v in all_vals]

        colors_bar = []
        for t, _ in all_vals:
            if t in hold_tickers:
                colors_bar.append('#FF8C00')
            else:
                colors_bar.append('#4682B4')

        bars = ax_left.barh(range(len(names)), values, color=colors_bar, edgecolor='black', linewidth=0.5, height=0.6)
        for i, (t, v) in enumerate(all_vals):
            ax_left.text(v + 3 if v >= 0 else v - 3, i, '{:.1f}'.format(v), va='center', fontsize=9, color='black',
                        ha='left' if v >= 0 else 'right')
        ax_left.set_yticks(range(len(names)))
        ax_left.set_yticklabels(names, fontsize=9)
        ax_left.set_xlabel('{}值'.format(SORT_INDICATOR), fontproperties=fp_label)
        ax_left.set_title('各标的最新{}排名 ({})'.format(SORT_INDICATOR, last_date.date()), fontproperties=fp_subtitle)
        ax_left.axvline(x=0, color='black', linewidth=0.8)
        ax_left.grid(True, alpha=0.3, axis='x')
        ax_left.invert_yaxis()
        for label in ax_left.get_xticklabels(): label.set_fontproperties(fp_tick)
        for label in ax_left.get_yticklabels(): label.set_fontproperties(fp_tick)

        orange_patch = mpatches.Patch(color='#FF8C00', label='当前持仓')
        blue_patch = mpatches.Patch(color='#4682B4', label='未持仓')
        ax_left.legend(handles=[orange_patch, blue_patch], loc='lower right', fontsize=9)

        d = SCRIPT_DIR
        if not os.path.exists(d):
            os.makedirs(d)
        plt.savefig(DASHBOARD_PATH, dpi=150, bbox_inches='tight')
        plt.close()
        log_critical("监控面板已保存: {}".format(DASHBOARD_PATH))
        return DASHBOARD_PATH
    except Exception as e:
        log_error("生成监控面板异常: {}".format(e))
        log_error(traceback.format_exc())
        return None
''')

    # === 16. passorder下单模块 ===
    lines.append(r'''
# ========================================
#  下单模块
# ========================================

def _passorder_compat(ContextInfo, side, accid, code, price_type, price, volume, tag):
    order_type = BUY_ORDER_TYPE if side == "buy" else SELL_ORDER_TYPE
    code = _normalize_code(code)
    vi = int(volume)
    today = _today_str()
    now = _now_str()
    log_info("[{} {}] {}委托: {} 数量={}".format(today, now, side, code, vi))
    try:
        passorder(order_type, ACCOUNT_TYPE, str(accid), code, price_type, price, vi, tag, 1, "按金额", ContextInfo)
        log_info("[{} {}] {}成功: {}".format(today, now, side, code))
        return True
    except Exception as e:
        log_info("[{} {}] 方式1失败: {}".format(today, now, e))
    try:
        passorder(order_type, ACCOUNT_TYPE, str(accid), code, price_type, price, vi, ContextInfo)
        log_info("[{} {}] {}成功(方式2): {}".format(today, now, side, code))
        return True
    except Exception as e:
        log_info("[{} {}] {}失败 {}: {}".format(today, now, side, code, e))
        return False

def _read_trade_plan_csv():
    """读取交易计划CSV（只读shares股数列）"""
    try:
        if not os.path.exists(TRADE_PLAN_CSV):
            log_info("CSV不存在: {}".format(TRADE_PLAN_CSV))
            return {}
        if os.path.getsize(TRADE_PLAN_CSV) == 0:
            log_info("CSV为空: {}".format(TRADE_PLAN_CSV))
            return {}
        target_shares = {}
        with codecs.open(TRADE_PLAN_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if 'sec_id' not in (reader.fieldnames or []) or 'shares' not in (reader.fieldnames or []):
                log_info("CSV格式错误: {}".format(reader.fieldnames))
                return {}
            row_count = 0
            for row in reader:
                row_count += 1
                sec_id = str(row.get('sec_id', '')).strip()
                shares_str = row.get('shares', '')
                if not sec_id or not shares_str or str(shares_str).strip() == '':
                    continue
                try:
                    shares = int(float(shares_str))
                    if shares >= 0:
                        code = _normalize_code(sec_id)
                        target_shares[code] = shares
                        log_info("读取到标的: {} = {}股".format(code, shares))
                except Exception:
                    continue
            log_info("CSV解析完成: {}行, {}个标的".format(row_count, len(target_shares)))
        return target_shares
    except Exception as e:
        log_error("读取CSV失败: {}".format(e))
        return {}

def _calculate_trade_diffs(ContextInfo, target_shares):
    """直接对比目标股数与线上持仓股数"""
    try:
        current_accid = ACCOUNT_ID
        try:
            positions = get_trade_detail_data(current_accid, "STOCK", "POSITION")
        except Exception as e:
            log_error("获取持仓异常: {}".format(e))
            return {}
        if positions is None:
            log_error("获取持仓返回None，中止交易防止误下单")
            return {}
        current_pos = {}
        for pos in positions:
            code = _normalize_code(pos.m_strInstrumentID)
            if code in target_shares:
                current_pos[code] = int(pos.m_nVolume)
        log_info("当前持仓: {}".format(current_pos))

        diffs = {}
        for code in target_shares:
            target_vol = target_shares.get(code, 0)
            current_vol = current_pos.get(code, 0)
            diff = target_vol - current_vol
            if diff != 0:
                diffs[code] = diff
                action = "买入" if diff > 0 else "卖出"
                log_info("交易计划: {} 目标{}股 持仓{}股 需要 {} {}股".format(
                    code, target_vol, current_vol, action, abs(diff)))
        return diffs
    except Exception as e:
        log_error("计算交易diff失败: {}".format(e))
        log_error(traceback.format_exc())
        return {}

def _execute_trades(ContextInfo, diffs, tag_prefix="定时"):
    if not diffs:
        log_info("[{}] 无调仓需求，跳过".format(tag_prefix))
        return
    today = _today_str()
    now = _now_str()
    current_accid = ACCOUNT_ID
    sell_diffs = {k: v for k, v in diffs.items() if v < 0}
    if sell_diffs:
        log_critical("[{}] [{} {}] 开始卖出".format(tag_prefix, today, now))
        for code, diff in sell_diffs.items():
            try:
                volume = abs(diff)
                log_info("[{}] 准备卖出 {}: {}股".format(tag_prefix, code, volume))
                ok = _passorder_compat(ContextInfo, "sell", current_accid, code, PRICE_TYPE, -1, volume, SELL_TAG)
                if ok:
                    log_info("[{}] 卖出成功 {}: {}股".format(tag_prefix, code, volume))
                time.sleep(0.1)
            except Exception as e:
                log_info("[{}] 卖出异常 {}: {}".format(tag_prefix, code, e))
    buy_diffs = {k: v for k, v in diffs.items() if v > 0}
    if buy_diffs:
        log_critical("[{}] [{} {}] 开始买入".format(tag_prefix, today, now))
        for code, diff in buy_diffs.items():
            try:
                volume = diff
                log_info("[{}] 准备买入 {}: {}股".format(tag_prefix, code, volume))
                ok = _passorder_compat(ContextInfo, "buy", current_accid, code, PRICE_TYPE, -1, volume, BUY_TAG)
                if ok:
                    log_info("[{}] 买入成功 {}: {}股".format(tag_prefix, code, volume))
                time.sleep(0.1)
            except Exception as e:
                log_info("[{}] 买入异常 {}: {}".format(tag_prefix, code, e))
    log_info("[{}] 交易执行完成".format(tag_prefix))
''')

    # === 17. 定时框架 ===
    lines.append(r'''
# ========================================
#  定时任务与QMT接口
# ========================================

def _setup_schedule(ContextInfo):
    global _last_trading_day, _intraday_done, _pre_market_done, _exec_done
    _last_trading_day = ""
    _intraday_done = False
    _pre_market_done = False
    _exec_done = False

    tasks = []
    if ENABLE_SCHEDULED_MODE:
        tasks.extend([
            (TASK_PRE_MARKET, "pre_market", task_pre_market),
            (TASK_EXEC_TIME, "execute", task_execute),
        ])
    if ENABLE_INTRADAY_MODE:
        tasks.append((TASK_INTRADAY_TIME, "intraday", task_intraday_calc))

    if not tasks:
        log_info("所有定时任务已关闭，跳过注册")
        return

    for t_str, key, fn in tasks:
        registered = False
        try:
            if hasattr(ContextInfo, "run_time"):
                try:
                    ContextInfo.run_time(t_str, fn)
                    registered = True
                except TypeError:
                    try:
                        ContextInfo.run_time(fn, t_str)
                        registered = True
                    except TypeError:
                        try:
                            ContextInfo.run_time(t_str, fn, "task_" + key)
                            registered = True
                        except TypeError:
                            pass
            if not registered and "run_time" in globals():
                try:
                    globals()["run_time"](t_str, fn)
                    registered = True
                except TypeError:
                    try:
                        globals()["run_time"](fn, t_str)
                        registered = True
                    except TypeError:
                        pass
            if registered:
                log_info("定时任务注册成功: {} -> {}".format(t_str, key))
            else:
                log_info("run_time签名不匹配，使用备用机制")
        except Exception as e:
            log_info("定时任务注册失败 {}: {}".format(t_str, e))

def _fallback_timecheck(ContextInfo):
    global _last_trading_day, _intraday_done, _pre_market_done, _exec_done
    today = _today_str()
    if _last_trading_day != today:
        _last_trading_day = today
        _intraday_done = False
        _pre_market_done = False
        _exec_done = False
    now = _now_str()
    now_min = int(now[:2]) * 60 + int(now[3:5])
    tasks = []
    if ENABLE_SCHEDULED_MODE:
        tasks.extend([
            (TASK_PRE_MARKET, "pre_market", task_pre_market, _pre_market_done),
            (TASK_EXEC_TIME, "execute", task_execute, _exec_done),
        ])
    if ENABLE_INTRADAY_MODE:
        tasks.append((TASK_INTRADAY_TIME, "intraday", task_intraday_calc, _intraday_done))
    for t_str, key, fn, done_flag in tasks:
        if done_flag:
            continue
        t_min = int(t_str[:2]) * 60 + int(t_str[3:5])
        if now_min >= t_min and now_min <= t_min + 5:
            if key == "pre_market":
                _pre_market_done = True
            elif key == "execute":
                _exec_done = True
            elif key == "intraday":
                _intraday_done = True
            fn(ContextInfo)

def task_pre_market(ContextInfo):
    today = _today_str()
    now = _now_str()
    log_critical("=" * 60)
    log_critical("[盘前更新] {} {}".format(today, now))
    log_critical("=" * 60)
    try:
        _download_and_save_daily_data(ContextInfo)
    except Exception as e:
        log_error("数据下载失败: {}".format(e))
    try:
        pkl_dir = _detect_pkl_dir()
        data_df = _load_data_from_pkl(pkl_dir)
        if data_df is None:
            log_error("数据加载失败，无法生成交易计划")
            return
        data_dict = _build_data_dict(data_df)
        data_dict, common_dates = _align_dates(data_dict)
        stock_tickers = [t for t in data_dict.keys() if t != BOND_TICKER]
        all_tickers = list(data_dict.keys())
        signals = _calc_signals(data_dict, all_tickers)
        nav_df, trade_df, hold_df, holdings, cash = _run_backtest(
            signals, stock_tickers, BOND_TICKER, all_tickers, common_dates,
            INITIAL_CAPITAL, start_date=STRATEGY_START
        )
        last_date = common_dates[-1]
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        rows = _generate_trade_plan(signals, holdings, cash, stock_tickers, last_date, next_date, REAL_CAPITAL, common_dates)
        report = _generate_signal_report(signals, holdings, stock_tickers, last_date, next_date, REAL_CAPITAL, common_dates)
        try:
            trade_df.to_csv(TRADE_RECORDS_PATH, index=False, encoding='utf-8')
            log_info("交易记录已保存: {}".format(TRADE_RECORDS_PATH))
        except Exception as e:
            log_error("保存交易记录失败: {}".format(e))
        try:
            with codecs.open(SIGNAL_REPORT_PATH, 'w', encoding='utf-8') as f:
                f.write(report)
        except Exception as e:
            log_error("保存报告失败: {}".format(e))
        log_critical("[盘前更新] 完成: CSV={} ({}行)".format(TRADE_PLAN_CSV, len(rows)))
        try:
            perf = _compute_performance(nav_df, trade_df, INITIAL_CAPITAL)
            _generate_dashboard(nav_df, trade_df, hold_df, holdings, signals, all_tickers, stock_tickers, perf, cash, REAL_CAPITAL, last_date=last_date)
        except Exception as e:
            log_error("[盘前更新] 监控面板生成失败: {}".format(e))
    except Exception as e:
        log_error("[盘前更新] 异常: {}".format(e))
        log_error(traceback.format_exc())

def task_execute(ContextInfo):
    today = _today_str()
    now = _now_str()
    log_critical("=" * 60)
    log_critical("[交易执行] {} {}".format(today, now))
    log_critical("=" * 60)
    try:
        target_shares = _read_trade_plan_csv()
        if not target_shares:
            log_info("[交易执行] 无交易计划，跳过")
            return
        diffs = _calculate_trade_diffs(ContextInfo, target_shares)
        if not diffs:
            log_info("[交易执行] 无调仓需求")
            return
        _execute_trades(ContextInfo, diffs, "定时")
    except Exception as e:
        log_error("[交易执行] 异常: {}".format(e))
        log_error(traceback.format_exc())

def task_intraday_calc(ContextInfo):
    today = _today_str()
    now = _now_str()
    log_critical("=" * 60)
    log_critical("[盘中重算] {} {}".format(today, now))
    log_critical("=" * 60)
    try:
        log_info("[盘中重算] 步骤1/4: 重新加载最新数据...")
        pkl_dir = _detect_pkl_dir()
        data_df = _load_data_from_pkl(pkl_dir)
        if data_df is None:
            log_error("[盘中重算] 数据加载失败，中止")
            return
        log_info("[盘中重算] 步骤2/4: 构建数据字典...")
        data_dict = _build_data_dict(data_df)
        data_dict, common_dates = _align_dates(data_dict)
        stock_tickers = [t for t in data_dict.keys() if t != BOND_TICKER]
        all_tickers = list(data_dict.keys())
        log_info("[盘中重算] 步骤3/4: 重新计算信号并回测...")
        signals = _calc_signals(data_dict, all_tickers)
        nav_df, trade_df, hold_df, holdings, cash = _run_backtest(
            signals, stock_tickers, BOND_TICKER, all_tickers, common_dates,
            INITIAL_CAPITAL, start_date=STRATEGY_START
        )
        last_date = common_dates[-1]
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        _generate_trade_plan(signals, holdings, cash, stock_tickers, last_date, next_date, REAL_CAPITAL, common_dates)
        log_critical("[盘中重算] 步骤4/4: 立即执行交易...")
        target_shares = _read_trade_plan_csv()
        if not target_shares:
            log_info("[盘中重算] 无新交易计划")
            return
        diffs = _calculate_trade_diffs(ContextInfo, target_shares)
        if not diffs:
            log_info("[盘中重算] 无调仓需求")
            return
        _execute_trades(ContextInfo, diffs, "盘中")
    except Exception as e:
        log_error("[盘中重算] 异常: {}".format(e))
        log_error(traceback.format_exc())
''')

    # === 18. QMT入口 ===
    lines.append(f'''
# ========================================
#  QMT入口
# ========================================

def init(ContextInfo):
    try:
        log_critical("=" * 60)
        log_critical("{strategy_name} 初始化")
        log_critical("=" * 60)
        ContextInfo.accid = ACCOUNT_ID
        try:
            ContextInfo.set_account(ACCOUNT_ID)
        except Exception as e:
            log_info("set_account可能已设置: {{}}".format(e))
        log_info("账号: {{}}".format(ACCOUNT_ID))
        log_info("模式: {{}}".format(RUN_MODE))
        log_info("实盘资金: {{:.0f}}元".format(REAL_CAPITAL))
        log_info("回测资金: {{:.0f}}元".format(INITIAL_CAPITAL))
        ContextInfo.data_df = None
        ContextInfo.signals = None
        ContextInfo.holdings = None
        ContextInfo.cash = 0
        ContextInfo.last_date = None
        ContextInfo.next_date = None
        ContextInfo.nav_df = None
        try:
            with codecs.open(LOG_FILE_PATH, "w", encoding="gbk", errors="replace") as f:
                f.write("")
        except Exception as e:
            log_info("日志初始化失败: {{}}".format(e))
        log_critical("初始化完成")
    except Exception as e:
        log_error("init异常: {{}}".format(e))
        log_error(traceback.format_exc())
        raise

def after_init(ContextInfo):
    try:
        log_critical("=" * 60)
        log_critical("after_init: 设置定时任务")
        log_critical("  定时模式: {{}}".format("开启" if ENABLE_SCHEDULED_MODE else "关闭"))
        log_critical("  盘中模式: {{}}".format("开启" if ENABLE_INTRADAY_MODE else "关闭"))
        log_critical("=" * 60)
        if RUN_MODE == "live":
            _setup_schedule(ContextInfo)
        else:
            log_info("回测模式，跳过定时任务")
        log_critical("after_init完成")
        log_critical("=" * 60)
        log_critical("启动时自动运行策略计算")
        log_critical("=" * 60)
        _run_calc_once(ContextInfo)
    except Exception as e:
        log_error("after_init异常: {{}}".format(e))
        log_error(traceback.format_exc())
        raise

def handlebar(ContextInfo):
    try:
        if RUN_MODE == "live":
            _fallback_timecheck(ContextInfo)
    except Exception as e:
        log_error("handlebar异常: {{}}".format(e))

def on_strategy_end(ContextInfo):
    today = _today_str()
    now = _now_str()
    log_info("[{{}} {{}}] === 策略结束 ===".format(today, now))
    log_info("账号: {{}}, 模式: {{}}".format(ACCOUNT_ID, RUN_MODE))
    _close_log()
''')

    # === 19. 手动运行 ===
    lines.append(r'''
# ========================================
#  手动运行（不依赖QMT环境）
# ========================================

def _run_calc_once(ContextInfo=None):
    try:
        log_critical("=" * 60)
        log_critical("启动时自动计算策略信号")
        log_critical("=" * 60)
        if ContextInfo is not None:
            try:
                _download_and_save_daily_data(ContextInfo)
            except Exception as e:
                log_error("启动时下载数据失败: {}".format(e))
        pkl_dir = _detect_pkl_dir()
        data_df = _load_data_from_pkl(pkl_dir)
        if data_df is None:
            log_error("数据加载失败，跳过计算")
            return
        data_dict = _build_data_dict(data_df)
        data_dict, common_dates = _align_dates(data_dict)
        stock_tickers = [t for t in data_dict.keys() if t != BOND_TICKER]
        all_tickers = list(data_dict.keys())
        signals = _calc_signals(data_dict, all_tickers)
        nav_df, trade_df, hold_df, holdings, cash = _run_backtest(
            signals, stock_tickers, BOND_TICKER, all_tickers, common_dates,
            INITIAL_CAPITAL, start_date=STRATEGY_START
        )
        last_date = common_dates[-1]
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        _generate_trade_plan(signals, holdings, cash, stock_tickers, last_date, next_date, REAL_CAPITAL, common_dates)
        report = _generate_signal_report(signals, holdings, stock_tickers, last_date, next_date, REAL_CAPITAL, common_dates)
        try:
            with codecs.open(SIGNAL_REPORT_PATH, 'w', encoding='utf-8') as f:
                f.write(report)
            log_critical("信号报告已保存: {}".format(SIGNAL_REPORT_PATH))
        except Exception as e:
            log_error("保存报告失败: {}".format(e))
        log_critical("\n" + report)
        log_critical("=" * 60)
        log_critical("交易计划: {}".format(TRADE_PLAN_CSV))
        log_critical("信号报告: {}".format(SIGNAL_REPORT_PATH))
        log_critical("=" * 60)
        try:
            perf = _compute_performance(nav_df, trade_df, INITIAL_CAPITAL)
            _generate_dashboard(nav_df, trade_df, hold_df, holdings, signals, all_tickers, stock_tickers, perf, cash, REAL_CAPITAL, last_date=last_date)
        except Exception as e:
            log_error("监控面板生成失败: {}".format(e))
    except Exception as e:
        log_error("自动计算异常: {}".format(e))
        log_error(traceback.format_exc())

def manual_run():
    print("=" * 60)
    print("手动运行策略计算")
    print("=" * 60)
    pkl_dir = _detect_pkl_dir()
    data_df = _load_data_from_pkl(pkl_dir)
    if data_df is None:
        print("数据加载失败")
        return
    data_dict = _build_data_dict(data_df)
    data_dict, common_dates = _align_dates(data_dict)
    stock_tickers = [t for t in data_dict.keys() if t != BOND_TICKER]
    all_tickers = list(data_dict.keys())
    signals = _calc_signals(data_dict, all_tickers)
    nav_df, trade_df, hold_df, holdings, cash = _run_backtest(
        signals, stock_tickers, BOND_TICKER, all_tickers, common_dates,
        INITIAL_CAPITAL, start_date=STRATEGY_START
    )
    last_date = common_dates[-1]
    next_date = last_date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)
    rows = _generate_trade_plan(signals, holdings, cash, stock_tickers, last_date, next_date, REAL_CAPITAL, common_dates)
    report = _generate_signal_report(signals, holdings, stock_tickers, last_date, next_date, REAL_CAPITAL, common_dates)
    try:
        with codecs.open(SIGNAL_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(report)
    except Exception as e:
        print("保存报告失败: {}".format(e))
    print("\n" + report)
    print("=" * 60)
    print("交易计划: {}".format(TRADE_PLAN_CSV))
    print("信号报告: {}".format(SIGNAL_REPORT_PATH))
    print("=" * 60)
    try:
        perf = _compute_performance(nav_df, trade_df, INITIAL_CAPITAL)
        _generate_dashboard(nav_df, trade_df, hold_df, holdings, signals, all_tickers, stock_tickers, perf, cash, REAL_CAPITAL, last_date=last_date)
    except Exception as e:
        print("监控面板生成失败: {}".format(e))

if __name__ == "__main__":
    try:
        manual_run()
    except Exception as _e:
        print("运行失败: {}".format(_e))
        traceback.print_exc()
    finally:
        input("\n按回车键退出...")
''')

    # ---- 写入文件 ----
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    content = '\n'.join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    line_count = content.count('\n') + 1
    print(f"[qmt_generator] 已生成: {output_path} ({line_count}行)")
    return output_path
