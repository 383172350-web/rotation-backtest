"""
本地pkl数据回测脚本
使用技能文件中的回测引擎，加载本地pkl数据运行回测
"""
import sys
import os
import pickle
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import BacktestEngine


# ========== 原始DIFv轮动策略配置 ==========
CONFIG = {
    'strategy': {
        'name': 'DIFv轮动双模式',
        'universe': [
            {'code': 'sh512100', 'name': '中证1000ETF'},
            {'code': 'sh513100', 'name': '纳指ETF'},
            {'code': 'sh513500', 'name': '标普500ETF'},
            {'code': 'sh518880', 'name': '黄金ETF'},
            {'code': 'sz159985', 'name': '豆粕ETF'},
            {'code': 'sz159981', 'name': '能源化工ETF'},
            {'code': 'sz159980', 'name': '有色ETF'},
            {'code': 'sh513030', 'name': '德国ETF'},
            {'code': 'sh513520', 'name': '日经ETF'},
            {'code': 'sh510300', 'name': '沪深300ETF'},
            {'code': 'sz159949', 'name': '创业板50ETF'},
            {'code': 'sh513050', 'name': '中概互联网ETF'},
            {'code': 'sh501018', 'name': '南方原油LOF'},
        ],
        'rank_formula': '(MACD_DIF(12,26,9) / ATR(26)) * 100',
        'rank_direction': 'desc',
        'position': {
            'max_count': 5,
            'mode': 'fixed',
        },
        'buy_match_mode': 'all',
        'buy_rules': [
            {'condition': 'close > MA(5)', 'description': '收盘价站上5日均线'},
            {'condition': 'close > MA(20)', 'description': '收盘价站上20日均线'},
            {'condition': 'MA(10) > MA(20)', 'description': '10日均线在20日均线上方'},
            {'condition': 'MA(5) > MA(10)', 'description': '5日均线在10日均线上方'},
            {'condition': 'MACD_DIF(12,26,9) / ATR(26) * 100 < 120', 'description': 'DIFv小于120'},
            {'condition': 'rank < 7', 'description': '排名在前6名'},
        ],
        'sell_match_mode': 'any',
        'sell_rules': [
            {'condition': 'rank > 6', 'description': '排名跌出前6'},
            {'condition': 'returns(1) < -0.03', 'description': '单日跌幅超3%止损'},
            {'condition': 'returns(20) > 0.25', 'description': '20日涨幅超25%止盈'},
        ],
        'alternative_asset': {
            'code': 'sh511880',
            'name': '银华日利ETF',
        },
        'rebalance': {
            'frequency': 'interval',
            'interval': 2,
        },
        'backtest': {
            'start_date': '2020-01-01',
            'initial_capital': 100000,
            'commission': 0.0001,
            'slippage': 0.001,
        },
    }
}


def code_to_filename(code: str) -> str:
    """将sh512690格式转为512690_SH_1d.pkl"""
    prefix = code[:2].lower()
    num = code[2:]
    if prefix == 'sh':
        return f"{num}_SH_1d.pkl"
    elif prefix == 'sz':
        return f"{num}_SZ_1d.pkl"
    return f"{num}_{prefix.upper()}_1d.pkl"


def load_local_data(data_dir: str, universe_codes: list) -> dict:
    """从本地pkl加载数据"""
    all_data = {}
    
    for code in universe_codes:
        filename = code_to_filename(code)
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  [WARN] 文件不存在: {filepath}")
            continue
        
        try:
            df = pickle.load(open(filepath, 'rb'))
            
            # 处理索引：stime (如 20150105) 转为 date
            df = df.copy()
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d')
            df.index.name = 'date'
            
            # 去掉前导0值（数据对齐填充）
            df = df[df['close'] > 0].copy()
            
            # 重置索引，让date成为列（backtest_engine.load_data期望有date列）
            df = df.reset_index()
            
            if df.empty:
                print(f"  [WARN] {code} 数据为空（去0后）")
                continue
                
            all_data[code] = df
            print(f"  [OK] {code}: {len(df)} 条记录, {df['date'].min()} ~ {df['date'].max()}")
        except Exception as e:
            print(f"  [ERR] {code} 加载失败: {e}")
    
    return all_data


def main():
    data_dir = r'D:\qmt_data\ETF\1d'
    
    universe_codes = [item['code'] for item in CONFIG['strategy']['universe']]
    alt_code = CONFIG['strategy']['alternative_asset']['code']
    
    print(f"\n[加载本地pkl数据] from {data_dir}")
    all_data = load_local_data(data_dir, universe_codes + [alt_code])
    
    if len(all_data) < len(universe_codes):
        print(f"❌ 只加载了 {len(all_data)} 只标的，需要 {len(universe_codes)} 只")
        return
    
    # 自动检测回测起始日期
    # 找到所有标的都有足够数据（60条用于指标预热）的最晚起始日
    bt = CONFIG['strategy']['backtest']
    start_date = bt.get('start_date', '')
    if not start_date or start_date == '自动':
        latest_start = None
        for code, df in all_data.items():
            if code == alt_code:
                continue
            df_sorted = df.sort_values('date')
            if len(df_sorted) > 60:
                valid_date = pd.to_datetime(df_sorted['date'].iloc[59])
                if latest_start is None or valid_date > latest_start:
                    latest_start = valid_date
        if latest_start:
            auto_start = latest_start.strftime('%Y-%m-%d')
            print(f"[INFO] 起始日期自动检测为: {auto_start}")
            CONFIG['strategy']['backtest']['start_date'] = auto_start
    
    engine = BacktestEngine(CONFIG)
    engine.load_data(all_data)
    results = engine.run()
    
    print("\n" + "="*60)
    print("回测结果摘要")
    print("="*60)
    print(f"策略: {results['strategy_name']}")
    print(f"期间: {results['period']}")
    print(f"总收益率: {results['total_return']}")
    print(f"年化收益: {results['annual_return']}")
    print(f"最大回撤: {results['max_drawdown']}")
    print(f"夏普比率: {results['sharpe_ratio']}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"胜率: {results['win_rate']}")
    
    # 输出前10笔交易
    trade_df = results['trade_log']
    if not trade_df.empty:
        print("\n[最近交易记录]:")
        print(trade_df.tail(10).to_string())


if __name__ == '__main__':
    main()
