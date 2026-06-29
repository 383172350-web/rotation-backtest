#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接运行回测验证逻辑"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import BacktestEngine
from data_fetcher import fetch_kline

# 策略配置（直接对应APP参数）
config = {
    "strategy": {
        "name": "DIFv轮动",
        "universe": [
            {"code": "sh512100", "name": "中证500ETF"},
            {"code": "sh513100", "name": "纳斯达克ETF"},
            {"code": "sh513500", "name": "标普500ETF"},
            {"code": "sh518880", "name": "黄金ETF"},
            {"code": "sz159985", "name": "豆粕ETF"},
            {"code": "sz159981", "name": "能源化工ETF"},
            {"code": "sz159980", "name": "有色金属ETF"},
            {"code": "sh513030", "name": "德国ETF"},
            {"code": "sh513520", "name": "日本ETF"},
            {"code": "sh510300", "name": "沪深300ETF"},
            {"code": "sz159949", "name": "创业板ETF"},
            {"code": "sh513050", "name": "中概互联ETF"},
            {"code": "sh501018", "name": "南方原油"},
        ],
        "rank_formula": "(MACD_DIF(12,26,9) / ATR(26)) * 100",
        "rank_direction": "desc",
        "position": {"max_count": 5, "mode": "fixed"},
        "buy_match_mode": "all",
        "buy_rules": [
            {"condition": "close > MA(5)", "description": "close > MA(5)"},
            {"condition": "close > MA(20)", "description": "close > MA(20)"},
            {"condition": "MA(10) > MA(20)", "description": "MA(10) > MA(20)"},
            {"condition": "MA(5) > MA(10)", "description": "MA(5) > MA(10)"},
            {"condition": "(MACD_DIF(12,26,9) / ATR(26)) * 100 < 120", "description": "DIFv < 120"},
            {"condition": "rank < 7", "description": "rank < 7"},
        ],
        "sell_match_mode": "any",
        "sell_rules": [
            {"condition": "rank > 6", "description": "rank > 6"},
            {"condition": "returns(1) < -0.03", "description": "returns(1) < -0.03"},
            {"condition": "returns(20) > 0.25", "description": "returns(20) > 0.25"},
        ],
        "rebalance": {"frequency": "interval", "interval": 2},
        "backtest": {
            "start_date": "2020-01-01",
            "end_date": "2025-06-01",
            "initial_capital": 100000,
            "commission": 0.0001,
            "slippage": 0.001,
        },
        "alternative_asset": {"code": "sh511880", "name": "银华日利"},
        "benchmark": "sh510300",
    }
}

print("="*60)
print("直接回测验证")
print("="*60)

# 下载数据
from data_fetcher import fetch_kline
import pandas as pd

codes = [item["code"] for item in config["strategy"]["universe"]]
alt_code = config["strategy"]["alternative_asset"]["code"]
start = config["strategy"]["backtest"]["start_date"]
end = config["strategy"]["backtest"]["end_date"]

all_codes = codes + [alt_code]
all_data = {}
print(f"\n下载 {len(all_codes)} 只标的K线数据...")
for code in all_codes:
    df = fetch_kline(code, start, end)
    if df is not None and len(df) > 0:
        all_data[code] = df
        print(f"  {code}: {len(df)} 条")
    else:
        print(f"  {code}: 无数据 ⚠️")

if len(all_data) < len(codes) * 0.5:
    print("\n❌ 数据不足，无法回测")
    sys.exit(1)

# 运行回测
print("\n" + "="*60)
print("开始回测")
print("="*60)
engine = BacktestEngine(config)
engine.load_data(all_data)
results = engine.run()

print("\n" + "="*60)
print("回测结果")
print("="*60)
print(f"期间: {results['period']}")
print(f"初始资金: {results['initial_capital']:,}")
print(f"最终市值: {results['final_value']:,.2f}")
print(f"总收益率: {results['total_return']*100:.2f}%")
print(f"年化收益: {results['annual_return']*100:.2f}%")
print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
print(f"夏普比率: {results['sharpe_ratio']}")
print(f"总交易次数: {results['total_trades']}")
print(f"胜率: {results['win_rate']*100:.1f}%")

# 打印交易日志前20条
if not results['trade_log'].empty:
    print("\n--- 交易日志前20条 ---")
    print(results['trade_log'].head(20).to_string())
else:
    print("\n--- 无交易日志 ---")
