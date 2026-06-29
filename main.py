#!/usr/bin/env python3
"""
轮动策略回测系统 - 主入口
用法: python main.py [config_file]
"""
import sys
import os
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import fetch_kline, fetch_quotes
from src.backtest_engine import BacktestEngine


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> dict:
    strategy = config['strategy']
    bt = strategy['backtest']
    start_date = bt.get('start_date', '')
    # 截止日期默认为今天（不设或留空即自动取最新）
    end_date = bt.get('end_date', '')
    if not end_date or end_date == '最新':
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 如果起始日期为空或"不限"，自动从最早有数据的日期开始
    # 先获取一个较早的日期来拉取足够数据
    if not start_date or start_date == '不限':
        fetch_start = '2015-01-01'
    else:
        fetch_start = start_date

    start_dt = datetime.strptime(fetch_start, "%Y-%m-%d")
    warmup_start = (start_dt - pd.Timedelta(days=400)).strftime("%Y-%m-%d")

    all_data = {}
    universe = strategy['universe']

    print(f"\n📊 正在获取数据...")
    for item in universe:
        code = item['code']
        name = item['name']
        print(f"  获取 {name}({code})...")

        df = fetch_kline(code, warmup_start, end_date)
        if df.empty:
            print(f"    ⚠️ {name} 数据为空，跳过")
            continue

        df['date'] = pd.to_datetime(df['date'])
        all_data[code] = df
        print(f"    ✅ {len(df)} 条记录")

    benchmark = strategy.get('benchmark')
    if benchmark:
        print(f"  获取基准 {benchmark}...")
        bench_df = fetch_kline(benchmark, warmup_start, end_date)
        if not bench_df.empty:
            bench_df['date'] = pd.to_datetime(bench_df['date'])
            all_data['__benchmark__'] = bench_df
            print(f"    ✅ {len(bench_df)} 条记录")

    # 加载替代资产数据（现金管理用，不参与策略）
    alt = strategy.get('alternative_asset')
    if alt:
        alt_code = alt['code']
        print(f"  获取替代资产 {alt['name']}({alt_code})...")
        alt_df = fetch_kline(alt_code, warmup_start, end_date)
        if not alt_df.empty:
            alt_df['date'] = pd.to_datetime(alt_df['date'])
            all_data[alt_code] = alt_df
            print(f"    ✅ {len(alt_df)} 条记录")

    return all_data


def plot_results(results: dict, output_dir: str):
    df = results['daily_values']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    ax1.plot(df.index, df['nav'], label='策略净值', color='#2196F3', linewidth=1.5)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(df.index, df['nav'], 1, where=(df['nav'] >= 1),
                     alpha=0.1, color='green')
    ax1.fill_between(df.index, df['nav'], 1, where=(df['nav'] < 1),
                     alpha=0.1, color='red')
    ax1.set_title(f"{results['strategy_name']} - 回测结果", fontsize=14, fontweight='bold')
    ax1.set_ylabel('净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    text = f"总收益: {results['total_return']}  |  年化: {results['annual_return']}  |  回撤: {results['max_drawdown']}  |  夏普: {results['sharpe_ratio']}"
    ax1.text(0.02, 0.95, text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]
    ax2.fill_between(df.index, df['drawdown'] * 100, 0, color='red', alpha=0.3)
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(df.index, df['num_positions'], 0, color='#4CAF50', alpha=0.5)
    ax3.set_ylabel('持仓数')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    chart_path = os.path.join(output_dir, 'backtest_result.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📈 净值曲线已保存: {chart_path}")

    return chart_path


def save_trade_log(results: dict, output_dir: str):
    trade_df = results['trade_log']
    if isinstance(trade_df, pd.DataFrame) and not trade_df.empty:
        log_path = os.path.join(output_dir, 'trade_log.csv')
        trade_df.to_csv(log_path, index=False, encoding='utf-8-sig')
        print(f"📋 交易日志已保存: {log_path}")
        return log_path
    return None


def main():
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'config', 'strategy_example.yaml')

    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print(f"请先创建配置文件，参考 config/strategy_example.yaml")
        sys.exit(1)

    print(f"📂 加载配置: {config_path}")
    config = load_config(config_path)

    all_data = prepare_data(config)

    if len(all_data) == 0:
        print("❌ 无有效数据，回测终止")
        sys.exit(1)

    # 自动检测回测起始日期（如果配置为"不限"或空）
    # 逻辑：找到所有标的都有足够数据（60条用于指标预热）的最晚起始日
    bt = config['strategy']['backtest']
    start_date = bt.get('start_date', '')
    if not start_date or start_date == '不限':
        latest_start = None
        for code, df in all_data.items():
            if code.startswith('__'):
                continue
            if len(df) > 60:
                # 第60条数据的日期 = 该标的可参与回测的最早日期
                valid_date = pd.to_datetime(df['date']).iloc[59]
                if latest_start is None or valid_date > latest_start:
                    latest_start = valid_date
        if latest_start:
            auto_start = latest_start.strftime('%Y-%m-%d')
            print(f"📅 起始日期自动检测为: {auto_start} (所有标的均有足够指标预热数据)")
            config['strategy']['backtest']['start_date'] = auto_start

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)

    engine = BacktestEngine(config)
    engine.load_data(all_data)
    results = engine.run()

    chart_path = plot_results(results, output_dir)
    log_path = save_trade_log(results, output_dir)

    print(f"\n✅ 回测完成！结果保存在: {output_dir}")

    return results


if __name__ == '__main__':
    main()
