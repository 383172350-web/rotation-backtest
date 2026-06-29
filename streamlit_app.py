# ETF轮动策略回测系统 —— Streamlit 可视化版
# 基于 rotation-backtest 技能封装

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import json
import sys
import os
import traceback

# 页面配置
st.set_page_config(
    layout="wide",
    page_title="轮动策略回测系统",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import BacktestEngine
from data_fetcher import fetch_kline

# ============================================================
#  CSS 样式
# ============================================================
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  预设策略
# ============================================================
PRESETS = {
    "自定义": {},
    "全品类DIFv轮动": {
        "universe": "sh513100,纳指ETF\nsh518880,黄金ETF\nsh510300,沪深300ETF\nsh512100,中证1000ETF\nsz159915,创业板ETF\nsh588000,科创50ETF\nsh513500,标普500ETF\nsh513030,德国ETF",
        "rank_formula": "(MACD_DIF(12,26,9) / ATR(26)) * 100",
        "rank_direction": "desc",
        "max_count": 5,
        "position_mode": "fixed",
        "buy_rules": "rank < 6\nclose > MA(20)",
        "sell_rules": "rank > 6\nreturns(1) < -0.03",
        "rebalance_freq": "interval",
        "rebalance_interval": 2,
        "start_date": "2020-01-01",
    },
    "五斗米动量轮动": {
        "universe": "sh510050,上证50ETF\nsh510300,沪深300ETF\nsh588000,科创50ETF\nsz159915,创业板ETF\nsz159531,中证2000ETF",
        "rank_formula": "returns(20)",
        "rank_direction": "desc",
        "max_count": 1,
        "position_mode": "fixed",
        "buy_rules": "close > BOLL_upper(17,2)",
        "sell_rules": "close < BOLL_lower(17,2)",
        "rebalance_freq": "daily",
        "rebalance_interval": 1,
        "start_date": "2020-01-01",
    },
    "精选LOF轮动": {
        "universe": "sz163402,兴全趋势LOF\nsz163417,兴全合宜LOF\nsz161903,万家行业优选LOF\nsz162703,广发小盘LOF\nsz161005,富国天惠LOF",
        "rank_formula": "returns(20) + quality_score(20)",
        "rank_direction": "desc",
        "max_count": 1,
        "position_mode": "fixed",
        "buy_rules": "returns(20) > 0.05",
        "sell_rules": "rank > 1",
        "rebalance_freq": "interval",
        "rebalance_interval": 20,
        "start_date": "2020-01-01",
    },
}

# ============================================================
#  缓存数据
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(codes_json, start_date, end_date):
    codes = json.loads(codes_json)
    all_data = {}
    for item in codes:
        code = item['code']
        try:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            warmup = (start_dt - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
            df = fetch_kline(code, warmup, end_date)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                all_data[code] = df
        except Exception as e:
            st.warning(f"获取 {code} 失败: {e}")
    return all_data

# ============================================================
#  构建配置
# ============================================================
def build_config(form_data):
    strategy = {
        "name": form_data.get("strategy_name", "轮动策略"),
        "universe": form_data["universe"],
        "rank_formula": form_data["rank_formula"],
        "rank_direction": form_data["rank_direction"],
        "position": {
            "max_count": form_data["max_count"],
            "mode": form_data["position_mode"],
        },
        "buy_match_mode": "all" if form_data.get("buy_all", True) else "any",
        "buy_rules": [],
        "sell_match_mode": "any" if form_data.get("sell_any", True) else "all",
        "sell_rules": [],
        "rebalance": {
            "frequency": form_data["rebalance_freq"],
            "interval": form_data.get("rebalance_interval", 2),
        },
        "backtest": {
            "start_date": form_data["start_date"],
            "end_date": form_data.get("end_date", datetime.datetime.now().strftime("%Y-%m-%d")),
            "initial_capital": form_data["initial_capital"],
            "commission": form_data.get("commission", 0.0001),
            "slippage": form_data.get("slippage", 0.001),
        },
        "benchmark": form_data.get("benchmark", ""),
    }
    for i, rule in enumerate(form_data.get("buy_rules", [])):
        if rule.strip():
            strategy["buy_rules"].append({"condition": rule, "description": f"买入{i+1}"})
    for i, rule in enumerate(form_data.get("sell_rules", [])):
        if rule.strip():
            strategy["sell_rules"].append({"condition": rule, "description": f"卖出{i+1}"})
    if form_data.get("alternative_asset"):
        strategy["alternative_asset"] = {"code": form_data["alternative_asset"], "name": "替代资产"}
    return {"strategy": strategy}

# ============================================================
#  主函数
# ============================================================
def main():
    st.markdown('<div class="main-header">📊 轮动策略回测系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">完全可配置的股票轮动策略回测 · 自定义排序指标 · 自定义买卖条件</div>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ 策略配置")
        
        preset = st.selectbox("选择预设策略", list(PRESETS.keys()))
        preset_data = PRESETS[preset]
        
        strategy_name = st.text_input("策略名称", value=preset_data.get("strategy_name", "我的轮动策略"))
        
        st.divider()
        st.subheader("📋 股票池")
        universe_text = st.text_area(
            "代码,名称（每行一个）",
            value=preset_data.get("universe", "sh513100,纳指ETF\nsh518880,黄金ETF\nsh510300,沪深300ETF"),
            height=100
        )
        universe = []
        for line in universe_text.strip().split("\n"):
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            if len(parts) >= 2:
                universe.append({"code": parts[0].strip(), "name": parts[1].strip()})
            elif len(parts) == 1:
                universe.append({"code": parts[0].strip(), "name": parts[0].strip()})
        
        st.divider()
        st.subheader("📊 排序公式")
        rank_formula = st.text_area("公式", value=preset_data.get("rank_formula", "returns(20)"), height=50)
        rank_direction = st.radio("方向", ["desc", "asc"], index=0 if preset_data.get("rank_direction", "desc") == "desc" else 1,
                                   format_func=lambda x: "越大越好" if x == "desc" else "越小越好")
        
        st.divider()
        st.subheader("💰 持仓")
        max_count = st.number_input("最多持有", min_value=1, max_value=20, value=preset_data.get("max_count", 5))
        position_mode = st.radio("模式", ["fixed", "adaptive"], index=0,
                                   format_func=lambda x: "固定均分" if x == "fixed" else "动态均分")
        
        st.divider()
        st.subheader("🟢 买入")
        buy_all = st.radio("模式", ["all", "any"], index=0, format_func=lambda x: "全部满足" if x == "all" else "满足一条",
                           key="buy_mode")
        buy_rules_text = st.text_area("条件（每行一个）", value=preset_data.get("buy_rules", "rank < 6\nclose > MA(20)"), height=60)
        buy_rules = [r.strip() for r in buy_rules_text.strip().split("\n") if r.strip()]
        
        st.divider()
        st.subheader("🔴 卖出")
        sell_any = st.radio("模式", ["any", "all"], index=0, format_func=lambda x: "满足一条就卖" if x == "any" else "全部满足才卖",
                            key="sell_mode")
        sell_rules_text = st.text_area("条件（每行一个）", value=preset_data.get("sell_rules", "rank > 6\nreturns(1) < -0.03"), height=60)
        sell_rules = [r.strip() for r in sell_rules_text.strip().split("\n") if r.strip()]
        
        st.divider()
        st.subheader("🔄 轮动")
        rebalance_freq = st.selectbox("频率", ["daily", "interval", "weekly", "monthly"],
                                       index=["daily", "interval", "weekly", "monthly"].index(preset_data.get("rebalance_freq", "interval")) if preset_data.get("rebalance_freq") in ["daily", "interval", "weekly", "monthly"] else 1,
                                       format_func=lambda x: {"daily": "每天", "interval": "按间隔", "weekly": "每周", "monthly": "每月"}[x])
        rebalance_interval = st.number_input("间隔天数", min_value=1, max_value=30, value=preset_data.get("rebalance_interval", 2)) if rebalance_freq == "interval" else 1
        
        st.divider()
        st.subheader("📅 回测")
        start_date = st.date_input("开始", value=datetime.datetime.strptime(preset_data.get("start_date", "2020-01-01"), "%Y-%m-%d"))
        end_date = st.date_input("结束", value=datetime.datetime.now())
        initial_capital = st.number_input("初始资金", min_value=10000, value=100000, step=10000)
        commission = st.number_input("手续费", min_value=0.0, max_value=0.01, value=0.0001, format="%.4f")
        slippage = st.number_input("滑点", min_value=0.0, max_value=0.05, value=0.001, format="%.3f")
        benchmark = st.text_input("基准", value="sh510300")
        
        st.divider()
        run_btn = st.button("🚀 运行回测", type="primary", use_container_width=True)
    
    # ---------- 主页面 ----------
    if run_btn:
        if not universe:
            st.error("股票池为空！")
            return
        
        with st.spinner("获取数据并运行回测..."):
            try:
                form_data = {
                    "strategy_name": strategy_name,
                    "universe": universe,
                    "rank_formula": rank_formula,
                    "rank_direction": rank_direction,
                    "max_count": max_count,
                    "position_mode": position_mode,
                    "buy_all": buy_all == "all",
                    "buy_rules": buy_rules,
                    "sell_any": sell_any == "any",
                    "sell_rules": sell_rules,
                    "rebalance_freq": rebalance_freq,
                    "rebalance_interval": rebalance_interval,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "initial_capital": initial_capital,
                    "commission": commission,
                    "slippage": slippage,
                    "benchmark": benchmark,
                }
                
                config = build_config(form_data)
                
                # 获取数据
                codes_json = json.dumps(universe)
                all_data = get_data(codes_json, form_data["start_date"], form_data["end_date"])
                
                # 获取基准
                if benchmark:
                    try:
                        bench_df = fetch_kline(benchmark, form_data["start_date"], form_data["end_date"])
                        if not bench_df.empty:
                            bench_df['date'] = pd.to_datetime(bench_df['date'])
                            all_data['__benchmark__'] = bench_df
                    except: pass
                
                if not all_data:
                    st.error("未能获取数据，请检查代码！")
                    return
                
                # 运行回测
                engine = BacktestEngine(config)
                engine.load_data(all_data)
                results = engine.run()
                st.session_state.results = results
                st.session_state.config = config
                st.success("✅ 回测完成！")
                
            except Exception as e:
                st.error(f"回测失败: {str(e)}")
                st.code(traceback.format_exc())
                return
    
    # 展示结果
    if st.session_state.results:
        show_results(st.session_state.results)
    else:
        show_guide()


def show_results(results):
    """展示结果"""
    st.divider()
    st.subheader("📈 统计指标")
    
    cols = st.columns(5)
    metrics = [
        ("总收益率", f"{results.get('total_return', 0)*100:.2f}%", "#4CAF50" if results.get('total_return', 0) > 0 else "#F44336"),
        ("年化收益", f"{results.get('annual_return', 0)*100:.2f}%", "#4CAF50" if results.get('annual_return', 0) > 0 else "#F44336"),
        ("最大回撤", f"{results.get('max_drawdown', 0)*100:.2f}%", "#FF9800"),
        ("夏普比率", f"{results.get('sharpe_ratio', 0):.2f}", "#2196F3"),
        ("胜率", f"{results.get('win_rate', 0)*100:.1f}%", "#2196F3"),
    ]
    for i, (label, value, color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f'<div style="background:{color};padding:1rem;border-radius:8px;text-align:center;color:white;"><div style="font-size:1.6rem;font-weight:bold;">{value}</div><div style="font-size:0.8rem;">{label}</div></div>', unsafe_allow_html=True)
    
    # 净值曲线
    st.divider()
    st.subheader("📉 净值曲线")
    if 'daily_values' in results and not results['daily_values'].empty:
        df = results['daily_values']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.6, 0.2, 0.2], subplot_titles=('净值', '回撤', '持仓'))
        fig.add_trace(go.Scatter(x=df.index, y=df['nav'], name='策略', line=dict(color='#2196F3', width=1.5)), row=1, col=1)
        if 'benchmark_nav' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['benchmark_nav'], name='基准', line=dict(color='#FF9800', width=1.5)), row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], name='回撤', fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red', width=0.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['hold_count'], name='持仓数', line=dict(color='green', width=1)), row=3, col=1)
        fig.update_layout(height=650, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # 交易日志
    st.divider()
    st.subheader("📝 交易日志")
    if 'trade_log' in results and results['trade_log']:
        trades = pd.DataFrame(results['trade_log'])
        st.dataframe(trades, use_container_width=True)
        csv = trades.to_csv(index=False).encode('utf-8')
        st.download_button("下载交易日志", csv, "trades.csv", "text/csv")
    
    # 持仓
    st.divider()
    st.subheader("💼 当前持仓")
    if 'positions' in results and results['positions']:
        pos = []
        for code, p in results['positions'].items():
            pos.append({'代码': code, '名称': p.get('name', code), '数量': p.get('shares', 0),
                        '成本': f"{p.get('cost_price', 0):.3f}", '现价': f"{p.get('current_price', 0):.3f}",
                        '市值': f"{p.get('market_value', 0):.2f}", '收益率': f"{p.get('profit_pct', 0)*100:.2f}%"})
        st.dataframe(pd.DataFrame(pos), use_container_width=True)
    else:
        st.info("当前无持仓")


def show_guide():
    """使用说明"""
    st.info("👈 请在左侧配置策略参数，点击 **运行回测** 开始")
    with st.expander("📖 使用指南"):
        st.markdown("""
        **快速上手**：选择预设策略 → 调整参数 → 运行回测 → 查看结果
        
        **系统指标**：MA/EMA/RSI/MACD/ATR/BOLL/KDJ/returns/RSRS/quality_score等
        
        **排序公式**：`returns(20)` 或 `(MACD_DIF(12,26,9) / ATR(26)) * 100`
        
        **买卖条件**：`close > MA(20)` / `rank < 6` / `returns(1) < -0.03`
        
        **逻辑组合**：`AND` 全部满足 / `OR` 满足任一
        """)
    with st.expander("⚠️ 注意事项"):
        st.markdown("- 指标需约60日预热\n- T+1模式：收盘信号，次日开盘成交\n- A股代码格式：sh+代码（上海）/ sz+代码（深圳）")


if __name__ == "__main__":
    main()
