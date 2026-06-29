# -*- coding: utf-8 -*-
"""
轮动策略回测系统 —— 可视化网页版
基于 rotation-backtest 技能封装
支持：1912只ETF+LOF标的池可视化选择、排序公式构建器、买卖规则构建器
"""
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import BacktestEngine
from data_fetcher import fetch_kline

st.set_page_config(layout="wide", page_title="轮动策略回测系统", page_icon="📊", initial_sidebar_state="expanded")

# ============================================================
#  CSS
# ============================================================
st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
.sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
.stButton>button { border-radius: 8px; font-weight: 600; }
.rule-box { background: #f8f9fa; padding: 10px 14px; border-radius: 8px; margin: 6px 0; border-left: 4px solid #4CAF50; }
.rule-box-sell { border-left-color: #F44336; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  加载标的池数据
# ============================================================
@st.cache_data
def load_pool():
    with open(os.path.join(os.path.dirname(__file__), "etf_pool.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['代码'] = df['代码'].astype(str).str.strip()
    df['名称'] = df['名称'].astype(str).str.strip()
    return df

POOL_DF = load_pool()

# ============================================================
#  系统指标库
# ============================================================
INDICATORS = {
    "MA(n)": {"name": "均线MA", "params": [{"name": "n", "label": "周期", "default": 20, "min": 1, "max": 250}]},
    "EMA(n)": {"name": "指数均线EMA", "params": [{"name": "n", "label": "周期", "default": 12, "min": 1, "max": 250}]},
    "RSI(n)": {"name": "RSI", "params": [{"name": "n", "label": "周期", "default": 14, "min": 1, "max": 100}]},
    "MACD_DIF(fast,slow,signal)": {"name": "MACD快线", "params": [
        {"name": "fast", "label": "快线", "default": 12, "min": 1, "max": 50},
        {"name": "slow", "label": "慢线", "default": 26, "min": 1, "max": 50},
        {"name": "signal", "label": "信号线", "default": 9, "min": 1, "max": 50}]},
    "MACD_DEA(fast,slow,signal)": {"name": "MACD慢线", "params": [
        {"name": "fast", "label": "快线", "default": 12, "min": 1, "max": 50},
        {"name": "slow", "label": "慢线", "default": 26, "min": 1, "max": 50},
        {"name": "signal", "label": "信号线", "default": 9, "min": 1, "max": 50}]},
    "MACD_HIST(fast,slow,signal)": {"name": "MACD柱", "params": [
        {"name": "fast", "label": "快线", "default": 12, "min": 1, "max": 50},
        {"name": "slow", "label": "慢线", "default": 26, "min": 1, "max": 50},
        {"name": "signal", "label": "信号线", "default": 9, "min": 1, "max": 50}]},
    "ATR(n)": {"name": "ATR波动", "params": [{"name": "n", "label": "周期", "default": 26, "min": 1, "max": 100}]},
    "BOLL(n)": {"name": "布林带中轨", "params": [{"name": "n", "label": "周期", "default": 20, "min": 1, "max": 100}]},
    "BOLL_upper(n,std)": {"name": "布林上轨", "params": [
        {"name": "n", "label": "周期", "default": 20, "min": 1, "max": 100},
        {"name": "std", "label": "标准差倍数", "default": 2, "min": 1, "max": 5}]},
    "BOLL_lower(n,std)": {"name": "布林下轨", "params": [
        {"name": "n", "label": "周期", "default": 20, "min": 1, "max": 100},
        {"name": "std", "label": "标准差倍数", "default": 2, "min": 1, "max": 5}]},
    "KDJ_K(n,m1,m2)": {"name": "KDJ-K", "params": [
        {"name": "n", "label": "N日", "default": 9, "min": 1, "max": 50},
        {"name": "m1", "label": "M1", "default": 3, "min": 1, "max": 20},
        {"name": "m2", "label": "M2", "default": 3, "min": 1, "max": 20}]},
    "KDJ_D(n,m1,m2)": {"name": "KDJ-D", "params": [
        {"name": "n", "label": "N日", "default": 9, "min": 1, "max": 50},
        {"name": "m1", "label": "M1", "default": 3, "min": 1, "max": 20},
        {"name": "m2", "label": "M2", "default": 3, "min": 1, "max": 20}]},
    "KDJ_J(n,m1,m2)": {"name": "KDJ-J", "params": [
        {"name": "n", "label": "N日", "default": 9, "min": 1, "max": 50},
        {"name": "m1", "label": "M1", "default": 3, "min": 1, "max": 20},
        {"name": "m2", "label": "M2", "default": 3, "min": 1, "max": 20}]},
    "returns(n)": {"name": "N日涨幅", "params": [{"name": "n", "label": "天数", "default": 20, "min": 1, "max": 250}]},
    "BIAS(n)": {"name": "乖离率", "params": [{"name": "n", "label": "周期", "default": 20, "min": 1, "max": 250}]},
    "quality_score(n)": {"name": "质量得分", "params": [{"name": "n", "label": "周期", "default": 20, "min": 1, "max": 250}]},
    "volatility(n)": {"name": "波动率", "params": [{"name": "n", "label": "周期", "default": 20, "min": 1, "max": 250}]},
    "gain_percentile(n)": {"name": "涨幅百分位", "params": [{"name": "n", "label": "周期", "default": 250, "min": 1, "max": 500}]},
    "volume_percentile(n)": {"name": "成交量百分位", "params": [{"name": "n", "label": "周期", "default": 250, "min": 1, "max": 500}]},
    "RSRS_slope(n)": {"name": "RSRS斜率", "params": [{"name": "n", "label": "周期", "default": 18, "min": 1, "max": 100}]},
    "RSRS_zscore(n)": {"name": "RSRS标准分", "params": [{"name": "n", "label": "周期", "default": 18, "min": 1, "max": 100}]},
    "RSRS_right_zscore(n)": {"name": "RSRS右偏标准分", "params": [{"name": "n", "label": "周期", "default": 18, "min": 1, "max": 100}]},
}

BASIC_FIELDS = {
    "close": "收盘价",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "volume": "成交量",
    "amount": "成交额",
}

SPECIAL_VARS = {
    "rank": "当前排名",
    "profit": "持仓收益率",
    "hold_days": "持仓天数",
    "buy_price": "买入价格",
}

OPS = [">", "<", ">=", "<=", "==", "!="]

# ============================================================
#  预设策略
# ============================================================
PRESETS = {
    "🎯 自定义策略": {},
    "📈 全品类DIFv轮动": {
        "selected_codes": ["sh513100", "sh518880", "sh510300", "sh512100", "sz159915", "sh588000", "sh513500", "sh513030", "sh513520", "sz159980", "sz159981", "sz159985", "sh501018"],
        "alternative_asset": "sh511880",
        "rank_formula": "(MACD_DIF(12,26,9) / ATR(26)) * 100",
        "rank_direction": "desc",
        "max_count": 5, "position_mode": "fixed",
        "buy_rules": ["rank < 6", "close > MA(20)"],
        "sell_rules": ["rank > 6", "returns(1) < -0.03"],
        "rebalance_freq": "interval", "rebalance_interval": 2,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "📊 五斗米动量轮动": {
        "selected_codes": ["sh510050", "sh510300", "sh588000", "sz159915", "sz159531"],
        "rank_formula": "returns(20)",
        "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_rules": ["close > BOLL_upper(17,2)"],
        "sell_rules": ["close < BOLL_lower(17,2)"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "🏆 精选LOF轮动": {
        "selected_codes": ["sz163402", "sz163417", "sz161903", "sz162703", "sz161005"],
        "rank_formula": "returns(20) + quality_score(20)",
        "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_rules": ["returns(20) > 0.05"],
        "sell_rules": ["rank > 1"],
        "rebalance_freq": "interval", "rebalance_interval": 20,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "🔮 动量+RSRS轮动": {
        "selected_codes": ["sh518880", "sh513100", "sh588220", "sz159915", "sh511090"],
        "rank_formula": "RSRS_zscore(18)",
        "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_rules": ["RSRS_zscore(18) > 0.7"],
        "sell_rules": ["RSRS_zscore(18) < -0.7"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
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
            if not df.empty and len(df) > 60:
                df['date'] = pd.to_datetime(df['date'])
                all_data[code] = df
            else:
                st.warning(f"{code} 数据不足或为空，已跳过")
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
        "position": {"max_count": form_data["max_count"], "mode": form_data["position_mode"]},
        "buy_match_mode": "all" if form_data.get("buy_all", True) else "any",
        "buy_rules": [],
        "sell_match_mode": "any" if form_data.get("sell_any", True) else "all",
        "sell_rules": [],
        "rebalance": {"frequency": form_data["rebalance_freq"], "interval": form_data.get("rebalance_interval", 2)},
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
        if rule.strip(): strategy["buy_rules"].append({"condition": rule, "description": f"买入{i+1}"})
    for i, rule in enumerate(form_data.get("sell_rules", [])):
        if rule.strip(): strategy["sell_rules"].append({"condition": rule, "description": f"卖出{i+1}"})
    if form_data.get("alternative_asset"):
        strategy["alternative_asset"] = {"code": form_data["alternative_asset"], "name": "替代资产"}
    return {"strategy": strategy}


# ============================================================
#  指标参数渲染器
# ============================================================
def render_indicator_params(key_prefix, selected_indicator):
    """渲染指标参数输入框，返回填充后的指标字符串"""
    if not selected_indicator or selected_indicator not in INDICATORS:
        return ""
    info = INDICATORS[selected_indicator]
    params = {}
    cols = st.columns(len(info["params"]))
    for i, p in enumerate(info["params"]):
        with cols[i]:
            params[p["name"]] = st.number_input(
                p["label"], min_value=p["min"], max_value=p["max"], value=p["default"],
                key=f"{key_prefix}_param_{p['name']}"
            )
    # 构建函数字符串
    param_str = ",".join(str(params[p["name"]]) for p in info["params"])
    # 提取函数名（去掉括号部分）
    func_name = selected_indicator.split("(")[0]
    return f"{func_name}({param_str})"


# ============================================================
#  规则构建器
# ============================================================
def rule_builder(key_prefix, existing_rules, title, color="green"):
    """可视化规则构建器，返回规则列表"""
    st.markdown(f"**{title}**")
    rules = list(existing_rules)

    # 显示已有规则
    for i, rule in enumerate(rules):
        c1, c2 = st.columns([8, 1])
        with c1:
            st.markdown(f'<div class="rule-box" style="border-left-color: {"#4CAF50" if color=="green" else "#F44336"};">{rule}</div>', unsafe_allow_html=True)
        with c2:
            if st.button("🗑️", key=f"{key_prefix}_del_{i}"):
                rules.pop(i)
                st.session_state[f"{key_prefix}_rules"] = rules
                st.rerun()

    # 添加新规则
    with st.expander("➕ 添加规则"):
        # 选择左值类型
        left_type = st.radio("左值类型", ["系统指标", "基础字段", "特殊变量", "手动输入"], horizontal=True, key=f"{key_prefix}_left_type")

        left_val = ""
        if left_type == "系统指标":
            ind = st.selectbox("选择指标", list(INDICATORS.keys()), format_func=lambda x: f"{INDICATORS[x]['name']} ({x})", key=f"{key_prefix}_left_ind")
            left_val = render_indicator_params(f"{key_prefix}_left", ind)
        elif left_type == "基础字段":
            left_val = st.selectbox("选择字段", list(BASIC_FIELDS.keys()), format_func=lambda x: BASIC_FIELDS[x], key=f"{key_prefix}_left_field")
        elif left_type == "特殊变量":
            left_val = st.selectbox("选择变量", list(SPECIAL_VARS.keys()), format_func=lambda x: SPECIAL_VARS[x], key=f"{key_prefix}_left_var")
        else:
            left_val = st.text_input("输入表达式", value="close", key=f"{key_prefix}_left_manual")

        op = st.selectbox("运算符", OPS, key=f"{key_prefix}_op")

        # 右值类型
        right_type = st.radio("右值类型", ["数值", "系统指标", "基础字段", "特殊变量", "手动输入"], horizontal=True, key=f"{key_prefix}_right_type")
        right_val = ""
        if right_type == "数值":
            right_val = str(st.number_input("数值", value=0.0, step=0.01, key=f"{key_prefix}_right_num"))
        elif right_type == "系统指标":
            ind_r = st.selectbox("选择指标", list(INDICATORS.keys()), format_func=lambda x: f"{INDICATORS[x]['name']} ({x})", key=f"{key_prefix}_right_ind")
            right_val = render_indicator_params(f"{key_prefix}_right", ind_r)
        elif right_type == "基础字段":
            right_val = st.selectbox("选择字段", list(BASIC_FIELDS.keys()), format_func=lambda x: BASIC_FIELDS[x], key=f"{key_prefix}_right_field")
        elif right_type == "特殊变量":
            right_val = st.selectbox("选择变量", list(SPECIAL_VARS.keys()), format_func=lambda x: SPECIAL_VARS[x], key=f"{key_prefix}_right_var")
        else:
            right_val = st.text_input("输入表达式", value="MA(20)", key=f"{key_prefix}_right_manual")

        if st.button("添加此规则", key=f"{key_prefix}_add"):
            new_rule = f"{left_val} {op} {right_val}"
            rules.append(new_rule)
            st.session_state[f"{key_prefix}_rules"] = rules
            st.rerun()

    return rules


# ============================================================
#  排序公式构建器
# ============================================================
def rank_formula_builder(key_prefix, current_formula):
    """可视化排序公式构建器"""
    st.markdown("**📊 排序公式构建器**")
    st.caption("构建打分公式，分数越高排名越靠前（desc模式）")

    formula_parts = []

    # 使用session_state保存公式片段
    if f"{key_prefix}_formula_parts" not in st.session_state:
        st.session_state[f"{key_prefix}_formula_parts"] = []

    parts = st.session_state[f"{key_prefix}_formula_parts"]

    # 显示已有片段
    if parts:
        st.code(" + ".join(parts) if len(parts) > 1 else parts[0], language="python")

    with st.expander("➕ 添加公式项"):
        # 选择项类型
        item_type = st.radio("项类型", ["指标项", "加权项", "手动输入"], horizontal=True, key=f"{key_prefix}_item_type")

        if item_type == "指标项":
            ind = st.selectbox("选择指标", list(INDICATORS.keys()), format_func=lambda x: f"{INDICATORS[x]['name']} ({x})", key=f"{key_prefix}_rank_ind")
            expr = render_indicator_params(f"{key_prefix}_rank", ind)
            if st.button("添加指标项", key=f"{key_prefix}_add_ind"):
                parts.append(expr)
                st.session_state[f"{key_prefix}_formula_parts"] = parts
                st.rerun()

        elif item_type == "加权项":
            c1, c2 = st.columns([3, 1])
            with c1:
                ind = st.selectbox("选择指标", list(INDICATORS.keys()), format_func=lambda x: f"{INDICATORS[x]['name']} ({x})", key=f"{key_prefix}_rank_wind")
                expr = render_indicator_params(f"{key_prefix}_rank_w", ind)
            with c2:
                weight = st.number_input("权重", value=1.0, step=0.1, key=f"{key_prefix}_weight")
            if st.button("添加加权项", key=f"{key_prefix}_add_weight"):
                parts.append(f"({expr} * {weight})")
                st.session_state[f"{key_prefix}_formula_parts"] = parts
                st.rerun()

        else:  # 手动输入
            manual = st.text_input("手动输入表达式", value="returns(20)", key=f"{key_prefix}_manual")
            if st.button("添加手动项", key=f"{key_prefix}_add_manual"):
                parts.append(manual)
                st.session_state[f"{key_prefix}_formula_parts"] = parts
                st.rerun()

    # 最终公式
    if parts:
        final_formula = " + ".join(parts) if len(parts) > 1 else parts[0]
    else:
        final_formula = current_formula

    # 允许手动编辑
    final_formula = st.text_input("排序公式（可手动编辑）", value=final_formula, key=f"{key_prefix}_final")

    # 清空按钮
    if st.button("🗑️ 清空公式", key=f"{key_prefix}_clear"):
        st.session_state[f"{key_prefix}_formula_parts"] = []
        st.rerun()

    return final_formula


# ============================================================
#  标的池选择器
# ============================================================
def stock_pool_selector(key_prefix, selected_codes):
    """可视化标的池选择器"""
    st.markdown("**📋 标的池选择器**")

    df = POOL_DF.copy()

    # 搜索
    search = st.text_input("🔍 搜索（名称/代码）", "", key=f"{key_prefix}_search")
    if search:
        mask = df['名称'].str.contains(search, case=False, na=False) | df['代码'].str.contains(search, case=False, na=False)
        df = df[mask]

    # 分类筛选
    categories = sorted(df['分类'].unique().tolist())
    selected_cats = st.multiselect("分类筛选", categories, default=[], key=f"{key_prefix}_cats")
    if selected_cats:
        df = df[df['分类'].isin(selected_cats)]

    # 类型筛选
    types = sorted(df['类型'].unique().tolist())
    selected_types = st.multiselect("类型筛选", types, default=types, key=f"{key_prefix}_types")
    if selected_types:
        df = df[df['类型'].isin(selected_types)]

    st.caption(f"共 {len(df)} 只标的")

    # 表格多选
    # 添加选中状态列
    df['选中'] = df['代码'].isin(selected_codes)

    # 显示表格（使用data_editor支持多选）
    edited_df = st.data_editor(
        df[['选中', '代码', '名称', '分类', '类型']],
        column_config={"选中": st.column_config.CheckboxColumn("选中", default=False)},
        hide_index=True,
        use_container_width=True,
        height=300,
        key=f"{key_prefix}_editor"
    )

    # 获取选中的代码
    new_selected = edited_df[edited_df['选中'] == True]['代码'].tolist()

    # 已选列表
    if new_selected:
        st.markdown(f"**已选 {len(new_selected)} 只：**")
        selected_df = POOL_DF[POOL_DF['代码'].isin(new_selected)][['代码', '名称', '分类']]
        st.dataframe(selected_df, hide_index=True, use_container_width=True)
    else:
        st.info("尚未选择任何标的")

    return new_selected


# ============================================================
#  主函数
# ============================================================
def main():
    st.markdown('<div class="main-header">📊 轮动策略回测系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">1912只ETF+LOF · 可视化策略构建 · 自定义排序指标 · 自定义买卖规则</div>', unsafe_allow_html=True)

    if 'results' not in st.session_state: st.session_state.results = None
    if 'config' not in st.session_state: st.session_state.config = None

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ 策略配置")

        preset = st.selectbox("选择预设策略", list(PRESETS.keys()), key="preset_select")
        preset_data = PRESETS[preset] if preset != "🎯 自定义策略" else {}

        strategy_name = st.text_input("策略名称", value=preset_data.get("strategy_name", "我的轮动策略"), key="strategy_name")

        st.divider()
        st.subheader("📋 股票池")
        # 标的池选择
        init_codes = preset_data.get("selected_codes", [])
        selected_codes = stock_pool_selector("pool", init_codes)

        universe = []
        for code in selected_codes:
            row = POOL_DF[POOL_DF['代码'] == code]
            if not row.empty:
                universe.append({"code": code, "name": row.iloc[0]['名称']})

        st.divider()
        st.subheader("📊 排序公式")
        rank_formula = rank_formula_builder("rank", preset_data.get("rank_formula", "returns(20)"))
        rank_direction = st.radio("排名方向", ["desc", "asc"], index=0 if preset_data.get("rank_direction", "desc") == "desc" else 1,
                                   format_func=lambda x: "分数越大越好" if x == "desc" else "分数越小越好", key="rank_dir")

        st.divider()
        st.subheader("💰 持仓")
        max_count = st.number_input("最多持有", min_value=1, max_value=20, value=preset_data.get("max_count", 5), key="max_count")
        position_mode = st.radio("模式", ["fixed", "adaptive"], index=0, format_func=lambda x: "固定均分" if x == "fixed" else "动态均分", key="pos_mode")

        st.divider()
        st.subheader("🟢 买入规则")
        buy_all = st.radio("模式", ["all", "any"], index=0, format_func=lambda x: "全部满足才买" if x == "all" else "满足一条就买", key="buy_mode")
        buy_rules = rule_builder("buy", preset_data.get("buy_rules", []), "🟢 买入规则", "green")

        st.divider()
        st.subheader("🔴 卖出规则")
        sell_any = st.radio("模式", ["any", "all"], index=0, format_func=lambda x: "满足一条就卖" if x == "any" else "全部满足才卖", key="sell_mode")
        sell_rules = rule_builder("sell", preset_data.get("sell_rules", []), "🔴 卖出规则", "red")

        st.divider()
        st.subheader("🔄 轮动")
        rebalance_freq = st.selectbox("频率", ["daily", "interval", "weekly", "monthly"],
                                       index=1,
                                       format_func=lambda x: {"daily": "每天", "interval": "按间隔", "weekly": "每周", "monthly": "每月"}[x], key="rebal_freq")
        rebalance_interval = st.number_input("间隔天数", min_value=1, max_value=30, value=preset_data.get("rebalance_interval", 2), key="rebal_int") if rebalance_freq == "interval" else 1

        st.divider()
        st.subheader("📅 回测")
        start_date = st.date_input("开始", value=datetime.datetime.strptime(preset_data.get("start_date", "2020-01-01"), "%Y-%m-%d"), key="start_d")
        end_date = st.date_input("结束", value=datetime.datetime.now(), key="end_d")
        initial_capital = st.number_input("初始资金", min_value=10000, value=preset_data.get("initial_capital", 100000), step=10000, key="init_cap")
        commission = st.number_input("手续费", min_value=0.0, max_value=0.01, value=0.0001, format="%.4f", key="comm")
        slippage = st.number_input("滑点", min_value=0.0, max_value=0.05, value=0.001, format="%.3f", key="slip")
        benchmark = st.text_input("基准", value=preset_data.get("benchmark", "sh510300"), key="bench")
        alternative_asset = st.text_input("替代资产（闲置资金配置）", value=preset_data.get("alternative_asset", ""), key="alt_asset", help="例如：sh511880（银华日利）")

        st.divider()
        run_btn = st.button("🚀 运行回测", type="primary", use_container_width=True)

    # ---------- 主页面 ----------
    if run_btn:
        if not universe:
            st.error("请先选择标的池！")
            return
        if not rank_formula.strip():
            st.error("请先设置排序公式！")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("📥 正在获取数据...")
            progress_bar.progress(10)

            form_data = {
                "strategy_name": strategy_name, "universe": universe,
                "rank_formula": rank_formula, "rank_direction": rank_direction,
                "max_count": max_count, "position_mode": position_mode,
                "buy_all": buy_all == "all", "buy_rules": buy_rules,
                "sell_any": sell_any == "any", "sell_rules": sell_rules,
                "rebalance_freq": rebalance_freq, "rebalance_interval": rebalance_interval,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_capital": initial_capital, "commission": commission, "slippage": slippage,
                "benchmark": benchmark, "alternative_asset": alternative_asset,
            }
            config = build_config(form_data)

            status_text.text("📊 正在下载行情数据...")
            progress_bar.progress(30)
            codes_json = json.dumps(universe)
            all_data = get_data(codes_json, form_data["start_date"], form_data["end_date"])

            progress_bar.progress(50)
            if not all_data:
                st.error("未能获取任何数据，请检查代码是否正确！")
                return

            status_text.text("🔄 正在运行回测...")
            progress_bar.progress(70)
            engine = BacktestEngine(config)
            engine.load_data(all_data)
            results = engine.run()

            progress_bar.progress(100)
            status_text.text("✅ 回测完成！")
            st.session_state.results = results
            st.session_state.config = config
            st.success("✅ 回测完成！")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"回测失败: {str(e)}")
            st.code(traceback.format_exc())
            return

    if st.session_state.results:
        show_results(st.session_state.results)
    else:
        show_guide()


def show_results(results):
    st.divider()
    st.subheader("📈 回测统计")
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

    st.divider()
    st.subheader("📉 净值曲线")
    if 'daily_values' in results and not results['daily_values'].empty:
        df = results['daily_values']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], subplot_titles=('净值', '回撤', '持仓'))
        fig.add_trace(go.Scatter(x=df.index, y=df['nav'], name='策略', line=dict(color='#2196F3', width=1.5)), row=1, col=1)
        if 'benchmark_nav' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['benchmark_nav'], name='基准', line=dict(color='#FF9800', width=1.5)), row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], name='回撤', fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red', width=0.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['hold_count'], name='持仓数', line=dict(color='green', width=1)), row=3, col=1)
        fig.update_layout(height=650, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📝 交易日志")
    if 'trade_log' in results and results['trade_log']:
        trades = pd.DataFrame(results['trade_log'])
        st.dataframe(trades, use_container_width=True)
        csv = trades.to_csv(index=False).encode('utf-8')
        st.download_button("下载交易日志", csv, "trades.csv", "text/csv")

    st.divider()
    st.subheader("💼 当前持仓")
    if 'positions' in results and results['positions']:
        pos = []
        for code, p in results['positions'].items():
            pos.append({'代码': code, '名称': p.get('name', code), '数量': p.get('shares', 0), '成本': f"{p.get('cost_price', 0):.3f}", '现价': f"{p.get('current_price', 0):.3f}", '市值': f"{p.get('market_value', 0):.2f}", '收益率': f"{p.get('profit_pct', 0)*100:.2f}%"})
        st.dataframe(pd.DataFrame(pos), use_container_width=True)
    else:
        st.info("当前无持仓")


def show_guide():
    st.info("👈 请在左侧配置策略参数，点击 **运行回测** 开始")
    with st.expander("📖 使用指南"):
        st.markdown("""
        **快速上手**：选择预设策略 → 调整参数 → 运行回测 → 查看结果
        
        **系统指标**：MA/EMA/RSI/MACD/ATR/BOLL/KDJ/returns/quality_score/RSRS等
        
        **排序公式**：用公式构建器选择指标组合，或手动输入
        
        **买卖规则**：用规则构建器选择指标+运算符+值，支持AND/OR组合
        
        **标的池**：1912只ETF+LOF，支持搜索、分类筛选、多选
        """)
    with st.expander("⚠️ 注意事项"):
        st.markdown("- 指标需约60日预热\n- T+1模式：收盘信号，次日开盘成交\n- 首次回测需下载数据，可能较慢\n- 部分品种可能获取不到数据")


if __name__ == "__main__":
    main()
