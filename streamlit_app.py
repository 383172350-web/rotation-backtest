# -*- coding: utf-8 -*-
"""
轮动策略回测系统 —— 可视化网页版
基于 rotation-backtest 技能封装
支持：1912只ETF+LOF标的池可视化选择、排序公式构建器、买卖规则构建器
数据源：本地pkl优先，无本地则AKShare/Westock自动降级
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import json
import re
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import BacktestEngine
from data_fetcher import fetch_kline, batch_fetch_klines, LOCAL_PKL_DIR

st.set_page_config(layout="wide", page_title="轮动策略回测系统", page_icon="📊", initial_sidebar_state="expanded")

# ============================================================
#  CSS
# ============================================================
st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
.sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
.stButton>button { border-radius: 6px; font-weight: 600; }
.rule-box { background: #f8f9fa; padding: 8px 12px; border-radius: 6px; margin: 4px 0; border-left: 3px solid #4CAF50; font-size: 0.85rem; }
.rule-box-sell { border-left-color: #F44336; }
.metric-card { padding: 1rem; border-radius: 8px; text-align: center; color: white; }
.metric-value { font-size: 1.6rem; font-weight: bold; }
.metric-label { font-size: 0.8rem; }
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
    "close": "收盘价", "open": "开盘价", "high": "最高价", "low": "最低价",
    "volume": "成交量", "amount": "成交额",
}

SPECIAL_VARS = {
    "rank": "当前排名", "profit": "持仓收益率", "hold_days": "持仓天数", "buy_price": "买入价格",
}

OPS = [">", "<", ">=", "<=", "==", "!="]

# ============================================================
#  预设策略
# ============================================================
PRESETS = {
    "🎯 自定义策略": {},
    "📈 全品类DIFv轮动": {
        "selected_codes": ["sh512100", "sh513100", "sh513500", "sh518880", "sz159985",
                         "sz159981", "sz159980", "sh513030", "sh513520", "sh510300",
                         "sz159949", "sh513050", "sh501018"],
        "alternative_asset": "sh511880",
        "rank_formula": "(MACD_DIF(12,26,9) / ATR(26)) * 100",
        "rank_direction": "desc",
        "max_count": 5, "position_mode": "fixed",
        "buy_match_mode": "all",
        "buy_rules": [
            "close > MA(5)", "close > MA(20)", "MA(10) > MA(20)", "MA(5) > MA(10)",
            "(MACD_DIF(12,26,9) / ATR(26)) * 100 < 120", "rank < 7"
        ],
        "sell_match_mode": "any",
        "sell_rules": ["rank > 6", "returns(1) < -0.03", "returns(20) > 0.25"],
        "rebalance_freq": "interval", "rebalance_interval": 2,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "📊 五斗米动量轮动": {
        "selected_codes": ["sh510050", "sh510300", "sh588000", "sz159915", "sz159531"],
        "rank_formula": "returns(20)", "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_rules": ["close > BOLL_upper(17,2)"],
        "sell_rules": ["close < BOLL_lower(17,2)"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "🏆 精选LOF轮动": {
        "selected_codes": ["sz163402", "sz163417", "sz161903", "sz162703", "sz161005"],
        "rank_formula": "returns(20) + quality_score(20)", "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_rules": ["returns(20) > 0.05"],
        "sell_rules": ["rank > 1"],
        "rebalance_freq": "interval", "rebalance_interval": 20,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "🔮 动量+RSRS轮动": {
        "selected_codes": ["sh518880", "sh513100", "sh588220", "sz159915", "sh511090"],
        "rank_formula": "RSRS_zscore(18)", "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_rules": ["RSRS_zscore(18) > 0.7"],
        "sell_rules": ["RSRS_zscore(18) < -0.7"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2020-01-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
}

# ============================================================
#  缓存数据获取
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(codes_list, start_date, end_date, alt_code=""):
    """获取数据：本地pkl优先，无本地则在线获取"""
    all_codes = list(codes_list)
    if alt_code and alt_code.strip():
        all_codes.append({"code": alt_code.strip(), "name": "替代资产"})
    
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    warmup = (start_dt - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    
    all_data = {}
    for item in all_codes:
        code = item['code'] if isinstance(item, dict) else item
        try:
            df = fetch_kline(code, warmup, end_date)
            if not df.empty and len(df) > 60:
                df['date'] = pd.to_datetime(df['date'])
                all_data[code] = df
            elif code != alt_code:
                st.warning(f"{code} 数据不足或为空，已跳过")
        except Exception as e:
            if code != alt_code:
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
        "buy_match_mode": form_data.get("buy_match_mode", "all"),
        "buy_rules": [],
        "sell_match_mode": form_data.get("sell_match_mode", "any"),
        "sell_rules": [],
        "rebalance": {
            "frequency": form_data["rebalance_freq"],
            "interval": form_data.get("rebalance_interval", 2)
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
        condition = rule.get('condition', '') if isinstance(rule, dict) else str(rule).strip()
        description = rule.get('description', '') if isinstance(rule, dict) else condition
        if condition.strip():
            strategy["buy_rules"].append({"condition": condition, "description": description or f"买入{i+1}"})
    for i, rule in enumerate(form_data.get("sell_rules", [])):
        condition = rule.get('condition', '') if isinstance(rule, dict) else str(rule).strip()
        description = rule.get('description', '') if isinstance(rule, dict) else condition
        if condition.strip():
            strategy["sell_rules"].append({"condition": condition, "description": description or f"卖出{i+1}"})
    if form_data.get("alternative_asset"):
        strategy["alternative_asset"] = {"code": form_data["alternative_asset"], "name": "替代资产"}
    return {"strategy": strategy}


# ============================================================
#  指标参数渲染器
# ============================================================
def render_indicator_params(key_prefix, selected_indicator):
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
    param_str = ",".join(str(params[p["name"]]) for p in info["params"])
    func_name = selected_indicator.split("(")[0]
    return f"{func_name}({param_str})"


# ============================================================
#  万能公式编辑器（可靠版：下拉+按钮混合）
# ============================================================

ALL_OPS = ["+", "-", "*", "/", ">", "<", ">=", "<=", "==", "!=", "AND", "OR"]


def _append_to_formula(formula_key, text):
    """可靠地追加文本到公式，处理空格"""
    current = st.session_state.get(formula_key, "")
    if current and not current.endswith((" ", "(", "+", "-", "*", "/", ">", "<", "=")):
        current += " "
    st.session_state[formula_key] = current + text


def formula_editor(key_prefix, preset_formula=""):
    """万能公式编辑器：下拉选择 + 插入按钮，最可靠
    核心策略：按钮回调设置pending标志，rerun后在text_area渲染前更新session_state缓存
    """
    formula_key = f"{key_prefix}_formula_text"
    editor_key = f"{key_prefix}_editor"
    pending_key = f"{key_prefix}_pending"
    
    # 初始化
    if formula_key not in st.session_state:
        st.session_state[formula_key] = preset_formula
    
    # ===== 关键：在widget渲染前处理待更新 =====
    # 按钮回调设置了pending，rerun后在这里先更新editor缓存
    if pending_key in st.session_state:
        new_val = st.session_state[pending_key]
        st.session_state[formula_key] = new_val
        # 直接设置widget缓存值（必须在widget渲染前）
        st.session_state[editor_key] = new_val
        del st.session_state[pending_key]
    
    # 显示公式编辑区
    st.markdown("**当前公式**")
    current = st.text_area(
        "编辑公式",
        value=st.session_state[formula_key],
        key=editor_key,
        placeholder="点击下方选择元素插入...",
        height=64,
        label_visibility="collapsed"
    )
    # 同步用户手动编辑
    st.session_state[formula_key] = current
    
    # 选择元素插入
    st.markdown("**选择元素插入到公式**")
    
    # 分类选择
    categories = {
        "运算符": ["+", "-", "*", "/", ">", "<", ">=", "<=", "==", "!=", "AND", "OR", "(", ")"],
        "基础字段": ["close", "open", "high", "low", "volume", "amount"],
        "特殊变量": ["rank", "profit", "hold_days", "buy_price"],
    }
    
    cat = st.selectbox("选择分类", list(categories.keys()) + ["系统指标"],
                       key=f"{key_prefix}_cat_select")
    
    if cat in categories:
        element = st.selectbox("选择元素", categories[cat],
                               format_func=lambda x: f"{x}  ({_get_element_desc(x)})",
                               key=f"{key_prefix}_element_select")
        param_expr = element
    else:
        ind_options = list(INDICATORS.keys())
        selected_ind = st.selectbox("选择指标", ind_options,
                                    format_func=lambda x: f"{INDICATORS[x]['name']} ({x.split('(')[0]})",
                                    key=f"{key_prefix}_ind_select")
        
        info = INDICATORS[selected_ind]
        params = {}
        cols = st.columns(len(info["params"]))
        for i, p in enumerate(info["params"]):
            with cols[i]:
                params[p["name"]] = st.number_input(
                    p["label"], min_value=p["min"], max_value=p["max"], value=p["default"],
                    key=f"{key_prefix}_param_{p['name']}"
                )
        param_str = ",".join(str(params[p["name"]]) for p in info["params"])
        func_name = selected_ind.split("(")[0]
        param_expr = f"{func_name}({param_str})"
    
    # 插入按钮
    if st.button(f"➕ 插入 '{param_expr}'", key=f"{key_prefix}_insert", type="primary", use_container_width=True):
        cur = st.session_state[formula_key]
        if cur and not cur.endswith((" ", "(", "+", "-", "*", "/", ">", "<", "=")):
            cur += " "
        new_formula = cur + param_expr + " "
        # 设置pending标志，rerun后在widget渲染前更新
        st.session_state[pending_key] = new_formula
        st.rerun()
    
    return st.session_state[formula_key]


def _get_element_desc(x):
    """获取元素描述"""
    if x in BASIC_FIELDS:
        return BASIC_FIELDS[x]
    if x in SPECIAL_VARS:
        return SPECIAL_VARS[x]
    return x



# ============================================================
#  规则构建器（买入/卖出规则）
# ============================================================
def rule_builder(key_prefix, existing_rules, title, color="green"):
    """规则构建器：万能公式编辑器 + 添加按钮"""
    
    # 向后兼容
    rules = []
    for r in existing_rules:
        if isinstance(r, str):
            rules.append({"condition": r, "description": r})
        elif isinstance(r, dict):
            rules.append(r)
    
    rules_key = f"{key_prefix}_rules_list"
    if rules_key not in st.session_state:
        st.session_state[rules_key] = rules
    
    current_rules = st.session_state[rules_key]
    
    # 显示已有规则
    for i, rule in enumerate(current_rules):
        c1, c2 = st.columns([8, 1])
        with c1:
            display = rule.get('description', rule.get('condition', ''))
            border_color = "#4CAF50" if color == "green" else "#F44336"
            st.markdown(f'<div class="rule-box" style="border-left-color: {border_color};"><b>{i+1}.</b> {display}</div>', unsafe_allow_html=True)
        with c2:
            if st.button("🗑️", key=f"{key_prefix}_del_{i}"):
                current_rules.pop(i)
                st.session_state[rules_key] = current_rules
                st.rerun()
    
    if not current_rules:
        st.caption("暂无规则")
    
    # 公式编辑器
    st.markdown("---")
    st.markdown(f"**➕ 添加新{'买入' if color=='green' else '卖出'}规则**")
    formula = formula_editor(key_prefix, "")
    
    if st.button(f"✅ 添加为{'买入' if color=='green' else '卖出'}规则", key=f"{key_prefix}_add_rule", type="primary", use_container_width=True):
        if formula.strip():
            current_rules.append({"condition": formula.strip(), "description": formula.strip()})
            st.session_state[rules_key] = current_rules
            st.session_state[f"{key_prefix}_formula_text"] = ""
            st.rerun()
        else:
            st.error("公式不能为空")
    
    return current_rules


# ============================================================
#  排序公式构建器
# ============================================================
def rank_formula_builder(key_prefix, current_formula):
    st.markdown("**📊 排序公式**")
    formula = formula_editor(key_prefix, current_formula)
    return formula


# ============================================================
#  标的池选择器
# ============================================================
def stock_pool_selector(key_prefix, selected_codes):
    st.markdown("**📋 标的池选择器**")
    df = POOL_DF.copy()
    
    search = st.text_input("🔍 搜索（名称/代码）", "", key=f"{key_prefix}_search")
    if search:
        mask = df['名称'].str.contains(search, case=False, na=False) | df['代码'].str.contains(search, case=False, na=False)
        df = df[mask]
    
    categories = sorted(df['分类'].unique().tolist())
    selected_cats = st.multiselect("分类筛选", categories, default=[], key=f"{key_prefix}_cats")
    if selected_cats:
        df = df[df['分类'].isin(selected_cats)]
    
    types = sorted(df['类型'].unique().tolist())
    selected_types = st.multiselect("类型筛选", types, default=types, key=f"{key_prefix}_types")
    if selected_types:
        df = df[df['类型'].isin(selected_types)]
    
    st.caption(f"共 {len(df)} 只标的")
    
    df['选中'] = df['代码'].isin(selected_codes)
    edited_df = st.data_editor(
        df[['选中', '代码', '名称', '分类', '类型']],
        column_config={"选中": st.column_config.CheckboxColumn("选中", default=False)},
        hide_index=True, use_container_width=True, height=300,
        key=f"{key_prefix}_editor"
    )
    
    new_selected = edited_df[edited_df['选中'] == True]['代码'].tolist()
    
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
    
    # 数据源状态提示
    if LOCAL_PKL_DIR:
        st.success(f"✅ 本地数据已连接：{LOCAL_PKL_DIR}")
    else:
        st.info("ℹ️ 未检测到本地pkl数据，将使用在线数据源（AKShare/Westock）")
    
    if 'results' not in st.session_state: st.session_state.results = None
    if 'config' not in st.session_state: st.session_state.config = None

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ 策略配置")

        preset = st.selectbox("选择预设策略", list(PRESETS.keys()), key="preset_select")
        preset_data = PRESETS[preset] if preset != "🎯 自定义策略" else {}

        # 预设切换时清理session_state
        if "last_preset" not in st.session_state:
            st.session_state.last_preset = preset
        if preset != st.session_state.last_preset:
            st.session_state.last_preset = preset
            for k in list(st.session_state.keys()):
                if k.startswith(("buy_", "sell_", "rank_", "pool_")):
                    del st.session_state[k]

        strategy_name = st.text_input("策略名称", value=preset_data.get("strategy_name", "我的轮动策略"), key="strategy_name")

        st.divider()
        st.subheader("📋 股票池")
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
        buy_rules = rule_builder("buy", preset_data.get("buy_rules", []), "🟢 买入规则", "green")

        st.divider()
        st.subheader("🔴 卖出规则")
        sell_rules = rule_builder("sell", preset_data.get("sell_rules", []), "🔴 卖出规则", "red")

        st.divider()
        st.subheader("🔄 轮动")
        rebalance_interval = st.number_input("轮动周期（每N个交易日）", min_value=1, max_value=60, value=preset_data.get("rebalance_interval", 2), key="rebal_int")
        rebalance_freq = "interval"

        st.divider()
        st.subheader("📅 回测")
        start_date = st.date_input("开始", value=datetime.datetime.strptime(preset_data.get("start_date", "2020-01-01"), "%Y-%m-%d"), key="start_d")
        end_date = st.date_input("结束", value=datetime.datetime.now(), key="end_d")
        initial_capital = st.number_input("初始资金", min_value=10000, value=preset_data.get("initial_capital", 100000), step=10000, key="init_cap")
        commission = st.number_input("手续费", min_value=0.0, max_value=0.01, value=0.0001, format="%.4f", key="comm")
        slippage = st.number_input("滑点", min_value=0.0, max_value=0.05, value=0.001, format="%.3f", key="slip")
        benchmark = st.text_input("基准", value=preset_data.get("benchmark", "sh510300"), key="bench")
        
        st.markdown("**替代资产（闲置资金配置）**")
        alternative_asset = st.text_input("代码", value=preset_data.get("alternative_asset", "sh511880"), key="alt_asset",
                                          help="例如：sh511880（银华日利）")

    # ---------- 主页面 ----------
    col_btn1, col_btn2 = st.columns([6, 1])
    with col_btn2:
        run_btn = st.button("🚀 运行回测", type="primary", use_container_width=True)

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
                "buy_match_mode": "all", "buy_rules": buy_rules,
                "sell_match_mode": "any", "sell_rules": sell_rules,
                "rebalance_freq": rebalance_freq, "rebalance_interval": rebalance_interval,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_capital": initial_capital, "commission": commission, "slippage": slippage,
                "benchmark": benchmark, "alternative_asset": alternative_asset,
            }
            config = build_config(form_data)
            st.session_state.config = config

            status_text.text("📊 正在下载行情数据...")
            progress_bar.progress(30)
            all_data = get_data(universe, form_data["start_date"], form_data["end_date"], alternative_asset)

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


def _parse_pct(val):
    """解析百分比字符串或数值"""
    if isinstance(val, str):
        val = val.replace('%', '').strip()
        try:
            return float(val) / 100
        except ValueError:
            return 0
    return float(val) if val is not None else 0


def show_results(results):
    st.divider()
    st.subheader("📈 回测统计")
    
    # 解析结果（支持字符串百分比和数值）
    total_return = _parse_pct(results.get('total_return', 0))
    annual_return = _parse_pct(results.get('annual_return', 0))
    max_drawdown = _parse_pct(results.get('max_drawdown', 0))
    sharpe_ratio = results.get('sharpe_ratio', 0)
    win_rate = _parse_pct(results.get('win_rate', 0))
    total_trades = results.get('total_trades', 0)
    
    cols = st.columns(6)
    metrics = [
        ("总收益率", f"{total_return*100:.2f}%", "#4CAF50" if total_return > 0 else "#F44336"),
        ("年化收益", f"{annual_return*100:.2f}%", "#4CAF50" if annual_return > 0 else "#F44336"),
        ("最大回撤", f"{max_drawdown*100:.2f}%", "#FF9800"),
        ("夏普比率", f"{sharpe_ratio:.2f}", "#2196F3"),
        ("胜率", f"{win_rate*100:.1f}%", "#2196F3"),
        ("交易次数", str(total_trades), "#9C27B0"),
    ]
    for i, (label, value, color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f'<div class="metric-card" style="background:{color};"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    # ========== 净值曲线 ==========
    st.divider()
    st.subheader("📉 净值曲线")
    if 'daily_values' in results and results['daily_values'] is not None and not results['daily_values'].empty:
        df = results['daily_values']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Scatter(x=df.index, y=df['nav'], name='策略净值', line=dict(color='#2196F3', width=1.5)), row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['drawdown']*100, name='回撤%', fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red', width=0.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['num_positions'], name='持仓数', line=dict(color='green', width=1)), row=3, col=1)
        fig.update_layout(height=650, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无净值数据")

    # ========== 交易日志 ==========
    st.divider()
    st.subheader("📝 交易日志")
    if 'trade_log' in results and results['trade_log'] is not None and not results['trade_log'].empty:
        trades = results['trade_log']
        st.dataframe(trades, use_container_width=True, hide_index=True)
        csv = trades.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ 下载交易日志", csv, "trades.csv", "text/csv")
    else:
        st.info("暂无交易记录")


def show_guide():
    st.info("👈 请在左侧配置策略参数，点击 **运行回测** 开始")
    with st.expander("📖 使用指南"):
        st.markdown("""
        **快速上手**：选择预设策略 → 调整参数 → 运行回测 → 查看结果
        
        **数据源**：本地pkl优先，无本地则AKShare/Westock自动降级
        
        **系统指标**：MA/EMA/RSI/MACD/ATR/BOLL/KDJ/returns/quality_score/RSRS等
        
        **排序公式**：点击下方按钮自由组装，或直接输入
        
        **买卖规则**：点击按钮组装条件表达式，支持任意组合
        
        **标的池**：1912只ETF+LOF，支持搜索、分类筛选、多选
        """)
    with st.expander("⚠️ 注意事项"):
        st.markdown("""
        - 指标需约60日预热，回测开始日期会自动对齐
        - T+1模式：收盘信号，次日开盘成交
        - 首次回测需下载数据，可能较慢
        - 部分品种可能获取不到数据，将自动跳过
        - 回测结果仅供参考，不构成投资建议
        """)


if __name__ == "__main__":
    main()
