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

# Optional: akshare may fail in some environments
try:
    import akshare as _ak
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import BacktestEngine
from data_fetcher import fetch_kline, batch_fetch_klines, LOCAL_PKL_DIR, _is_trading_time
import qmt_adapter
import qmt_generator

st.set_page_config(layout="wide", page_title="轮动策略回测系统", page_icon="📊", initial_sidebar_state="expanded")

# ============================================================
#  CSS（美化版）
# ============================================================
st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.3rem; }
.sub-header { font-size: 1.0rem; color: #888; margin-bottom: 1rem; }
.stButton>button { border-radius: 6px; font-weight: 600; }
.rule-box { background: #f8f9fa; padding: 6px 10px; border-radius: 4px; margin: 2px 0; border-left: 3px solid #4CAF50; font-size: 0.82rem; }
.rule-box-sell { border-left-color: #F44336; }
.metric-card { padding: 0.8rem 0.5rem; border-radius: 8px; text-align: center; color: white; box-shadow: 0 2px 6px rgba(0,0,0,0.15); transition: transform 0.15s; }
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 1.5rem; font-weight: bold; }
.metric-label { font-size: 0.75rem; opacity: 0.9; }
/* 侧边栏紧凑 */
[data-testid="stSidebar"] .stMarkdown { margin-bottom: 0.2rem !important; }
[data-testid="stSidebar"] .stExpander { margin-bottom: 0.3rem !important; }
[data-testid="stSidebar"] .stNumberInput { margin-bottom: 0.3rem !important; }
[data-testid="stSidebar"] .stRadio { margin-bottom: 0.3rem !important; }
[data-testid="stSidebar"] .stDateInput { margin-bottom: 0.3rem !important; }
[data-testid="stSidebar"] .stTextInput { margin-bottom: 0.3rem !important; }
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

# 大类到小类的映射（双层分类）
CATEGORY_GROUPS = {
    "全部": None,
    "宽基": ["宽基"],
    "行业": ["行业指数", "科技", "医药医疗", "商业百货", "地产", "金融", "传媒"],
    "主题": ["新基建", "新能源"],
    "商品": ["商品/资源"],
    "跨境": ["港股", "跨境", "中概跨境"],
    "LOF": ["LOF"],
}

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
    "RSRS_slope(n)": {"name": "RSRS斜率", "params": [{"name": "n", "label": "斜率周期", "default": 18, "min": 1, "max": 100}]},
    "RSRS_zscore(n,period)": {"name": "RSRS标准分", "params": [
        {"name": "n", "label": "斜率周期", "default": 18, "min": 1, "max": 100},
        {"name": "period", "label": "标准化窗口", "default": 600, "min": 100, "max": 1000}]},
    "RSRS_right_zscore(n,period)": {"name": "RSRS右偏标准分", "params": [
        {"name": "n", "label": "斜率周期", "default": 18, "min": 1, "max": 100},
        {"name": "period", "label": "标准化窗口", "default": 600, "min": 100, "max": 1000}]},
    "penalty(days,threshold,penalty_value)": {"name": "惩罚项（跌幅扣分）", "params": [
        {"name": "days", "label": "检查天数", "default": 3, "min": 1, "max": 30},
        {"name": "threshold", "label": "跌幅阈值", "default": -0.05, "min": -0.5, "max": 0.0, "step": 0.01},
        {"name": "penalty_value", "label": "惩罚分数", "default": -300, "min": -1000, "max": 0, "step": 10}]},
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
    "📊 DIFv动量轮动": {
        "selected_codes": ["sh512890", "sh515050", "sh513310", "sh513100", "sh513500", "sh513520", "sh513030", "sh518880"],
        "rank_formula": "(MACD_DIF(12,26,9) / ATR(26)) * 100",
        "rank_direction": "desc",
        "max_count": 5, "position_mode": "fixed",
        "buy_match_mode": "all",
        "buy_rules": ["close > MA(5)", "(MACD_DIF(12,26,9) / ATR(26)) * 100 > 0", "(MACD_DIF(12,26,9) / ATR(26)) * 100 < 120"],
        "sell_match_mode": "any",
        "sell_rules": ["rank > 5"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2020-03-12", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "🚀 科技成长DIFv轮动": {
        "selected_codes": ["sh588200", "sz159819", "sh515050", "sh515880", "sh516510", "sz159852", "sh512480", "sz159732", "sh588250", "sh516010", "sh562500", "sh512660", "sz159667", "sz159992", "sz159883", "sh516800", "sh588010", "sh515790", "sh515700", "sz159755", "sz159566", "sh515400", "sz159542", "sh563010", "sz159786", "sz159997"],
        "rank_formula": "(MACD_DIF(12,26,9) / ATR(26)) * 100",
        "rank_direction": "desc",
        "max_count": 10, "position_mode": "fixed",
        "buy_match_mode": "all",
        "buy_rules": ["close > MA(5)", "(MACD_DIF(12,26,9) / ATR(26)) * 100 > 0", "(MACD_DIF(12,26,9) / ATR(26)) * 100 < 120"],
        "sell_match_mode": "any",
        "sell_rules": ["rank > 10"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2024-02-08", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "🔮 RSRS动量轮动": {
        "selected_codes": ["sh518880", "sh513100", "sh588220", "sz159915", "sh511090"],
        "rank_formula": "RSRS_right_zscore(18)",
        "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_match_mode": "any",
        "buy_rules": ["RSRS_right_zscore(18) > 0.15", "RSRS_right_zscore(18) > 0.03 AND close > MA(5)", "close > MA(10)"],
        "sell_match_mode": "all",
        "sell_rules": ["profit < -0.03"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2020-03-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "🏆 精选LOF轮动（含惩罚）": {
        "selected_codes": ["sz163402", "sz163417", "sz161903", "sz162703", "sz161005"],
        "rank_formula": "returns(20) + penalty(3, -0.05, -300)",
        "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_match_mode": "all",
        "buy_rules": ["returns(20) > 0.05", "(close - MA(20)) / volatility(20) > 0"],
        "sell_match_mode": "any",
        "sell_rules": ["returns(20) < 0", "rank > 1"],
        "rebalance_freq": "interval", "rebalance_interval": 20,
        "start_date": "2020-03-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
    "📉 五斗米动量轮动": {
        "selected_codes": ["sh510050", "sh510300", "sh588000", "sz159915", "sz159531"],
        "rank_formula": "close / MA(12) * 100 - 100",
        "rank_direction": "desc",
        "max_count": 1, "position_mode": "fixed",
        "buy_match_mode": "all",
        "buy_rules": ["close > BOLL_upper(17,2)"],
        "sell_match_mode": "all",
        "sell_rules": ["close / MA(12) * 100 - 100 < 0"],
        "rebalance_freq": "daily", "rebalance_interval": 1,
        "start_date": "2020-03-01", "initial_capital": 100000,
        "benchmark": "sh510300",
    },
}

# ============================================================
#  缓存数据获取
# ============================================================

def get_data(codes_list, start_date, end_date, alt_code=""):
    """获取数据：本地pkl优先，无本地或过期则在线获取并自动保存
    智能缓存：
    - 交易时间（9:30-15:00）：fetch_kline 强制在线下载，盘中数据实时变化
    - 非交易时间：1小时缓存，减少重复下载
    """
    # 生成缓存键
    cache_key = f"data_cache_{hash(str(codes_list)+start_date+end_date+alt_code)}"
    
    # 非交易时间：尝试使用缓存（1小时有效）
    if not _is_trading_time():
        if cache_key in st.session_state:
            cached = st.session_state[cache_key]
            age = (datetime.datetime.now() - cached['time']).total_seconds()
            if age < 3600:  # 1小时内
                return cached['data']
    
    # 交易时间 或 缓存过期：重新获取数据
    all_codes = list(codes_list)
    if alt_code and alt_code.strip():
        all_codes.append({"code": alt_code.strip(), "name": "替代资产"})
    
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    warmup = (start_dt - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    
    all_data = {}
    for item in all_codes:
        code = item['code'] if isinstance(item, dict) else item
        try:
            df = fetch_kline(code, warmup, end_date, auto_save=True)
            if not df.empty and len(df) > 60:
                df['date'] = pd.to_datetime(df['date'])
                all_data[code] = df
            elif code != alt_code:
                st.warning(f"{code} 数据不足或为空，已跳过")
        except Exception as e:
            if code != alt_code:
                st.warning(f"获取 {code} 失败: {e}")
    
    # 保存到缓存（供非交易时间使用）
    st.session_state[cache_key] = {
        'data': all_data,
        'time': datetime.datetime.now()
    }
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
            step = p.get("step")
            kwargs = {"min_value": p["min"], "max_value": p["max"], "value": p["default"]}
            if step is not None:
                kwargs["step"] = step
            params[p["name"]] = st.number_input(
                p["label"], key=f"{key_prefix}_param_{p['name']}", **kwargs
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
        # 同时设置editor_key，确保widget重新初始化时使用新值（策略切换时）
        st.session_state[editor_key] = preset_formula
    
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
    
    # 分类选择（系统指标最上，运算符最下）
    categories = {
        "基础字段": ["close", "open", "high", "low", "volume", "amount"],
        "特殊变量": ["rank", "profit", "hold_days", "buy_price"],
        "运算符": ["+", "-", "*", "/", ">", "<", ">=", "<=", "==", "!=", "AND", "OR", "(", ")"],
    }
    
    cat = st.selectbox("选择分类", ["系统指标"] + list(categories.keys()),
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
                # key 中嵌入指标名，切换指标时参数完全重置，避免缓存串扰
                step = p.get("step")
                kwargs = {"min_value": p["min"], "max_value": p["max"], "value": p["default"]}
                if step is not None:
                    kwargs["step"] = step
                params[p["name"]] = st.number_input(
                    p["label"], key=f"{key_prefix}_{selected_ind}_param_{p['name']}", **kwargs
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
    """规则构建器：已有规则折叠 + 公式编辑器"""
    
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
    
    # 已有规则（展开）
    with st.expander(f"📋 已有 {len(current_rules)} 条规则", expanded=True):
        for i, rule in enumerate(current_rules):
            c1, c2 = st.columns([8, 1])
            with c1:
                display = rule.get('description', rule.get('condition', ''))
                border_color = "#4CAF50" if color == "green" else "#F44336"
                st.markdown(f'<div class="rule-box" style="border-left-color: {border_color}; font-size:0.8rem;"><b>{i+1}.</b> {display}</div>', unsafe_allow_html=True)
            with c2:
                if st.button("🗑️", key=f"{key_prefix}_del_{i}"):
                    current_rules.pop(i)
                    st.session_state[rules_key] = current_rules
                    st.rerun()
        if not current_rules:
            st.caption("暂无规则")
    
    # 添加新规则（简化）
    st.markdown(f"**➕ 添加新{'买入' if color=='green' else '卖出'}规则**")
    formula = formula_editor(key_prefix, "")
    
    if st.button(f"✅ 添加", key=f"{key_prefix}_add_rule", type="primary", use_container_width=True):
        if formula.strip():
            current_rules.append({"condition": formula.strip(), "description": formula.strip()})
            st.session_state[rules_key] = current_rules
            st.session_state[f"{key_prefix}_formula_text"] = ""
            st.rerun()
        else:
            st.error("公式不能为空")
    
    return current_rules


# ============================================================
#  排序公式构建器（含惩罚项面板）
# ============================================================
def rank_formula_builder(key_prefix, current_formula):
    
    # 惩罚项面板
    penalty_key = f"{key_prefix}_penalty_enabled"
    penalty_days_key = f"{key_prefix}_penalty_days"
    penalty_threshold_key = f"{key_prefix}_penalty_threshold"
    penalty_value_key = f"{key_prefix}_penalty_value"
    
    if penalty_key not in st.session_state:
        st.session_state[penalty_key] = False
    if penalty_days_key not in st.session_state:
        st.session_state[penalty_days_key] = 3
    if penalty_threshold_key not in st.session_state:
        st.session_state[penalty_threshold_key] = -0.05
    if penalty_value_key not in st.session_state:
        st.session_state[penalty_value_key] = -300
    
    # 惩罚项面板（默认收起）
    with st.expander("⚠️ 惩罚项设置（跌幅扣分）", expanded=False):
        enable_penalty = st.checkbox("启用惩罚项", value=st.session_state[penalty_key], key=penalty_key)
        if enable_penalty:
            pcols = st.columns(3)
            with pcols[0]:
                p_days = st.number_input("检查天数", min_value=1, max_value=30, value=st.session_state[penalty_days_key], key=penalty_days_key)
            with pcols[1]:
                p_threshold = st.number_input("跌幅阈值", min_value=-0.5, max_value=0.0, value=st.session_state[penalty_threshold_key], step=0.01, format="%.2f", key=penalty_threshold_key)
            with pcols[2]:
                p_value = st.number_input("惩罚分数", min_value=-1000, max_value=0, value=st.session_state[penalty_value_key], step=10, key=penalty_value_key)
            st.caption(f"效果：最近{p_days}日任意一天跌幅<{p_threshold:.0%}，排序扣{p_value}分")
    
    formula = formula_editor(key_prefix, current_formula)
    
    # 自动追加/移除惩罚项
    if st.session_state.get(penalty_key, False):
        p_days = st.session_state.get(penalty_days_key, 3)
        p_threshold = st.session_state.get(penalty_threshold_key, -0.05)
        p_value = st.session_state.get(penalty_value_key, -300)
        penalty_expr = f"penalty({p_days}, {p_threshold}, {p_value})"
        if penalty_expr not in formula:
            if formula.strip():
                formula = f"{formula.strip()} + {penalty_expr}"
            else:
                formula = penalty_expr
    else:
        # 移除惩罚项
        formula = re.sub(r'\s*\+\s*penalty\([^)]+\)', '', formula).strip()
        formula = re.sub(r'penalty\([^)]+\)\s*\+\s*', '', formula).strip()
    
    return formula

# ============================================================
#  用户自定义策略持久化
# ============================================================
USER_PRESETS_FILE = os.path.join(os.path.dirname(__file__), "user_presets.json")

def load_user_presets():
    """加载用户保存的自定义策略"""
    if os.path.exists(USER_PRESETS_FILE):
        try:
            with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_preset(name, preset_data):
    """保存用户自定义策略到本地文件"""
    presets = load_user_presets()
    presets[name] = preset_data
    with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)
    return True

def delete_user_preset(name):
    """删除用户自定义策略"""
    presets = load_user_presets()
    if name in presets:
        del presets[name]
        with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
        return True
    return False


def get_all_presets():
    """合并系统预设和用户预设"""
    user_presets = load_user_presets()
    all_presets = dict(PRESETS)
    for name, data in user_presets.items():
        all_presets[f"💾 {name}"] = data
    return all_presets


# ============================================================
#  标的池选择器（简洁版：搜索+下拉选择+已选列表+删除）
# ============================================================
def stock_pool_selector(key_prefix, selected_codes):
    """简洁股票池选择器：搜索下拉选择、快捷分类、已选列表可删除"""
    pool_key = f"{key_prefix}_selected_codes"
    if pool_key not in st.session_state:
        st.session_state[pool_key] = list(selected_codes)
    current_codes = st.session_state[pool_key]
    
    # ===== 搜索添加（下拉选择） =====
    with st.expander("🔍 搜索添加", expanded=True):
        search = st.text_input("输入代码/名称关键词", "", key=f"{key_prefix}_search_input")
        
        if search:
            terms = [s.strip() for s in search.replace("，", ",").split(",") if s.strip()]
            matched = []
            for term in terms:
                mask = (POOL_DF['名称'].str.contains(term, case=False, na=False) | 
                        POOL_DF['代码'].str.contains(term, case=False, na=False))
                matched.extend(POOL_DF[mask].to_dict('records'))
            
            # 去重，排除已选
            seen = set(current_codes)
            unique_matched = []
            for row in matched:
                if row['代码'] not in seen:
                    seen.add(row['代码'])
                    unique_matched.append(row)
            
            if unique_matched:
                options = {f"{r['代码']} | {r['名称']} | {r['分类']}": r['代码'] for r in unique_matched[:20]}
                selected_option = st.selectbox(f"匹配到 {len(unique_matched)} 只，选择添加", 
                                                ["-- 请选择 --"] + list(options.keys()),
                                                key=f"{key_prefix}_select_add")
                if selected_option != "-- 请选择 --":
                    code_to_add = options[selected_option]
                    st.session_state[pool_key].append(code_to_add)
                    st.rerun()
            else:
                st.caption("无匹配结果或已全选")
    
    # ===== 快捷分类（直接选择子类 + 标的）=====
    with st.expander("📂 快捷分类添加", expanded=False):
        # 获取所有有未选标的的子类
        all_sub_cats = sorted(set(POOL_DF['分类'].tolist()))
        available_sub_cats = []
        for cat in all_sub_cats:
            count = len(POOL_DF[(POOL_DF['分类'] == cat) & (~POOL_DF['代码'].isin(current_codes))])
            if count > 0:
                available_sub_cats.append(f"{cat} ({count})")
        
        if available_sub_cats:
            selected_sub = st.selectbox("选择分类", available_sub_cats, key=f"{key_prefix}_subcat_select")
            active_subcat = selected_sub.split(" (")[0] if selected_sub else None
            
            # 该子类的标的选择（下拉框滚动选择）
            if active_subcat:
                cat_df = POOL_DF[(POOL_DF['分类'] == active_subcat) & (~POOL_DF['代码'].isin(current_codes))]
                if not cat_df.empty:
                    st.caption(f"**{active_subcat}**：共 {len(cat_df)} 只，下拉选择添加")
                    options = {f"{r['名称']} ({r['代码']})": r['代码'] for _, r in cat_df.iterrows()}
                    selected = st.selectbox("选择标的", ["-- 请选择 --"] + list(options.keys()), key=f"{key_prefix}_item_select")
                    if selected != "-- 请选择 --":
                        st.session_state[pool_key].append(options[selected])
                        st.rerun()
                else:
                    st.caption(f"{active_subcat} 已全部选中")
        else:
            st.caption("无可添加标的")

    # ===== 已选列表（折叠）=====
    with st.expander(f"📋 已选 {len(current_codes)} 只", expanded=False):
        if current_codes:
            selected_df = POOL_DF[POOL_DF['代码'].isin(current_codes)][['代码', '名称', '分类']].copy()
            order_map = {code: i for i, code in enumerate(current_codes)}
            selected_df['order'] = selected_df['代码'].map(order_map)
            selected_df = selected_df.sort_values('order').drop('order', axis=1)
            
            for idx, row in selected_df.iterrows():
                c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
                with c1:
                    st.text(row['代码'])
                with c2:
                    st.text(row['名称'])
                with c3:
                    st.text(row['分类'])
                with c4:
                    if st.button("🗑️", key=f"{key_prefix}_del_{row['代码']}", help="删除"):
                        st.session_state[pool_key].remove(row['代码'])
                        st.rerun()
            
            if st.button("🗑️ 清空全部", key=f"{key_prefix}_clear_all"):
                st.session_state[pool_key] = []
                st.rerun()
        else:
            st.caption("尚未选择任何标的")

    return st.session_state[pool_key]


# ============================================================
#  主函数
# ============================================================
def main():
    st.markdown('<div class="main-header">📊 轮动策略回测系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">1912只ETF+LOF · 可视化策略构建 · 自定义排序指标 · 自定义买卖规则</div>', unsafe_allow_html=True)
    
    # 数据源状态提示（移到侧边栏底部）
    data_source_info = f"✅ 本地数据：{LOCAL_PKL_DIR}" if LOCAL_PKL_DIR else "ℹ️ 在线数据源（AKShare/Westock）"
    
    if 'results' not in st.session_state: st.session_state.results = None
    if 'config' not in st.session_state: st.session_state.config = None

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ 策略配置")

        all_presets = get_all_presets()
        preset = st.selectbox("选择预设策略", list(all_presets.keys()), key="preset_select")
        preset_data = all_presets[preset] if preset != "🎯 自定义策略" else {}
        is_user_preset = preset.startswith("💾 ")

        # 预设切换时清理 session_state
        if "last_preset" not in st.session_state:
            st.session_state.last_preset = preset
        if preset != st.session_state.last_preset:
            st.session_state.last_preset = preset
            for k in list(st.session_state.keys()):
                if k.startswith(("buy_", "sell_", "rank_", "pool_")):
                    del st.session_state[k]

        # ===== 自定义策略保存（移到最上面）=====
        with st.expander("💾 保存/删除自定义策略"):
            save_name = st.text_input("策略名称", value="", key="save_preset_name", placeholder="输入名称后点击保存")
            if st.button("💾 保存当前策略", key="btn_save_preset"):
                if save_name.strip():
                    preset_to_save = {
                        "selected_codes": selected_codes,
                        "rank_formula": rank_formula,
                        "rank_direction": rank_direction,
                        "max_count": max_count,
                        "position_mode": position_mode,
                        "buy_match_mode": "all",
                        "buy_rules": buy_rules,
                        "sell_match_mode": "any",
                        "sell_rules": sell_rules,
                        "rebalance_freq": rebalance_freq,
                        "rebalance_interval": rebalance_interval,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "initial_capital": initial_capital,
                        "benchmark": benchmark,
                        "alternative_asset": alternative_asset,
                    }
                    if save_user_preset(save_name.strip(), preset_to_save):
                        st.success(f"✅ 策略 '{save_name}' 已保存！刷新页面后可在下拉框中选择。")
                    else:
                        st.error("保存失败")
                else:
                    st.warning("请输入策略名称")
            
            # 删除已保存的自定义策略
            if is_user_preset:
                preset_name = preset[2:]  # 去掉 "💾 " 前缀
                if st.button("🗑️ 删除当前策略", key="btn_delete_preset"):
                    if delete_user_preset(preset_name):
                        st.success(f"✅ 策略 '{preset_name}' 已删除！")
                        st.rerun()
                    else:
                        st.error("删除失败")
        
        st.markdown("---")
        st.markdown("**📋 股票池**")
        init_codes = preset_data.get("selected_codes", [])
        selected_codes = stock_pool_selector("pool", init_codes)

        universe = []
        for code in selected_codes:
            row = POOL_DF[POOL_DF['代码'] == code]
            if not row.empty:
                universe.append({"code": code, "name": row.iloc[0]['名称']})

        st.markdown("---")
        st.markdown("**📊 排序公式**")
        rank_formula = rank_formula_builder("rank", preset_data.get("rank_formula", "returns(20)"))
        rank_direction = st.radio("排名方向", ["desc", "asc"], index=0 if preset_data.get("rank_direction", "desc") == "desc" else 1,
                                   format_func=lambda x: "分数越大越好" if x == "desc" else "分数越小越好", key="rank_dir")

        st.markdown("---")
        st.markdown("**💰 持仓与轮动**")
        c1, c2 = st.columns(2)
        with c1:
            max_count = st.number_input("最多持有", min_value=1, max_value=20, value=preset_data.get("max_count", 5), key="max_count")
        with c2:
            position_mode = st.radio("模式", ["fixed", "adaptive"], index=0, format_func=lambda x: "固定均分" if x == "fixed" else "动态", key="pos_mode")
        rebalance_interval = st.number_input("轮动周期（每N个交易日）", min_value=1, max_value=60, value=preset_data.get("rebalance_interval", 2), key="rebal_int")
        rebalance_freq = "interval"

        st.markdown("---")
        st.markdown("**🟢 买入规则**")
        buy_rules = rule_builder("buy", preset_data.get("buy_rules", []), "🟢 买入规则", "green")

        st.markdown("---")
        st.markdown("**🔴 卖出规则**")
        sell_rules = rule_builder("sell", preset_data.get("sell_rules", []), "🔴 卖出规则", "red")

        st.markdown("---")
        with st.expander("📅 回测参数", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                start_date = st.date_input("开始", value=datetime.datetime.strptime(preset_data.get("start_date", "2020-01-01"), "%Y-%m-%d"), key="start_d")
            with c2:
                end_date = st.date_input("结束", value=datetime.datetime.now(), key="end_d")
            initial_capital = st.number_input("初始资金", min_value=10000, value=preset_data.get("initial_capital", 100000), step=10000, key="init_cap")
            benchmark = st.text_input("基准", value=preset_data.get("benchmark", "sh510300"), key="bench")
            alternative_asset = st.text_input("替代资产（闲置资金配置）", value=preset_data.get("alternative_asset", "sh511880"), key="alt_asset",
                                              help="例如：sh511880（银华日利）")

        with st.expander("⚙️ 高级参数"):
            commission = st.number_input("手续费", min_value=0.0, max_value=0.01, value=0.0001, format="%.4f", key="comm")
            slippage = st.number_input("滑点", min_value=0.0, max_value=0.05, value=0.0, format="%.3f", key="slip")
        
        st.markdown("---")
        st.caption(data_source_info)

    # ---------- 主页面 ----------
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
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
                "strategy_name": preset, "universe": universe,
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

    # 净值曲线
    with st.container():
        if 'daily_values' in results and results['daily_values'] is not None and not results['daily_values'].empty:
            df = results['daily_values']
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.55, 0.25, 0.2])
            fig.add_trace(go.Scatter(
                x=df.index, y=df['nav'], name='策略净值',
                line=dict(color='#2196F3', width=1.5),
                hovertemplate='%{x|%Y-%m-%d}<br>策略净值: %{y:.4f}<extra></extra>'
            ), row=1, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['drawdown']*100, name='回撤%',
                fill='tozeroy', fillcolor='rgba(220,20,60,0.6)',
                line=dict(color='crimson', width=1.2),
                hovertemplate='%{x|%Y-%m-%d}<br>回撤: %{y:.2f}%<extra></extra>'
            ), row=2, col=1)
            fig.update_yaxes(zeroline=False, row=2, col=1)
            fig.update_yaxes(zeroline=False, row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['num_positions'], name='持仓数',
                line=dict(color='green', width=1),
                hovertemplate='%{x|%Y-%m-%d}<br>持仓数: %{y}<extra></extra>'
            ), row=3, col=1)
            fig.update_layout(
                height=600, showlegend=True, hovermode='x unified',
                xaxis=dict(
                    tickformat='%Y-%m-%d', type='date', nticks=20, showgrid=True,
                    rangeslider=dict(visible=True, thickness=0.08),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1月', step='month', stepmode='backward'),
                            dict(count=6, label='6月', step='month', stepmode='backward'),
                            dict(count=1, label='1年', step='year', stepmode='backward'),
                            dict(count=2, label='2年', step='year', stepmode='backward'),
                            dict(step='all', label='全部')
                        ])
                    )
                ),
                xaxis2=dict(tickformat='%Y-%m-%d', type='date', nticks=20, showgrid=True),
                xaxis3=dict(tickformat='%Y-%m-%d', type='date', nticks=20, showgrid=True),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无净值数据")

    # 交易日志（折叠）
    with st.expander("📝 交易日志"):
        if 'trade_log' in results and results['trade_log'] is not None and not results['trade_log'].empty:
            trades = results['trade_log']
            st.dataframe(trades, use_container_width=True, hide_index=True)
            csv = trades.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ 下载交易日志", csv, "trades.csv", "text/csv")
        else:
            st.info("暂无交易记录")

    # 调仓计划（折叠）
    with st.expander("📋 调仓计划"):
        if 'rebalance_plans' in results and results['rebalance_plans'] is not None and not results['rebalance_plans'].empty:
            plan_df = results['rebalance_plans'].copy()
            # 按日期倒序，显示最新的调仓
            plan_df['date'] = pd.to_datetime(plan_df['date'])
            plan_df = plan_df.sort_values('date', ascending=False)
            buy_plans = plan_df[plan_df['plan_type'] == 'BUY_PLAN']
            sell_plans = plan_df[plan_df['plan_type'] == 'SELL_PLAN']

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🟢 最新计划买入**")
                if not buy_plans.empty:
                    buy_display = buy_plans[['date', 'code', 'name', 'detail']].head(20)
                    buy_display.columns = ['日期', '代码', '名称', '详情']
                    st.dataframe(buy_display, use_container_width=True, hide_index=True)
                else:
                    st.caption("无买入计划")
            with c2:
                st.markdown("**🔴 最新计划卖出**")
                if not sell_plans.empty:
                    sell_display = sell_plans[['date', 'code', 'name', 'detail', 'profit_pct', 'hold_days']].head(20)
                    sell_display['profit_pct'] = (sell_display['profit_pct'] * 100).round(2).astype(str) + '%'
                    sell_display.columns = ['日期', '代码', '名称', '卖出原因', '收益率', '持仓天数']
                    st.dataframe(sell_display, use_container_width=True, hide_index=True)
                else:
                    st.caption("无卖出计划")

            csv_plan = plan_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ 下载调仓计划", csv_plan, "rebalance_plans.csv", "text/csv")
        else:
            st.info("暂无调仓计划数据")

    # QMT 实盘代码生成
    with st.expander("🖥️ QMT 实盘代码生成"):
        qmt_account = st.text_input("QMT 资金账号", value="520000249836", key="qmt_account")
        qmt_capital = st.number_input("实盘资金（元）", value=100000.0, step=10000.0, key="qmt_capital")
        qmt_script_dir = st.text_input("交易计划 CSV 保存目录", value=r"C:\QMT\trade_plan", key="qmt_script_dir")
        qmt_run_mode = st.selectbox("运行模式", ["live", "paper"], format_func=lambda x: "实盘" if x == "live" else "模拟", key="qmt_run_mode")
        
        strategy_name = st.session_state.get('selected_preset', 'custom_strategy')
        if st.button("🚀 生成 QMT 实盘代码", key="btn_gen_qmt"):
            with st.spinner("正在生成 QMT 实盘代码..."):
                try:
                    qmt_config = qmt_adapter.build_qmt_config(st.session_state.config)
                    import tempfile
                    fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="qmt_strategy_")
                    os.close(fd)
                    qmt_generator.generate_qmt_file(
                        config=qmt_config,
                        output_path=temp_path,
                        strategy_name=strategy_name,
                        account_id=qmt_account,
                        real_capital=qmt_capital,
                        script_dir=qmt_script_dir,
                        run_mode=qmt_run_mode,
                    )
                    with open(temp_path, "r", encoding="utf-8") as f:
                        qmt_code = f.read()
                    os.unlink(temp_path)
                    st.download_button(
                        label="⬇️ 下载 QMT 策略文件 (.py)",
                        data=qmt_code.encode("utf-8"),
                        file_name=f"qmt_strategy_{strategy_name}.py",
                        mime="text/x-python",
                        key="btn_download_qmt",
                    )
                    st.success("QMT 实盘代码生成成功！")
                    with st.expander("📄 预览代码（前100行）"):
                        st.code("\n".join(qmt_code.splitlines()[:100]), language="python")
                except Exception as e:
                    st.error(f"生成 QMT 代码失败: {e}")
                    st.exception(e)


def show_guide():
    st.info("👈 在左侧配置策略参数，点击 **运行回测** 开始")
    with st.expander("📖 使用指南"):
        st.markdown("""
        **快速上手**：选择预设策略 → 调整参数 → 运行回测 → 查看结果
        
        **数据源**：本地pkl优先，无本地则AKShare/Westock自动降级
        
        **T+1模式**：收盘信号，次日开盘成交
        """)
    with st.expander("⚠️ 注意事项"):
        st.markdown("""
        - 指标需约60日预热，回测开始日期会自动对齐
        - 首次回测需下载数据，可能较慢
        - 回测结果仅供参考，不构成投资建议
        """)


if __name__ == "__main__":
    main()
