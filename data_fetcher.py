"""
数据获取模块 - 支持本地pkl + yfinance双源
优先本地读取，缺失时yfinance补充，支持增量更新
"""
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import os

# 本地数据目录
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ETF", "1d")


def fetch_kline(code: str, start_date: str, end_date: str,
                period: str = "day", fq: str = "qfq") -> pd.DataFrame:
    """
    获取K线数据：优先本地pkl，缺失时yfinance补充
    
    code格式: sh600519, sz000001, hk00700, usAAPL
    """
    # 1. 尝试从本地pkl读取
    df_local = _load_local_pkl(code, start_date, end_date)
    if df_local is not None and not df_local.empty:
        return df_local
    
    # 2. 本地没有，尝试yfinance下载
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance 未安装")
        return pd.DataFrame()
    
    ticker = _convert_to_yfinance(code)
    if not ticker:
        return pd.DataFrame()
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_dt.strftime("%Y-%m-%d"))
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        
        # 标准化日期列
        if 'date' not in df.columns:
            for col in df.columns:
                if 'date' in col or 'datetime' in col:
                    df = df.rename(columns={col: 'date'})
                    break
        
        # 标准化列名
        col_map = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'vol': 'volume'}
        for old, new in col_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        if 'volume' in df.columns and 'amount' not in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        
        return df.sort_values('date').reset_index(drop=True)
        
    except Exception as e:
        print(f"  yfinance 获取 {code} 失败: {e}")
        return pd.DataFrame()


def _load_local_pkl(code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """从本地pkl加载数据"""
    # 代码格式转换：sh510300 -> 510300, SH
    pure_code, suffix = _extract_code_suffix(code)
    if not pure_code or not suffix:
        return None
    
    pkl_name = f"{pure_code}_{suffix}_1d.pkl"
    pkl_path = os.path.join(LOCAL_DATA_DIR, pkl_name)
    
    if not os.path.exists(pkl_path):
        return None
    
    try:
        df = pd.read_pickle(pkl_path)
        
        # 本地pkl格式：index=stime(int), columns=open/high/low/close/volume
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # 转换 stime 索引为日期
        if 'stime' in df.columns:
            df['date'] = pd.to_datetime(df['stime'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
        elif 'date' not in df.columns:
            # 尝试其他日期列
            for col in df.columns:
                if 'date' in col or 'time' in col:
                    df = df.rename(columns={col: 'date'})
                    break
        
        # 确保列名标准
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                return None
        
        # 过滤有效数据（排除0值）
        df = df[(df['close'] > 0) & (df['open'] > 0) & (df['volume'] > 0)]
        
        # 按日期范围筛选
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        if df.empty:
            return None
        
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)
        
        # 添加amount列
        df['amount'] = df['volume'] * df['close']
        
        print(f"  ✅ 本地 {code}: {len(df)} 条记录 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
        return df
        
    except Exception as e:
        print(f"  本地pkl读取 {code} 失败: {e}")
        return None


def _extract_code_suffix(code: str) -> tuple:
    """从sh510300提取510300和SH"""
    code = code.strip().lower()
    if code.startswith('sh'):
        return code[2:], 'SH'
    elif code.startswith('sz'):
        return code[2:], 'SZ'
    elif code.startswith('hk'):
        return code[2:], 'HK'
    elif code.startswith('us'):
        return code[2:], 'US'
    # 纯数字，默认上海
    elif code.isdigit():
        if code.startswith(('51', '56', '58', '59', '60')):
            return code, 'SH'
        return code, 'SZ'
    return None, None


def _convert_to_yfinance(code: str) -> str:
    """将内部代码格式转换为 yfinance 格式"""
    code = code.strip().lower()
    
    if code.startswith('sh'):
        return f"{code[2:]}.SS"
    if code.startswith('sz'):
        return f"{code[2:]}.SZ"
    if code.startswith('hk'):
        return f"{code[2:]}.HK"
    if code.startswith('us'):
        return code[2:]
    if code.isdigit():
        if code.startswith(('51', '56', '58', '59', '60')):
            return f"{code}.SS"
        return f"{code}.SZ"
    return code
