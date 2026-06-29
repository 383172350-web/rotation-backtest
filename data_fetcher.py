"""
数据获取模块 - 本地pkl读取
从本地pkl目录读取ETF日线数据
"""
import pandas as pd
import numpy as np
from typing import Optional
import os


# 本地数据目录（默认项目内，可通过环境变量覆盖）
LOCAL_DATA_DIR = os.environ.get('ETF_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ETF", "1d"))


def fetch_kline(code: str, start_date: str, end_date: str,
                period: str = "day", fq: str = "qfq") -> pd.DataFrame:
    """
    从本地pkl读取K线数据

    code格式: sh510300, sz159915
    """
    pure_code, suffix = _extract_code_suffix(code)
    if not pure_code or not suffix:
        return pd.DataFrame()

    pkl_name = f"{pure_code}_{suffix}_1d.pkl"
    pkl_path = os.path.join(LOCAL_DATA_DIR, pkl_name)

    if not os.path.exists(pkl_path):
        return pd.DataFrame()

    try:
        df = pd.read_pickle(pkl_path)

        # 本地pkl格式：index=stime(int), columns=open/high/low/close/volume
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # 转换 stime 索引为日期
        if 'stime' in df.columns:
            df['date'] = pd.to_datetime(df['stime'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
        elif 'date' not in df.columns:
            for col in df.columns:
                if 'date' in col or 'time' in col:
                    df = df.rename(columns={col: 'date'})
                    break

        # 确保列名标准
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                return pd.DataFrame()

        # 过滤有效数据
        df = df[(df['close'] > 0) & (df['open'] > 0) & (df['volume'] > 0)]

        # 按日期范围筛选
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

        if df.empty:
            return pd.DataFrame()

        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)

        # 添加amount列
        df['amount'] = df['volume'] * df['close']

        return df

    except Exception:
        return pd.DataFrame()


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
    elif code.isdigit():
        if code.startswith(('51', '56', '58', '59', '60')):
            return code, 'SH'
        return code, 'SZ'
    return None, None
