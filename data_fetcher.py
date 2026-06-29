"""
数据获取模块
支持本地pkl + AKShare + Westock 三源自动降级
优先本地pkl，本地没有则AKShare，AKShare失败则Westock
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import os
import subprocess

# ========== 本地数据目录配置 ==========
LOCAL_DATA_DIRS = [
    r"D:\qmt_data\ETF\1d",
    r"C:\qmt_data\ETF\1d",
    os.environ.get('ETF_DATA_DIR', ''),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ETF", "1d"),
]


def _find_local_pkl_dir():
    """找到可用的本地pkl目录"""
    for d in LOCAL_DATA_DIRS:
        if d and os.path.exists(d) and os.path.isdir(d):
            pkls = [f for f in os.listdir(d) if f.endswith("_1d.pkl")]
            if len(pkls) > 3:
                return d
    return None


LOCAL_PKL_DIR = _find_local_pkl_dir()


# ========== 主入口：自动降级 ==========
def fetch_kline(code: str, start_date: str, end_date: str,
                period: str = "day", fq: str = "qfq") -> pd.DataFrame:
    """
    获取K线数据（自动降级：本地pkl -> AKShare -> Westock）
    
    Args:
        code: 股票代码，如 sh510300 或 510300.SH
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        period: day/week/month
        fq: 复权方式 qfq/hfq/bfq
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    # 1. 尝试本地pkl
    df = _fetch_local_pkl(code, start_date, end_date)
    if not df.empty:
        return df
    
    # 2. 尝试AKShare
    try:
        df = _fetch_akshare(code, start_date, end_date, period, fq)
        if not df.empty:
            return df
    except Exception:
        pass
    
    # 3. 尝试Westock
    try:
        df = _fetch_westock(code, start_date, end_date, period, fq)
        if not df.empty:
            return df
    except Exception:
        pass
    
    return pd.DataFrame()


def batch_fetch_klines(codes: list, start_date: str, end_date: str,
                        period: str = "day", fq: str = "qfq") -> Dict[str, pd.DataFrame]:
    """批量获取多只股票K线"""
    result = {}
    for item in codes:
        code = item['code'] if isinstance(item, dict) else item
        name = item.get('name', code) if isinstance(item, dict) else code
        df = fetch_kline(code, start_date, end_date, period, fq)
        if not df.empty:
            result[code] = df
    return result


# ========== 本地pkl读取 ==========
def _fetch_local_pkl(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从本地pkl读取K线数据"""
    if not LOCAL_PKL_DIR:
        return pd.DataFrame()
    
    pure_code, suffix = _extract_code_suffix(code)
    if not pure_code or not suffix:
        return pd.DataFrame()
    
    pkl_name = f"{pure_code}_{suffix}_1d.pkl"
    pkl_path = os.path.join(LOCAL_PKL_DIR, pkl_name)
    
    if not os.path.exists(pkl_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_pickle(pkl_path)
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
        df = df[(df['close'] > 0) & (df['open'] > 0)]
        
        # 按日期范围筛选
        df['date'] = pd.to_datetime(df['date'])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)
        df['amount'] = df['volume'] * df['close']
        return df
    
    except Exception:
        return pd.DataFrame()


# ========== AKShare ==========
def _fetch_akshare(code: str, start_date: str, end_date: str,
                   period: str = "day", fq: str = "qfq") -> pd.DataFrame:
    """使用AKShare获取ETF K线数据（东方财富数据源）"""
    try:
        import akshare as ak
    except ImportError:
        return pd.DataFrame()
    
    # 去掉 sh/sz 前缀，AKShare只需要纯数字代码
    pure_code = code[2:] if code[:2].lower() in ('sh', 'sz', 'SH', 'SZ') else code
    
    # 映射复权参数
    adjust_map = {"qfq": "qfq", "hfq": "hfq", "bfq": ""}
    adjust = adjust_map.get(fq, "qfq")
    
    try:
        df = ak.fund_etf_hist_em(
            symbol=pure_code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust=adjust
        )
        if df.empty:
            return pd.DataFrame()
        return _standardize_df(df)
    except Exception:
        return pd.DataFrame()


# ========== Westock ==========
def _fetch_westock(code: str, start_date: str, end_date: str,
                   period: str = "day", fq: str = "qfq") -> pd.DataFrame:
    """使用westockdata获取K线数据"""
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end_dt - start_dt).days
    limit = int(days * 250 / 365) + 100
    
    cmd = f"npx -y westock-data-clawhub@1.0.4 kline {code} --period {period} --limit {limit} --fq {fq}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return pd.DataFrame()
        df = _parse_markdown_table(result.stdout)
        if df.empty:
            return df
        df = _standardize_df(df)
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df = df[mask].reset_index(drop=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        return df.sort_values('date').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ========== 工具函数 ==========
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


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名为统一格式"""
    column_mapping = {
        '日期': 'date', 'date': 'date', 'Date': 'date',
        '开盘': 'open', 'open': 'open', 'Open': 'open', '开盘价': 'open',
        '最高': 'high', 'high': 'high', 'High': 'high', '最高价': 'high',
        '最低': 'low', 'low': 'low', 'Low': 'low', '最低价': 'low',
        '收盘': 'close', 'close': 'close', 'Close': 'close', '收盘价': 'close',
        'last': 'close', 'Last': 'close',
        '成交量': 'volume', 'volume': 'volume', 'Volume': 'volume', 'vol': 'volume',
        '成交额': 'amount', 'amount': 'amount', 'Amount': 'amount',
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df


def _parse_markdown_table(text: str) -> pd.DataFrame:
    """解析Markdown表格为DataFrame"""
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return pd.DataFrame()
    
    table_start = -1
    for i, line in enumerate(lines):
        if '|' in line and (i+1 < len(lines) and '---' in lines[i+1]):
            table_start = i
            break
    
    if table_start == -1:
        if '|' not in lines[0]:
            return pd.DataFrame()
        table_start = 0
    
    header_line = lines[table_start].strip()
    if header_line.startswith('|'):
        header_line = header_line[1:]
    if header_line.endswith('|'):
        header_line = header_line[:-1]
    headers = [h.strip() for h in header_line.split('|')]
    
    data_start = table_start + 2
    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or not '|' in line:
            continue
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        values = [v.strip() for v in line.split('|')]
        if len(values) == len(headers):
            rows.append(dict(zip(headers, values)))
    
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
