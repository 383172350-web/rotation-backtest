"""
数据获取模块
支持本地pkl + AKShare + Westock 三源自动降级
优先本地pkl，本地没有则AKShare，AKShare失败则Westock
下载的数据自动保存到本地pkl，下次优先读取
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import os
import subprocess
import datetime

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


def _ensure_save_dir():
    """确保有可用于保存pkl的目录，优先D盘，其次C盘"""
    for d in [r"D:\qmt_data\ETF\1d", r"C:\qmt_data\ETF\1d"]:
        try:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            return d
        except PermissionError:
            continue
    #  fallback 到项目目录
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ETF", "1d")
    os.makedirs(fallback, exist_ok=True)
    return fallback


LOCAL_PKL_DIR = _find_local_pkl_dir()


def _save_to_pkl(code: str, df: pd.DataFrame, save_dir: str = None):
    """将下载的数据保存为本地pkl，格式与现有数据一致
    如果本地已有数据，自动合并新旧数据，避免覆盖导致数据丢失
    code: 如 sh510300
    df: 标准化的DataFrame (含 date, open, high, low, close, volume)
    """
    if df is None or df.empty:
        return
    if save_dir is None:
        save_dir = _ensure_save_dir()
    pure_code, suffix = _extract_code_suffix(code)
    if not pure_code or not suffix:
        return
    
    pkl_name = f"{pure_code}_{suffix}_1d.pkl"
    pkl_path = os.path.join(save_dir, pkl_name)
    
    try:
        # 转换新数据为现有pkl格式：stime(YYYYMMDD字符串) 作为索引，与旧数据格式一致
        new_df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        new_df['stime'] = pd.to_datetime(new_df['date']).dt.strftime('%Y%m%d')
        new_df = new_df.set_index('stime').sort_index()
        new_df = new_df[['open', 'high', 'low', 'close', 'volume']]
        
        # 如果本地已有数据，合并新旧数据（保留最新数据，去重）
        if os.path.exists(pkl_path):
            old_df = pd.read_pickle(pkl_path)
            old_count = len(old_df)
            
            # 合并并去重（按索引stime去重，保留最新数据）
            combined = pd.concat([old_df, new_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            
            new_count = len(combined) - old_count
            if new_count > 0:
                # 显示新旧数据对比
                old_end = str(old_df.index[-1])
                new_end = str(combined.index[-1])
                print(f"[更新] {code}: {old_count}条 -> {len(combined)}条 (+{new_count}条新数据, {old_end}~{new_end})")
            else:
                print(f"[跳过] {code}: 本地数据已是最新 ({old_count}条, 截止{old_df.index[-1]})")
                return
            save_df = combined
        else:
            save_df = new_df
            print(f"[保存] {code} -> {pkl_path} ({len(save_df)}条, {save_df.index[0]}~{save_df.index[-1]})")
        
        save_df.to_pickle(pkl_path)
    except Exception as e:
        print(f"[保存失败] {code}: {e}")



def _is_trading_time():
    """判断当前是否在交易时间（9:30-15:00）"""
    now = datetime.datetime.now()
    t = now.time()
    return datetime.time(9, 30) <= t <= datetime.time(15, 0)


def fetch_kline(code: str, start_date: str, end_date: str,
                period: str = "day", fq: str = "qfq", *,
                auto_save: bool = True) -> pd.DataFrame:
    """
    获取K线数据（自动降级：本地pkl -> AKShare -> Westock）
    下载成功后自动保存到本地pkl，如果本地数据过期也会自动补充
    
    智能策略：
    - 交易时间（9:30-15:00）：始终从在线下载最新数据，盘中实时变化
    - 非交易时间：优先本地pkl，核对收盘价是否一致，不一致则强制更新
    
    Args:
        code: 股票代码，如 sh510300 或 510300.SH
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        period: day/week/month
        fq: 复权方式 qfq/hfq/bfq
        auto_save: 下载成功后是否自动保存到本地pkl
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    today = pd.to_datetime(datetime.datetime.now().strftime('%Y-%m-%d'))
    
    # ========== 交易时间：始终从在线下载最新数据（解决盘中变化问题）==========
    if _is_trading_time():
        try:
            df = _fetch_akshare(code, start_date, end_date, period, fq)
            if not df.empty:
                if auto_save:
                    _save_to_pkl(code, df)
                return df
        except Exception:
            pass
        
        try:
            df = _fetch_westock(code, start_date, end_date, period, fq)
            if not df.empty:
                if auto_save:
                    _save_to_pkl(code, df)
                return df
        except Exception:
            pass
        
        return pd.DataFrame()
    
    # ========== 非交易时间：优先本地pkl，核对收盘价 ==========
    # 1. 尝试本地pkl
    df_local = _fetch_local_pkl(code, start_date, end_date)
    if not df_local.empty:
        last_date = pd.to_datetime(df_local['date'].iloc[-1])
        end_dt = pd.to_datetime(end_date)
        
        # 如果本地数据明显过期（差2天以上），直接下载补充
        if last_date < end_dt - pd.Timedelta(days=2) and last_date < today - pd.Timedelta(days=2):
            print(f"[本地过期] {code}: 本地最新={last_date.date()}, 需要补充到={end_date}")
        else:
            # 本地数据日期够新，需要核对收盘价是否一致
            # 下载最近10天数据做对比（减少下载量）
            verify_start = (today - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
            try:
                df_verify = _fetch_akshare(code, verify_start, end_date, period, fq)
                if not df_verify.empty:
                    # 找到本地和在线的最新共同交易日
                    local_last_date = df_local['date'].iloc[-1]
                    verify_row = df_verify[df_verify['date'] == local_last_date]
                    local_row = df_local[df_local['date'] == local_last_date]
                    
                    if not verify_row.empty and not local_row.empty:
                        local_close = float(local_row['close'].iloc[0])
                        verify_close = float(verify_row['close'].iloc[0])
                        
                        if abs(local_close - verify_close) < 1e-6:
                            # 收盘价一致，返回本地数据
                            return df_local
                        else:
                            # 收盘价不一致，强制更新完整数据
                            print(f"[数据核对] {code}: close不一致 本地={local_close} 在线={verify_close}，强制更新")
                            df_full = _fetch_akshare(code, start_date, end_date, period, fq)
                            if not df_full.empty:
                                if auto_save:
                                    _save_to_pkl(code, df_full)
                                return df_full
            except Exception as e:
                print(f"[核对失败] {code}: {e}")
            
            # 核对失败或不需要核对，返回本地数据
            return df_local
    
    # 2. 本地没有数据或过期，尝试AKShare
    try:
        df = _fetch_akshare(code, start_date, end_date, period, fq)
        if not df.empty:
            if auto_save:
                _save_to_pkl(code, df)
            return df
    except Exception:
        pass
    
    # 3. 尝试Westock
    try:
        df = _fetch_westock(code, start_date, end_date, period, fq)
        if not df.empty:
            if auto_save:
                _save_to_pkl(code, df)
            return df
    except Exception:
        pass
    
    return pd.DataFrame()


def batch_fetch_klines(codes: list, start_date: str, end_date: str,
                        period: str = "day", fq: str = "qfq") -> Dict[str, pd.DataFrame]:
    """批量获取多只股票K线，下载成功后自动保存"""
    result = {}
    for item in codes:
        code = item['code'] if isinstance(item, dict) else item
        name = item.get('name', code) if isinstance(item, dict) else code
        df = fetch_kline(code, start_date, end_date, period, fq, auto_save=True)
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
