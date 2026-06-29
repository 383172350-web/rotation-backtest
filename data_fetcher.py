"""
数据获取模块 - 云端适配版
支持 yfinance（A股/港股/美股）和本地CSV
"""
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta


def fetch_kline(code: str, start_date: str, end_date: str,
                period: str = "day", fq: str = "qfq") -> pd.DataFrame:
    """
    获取K线数据
    优先使用 yfinance，支持 A股/港股/美股/ETF
    
    code格式: sh600519, sz000001, hk00700, usAAPL
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance 未安装，请安装: pip install yfinance")
        return pd.DataFrame()
    
    # 转换代码格式为 yfinance 格式
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
        
        # 标准化列名
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        
        # 处理日期列
        if 'date' not in df.columns:
            for col in df.columns:
                if 'date' in col or 'datetime' in col:
                    df = df.rename(columns={col: 'date'})
                    break
        
        # 确保基础列存在
        col_map = {
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'vol': 'volume', 'turnover': 'amount',
        }
        for old, new in col_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        # 格式化日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # 选择需要的列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # 添加 amount 列
        if 'volume' in df.columns and 'amount' not in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        
        return df.sort_values('date').reset_index(drop=True)
        
    except Exception as e:
        print(f"  yfinance 获取 {code} 失败: {e}")
        return pd.DataFrame()


def _convert_to_yfinance(code: str) -> str:
    """将内部代码格式转换为 yfinance 格式"""
    code = code.strip().lower()
    
    # 上海 A股/ETF
    if code.startswith('sh'):
        pure = code[2:]
        return f"{pure}.SS"
    
    # 深圳 A股/ETF
    if code.startswith('sz'):
        pure = code[2:]
        return f"{pure}.SZ"
    
    # 港股
    if code.startswith('hk'):
        pure = code[2:]
        return f"{pure}.HK"
    
    # 美股
    if code.startswith('us'):
        return code[2:]
    
    # 纯数字，默认上海
    if code.isdigit():
        if code.startswith(('51', '56', '58', '59', '60')):
            return f"{code}.SS"
        return f"{code}.SZ"
    
    # 其他直接返回
    return code


def load_from_csv(file_path: str) -> pd.DataFrame:
    """从CSV文件加载数据"""
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        col_map = {
            'datetime': 'date', 'time': 'date', 'timestamp': 'date',
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'vol': 'volume', 'turnover': 'amount',
        }
        df = df.rename(columns=col_map)
        
        if 'date' not in df.columns:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"  CSV加载失败: {e}")
        return pd.DataFrame()
