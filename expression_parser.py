"""
规则表达式解析引擎
支持自定义条件：close > MA(20), RSI(6) < 30, returns(20) * 0.5 > 0.1 等
支持系统指标：BOLL(n), KDJ_K(n), KDJ_D(n), KDJ_J(n), MACD_DIF(), MACD_DEA(), MACD_HIST()
支持日历指标：year(), month(), day(), weekday(), month_end()
支持字段别名：O/H/L/C（open/high/low/close）、VOL（volume）
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import indicators


class ExpressionParser:
    """
    解析并执行自定义条件表达式

    支持的语法：
    - 基础字段：close, open, high, low, volume, amount
    - 字段别名：O(open), H(high), L(low), C(close), VOL(volume)
    - 技术指标函数（带参数）：
        - MA(n), EMA(n), RSI(n), ATR(n), returns(n)
        - BOLL(n) — 布林带中轨值，n为周期，默认20
        - KDJ_K(n), KDJ_D(n), KDJ_J(n) — KDJ指标，n为计算周期，默认9
    - 技术指标函数（无参数，读取预计算列）：
        - MACD_DIF(), MACD_DEA(), MACD_HIST()
    - 日历指标函数（无参数）：
        - year() — 当前日期年份
        - month() — 当前月份
        - day() — 当前日期（几号）
        - weekday() — 当前是周几（1=周一 ... 7=周日）
        - month_end() — 是否月末（返回布尔Series）
    - 历史引用：close[n] 表示n天前的值
    - 四则运算：+ - * / ( )
    - 比较运算：> < >= <= == !=
    - 逻辑运算：AND OR NOT
    - 特殊变量（通过extra_vars传入）：
        - profit — 当前收益率
        - hold_days — 持仓天数
        - rank — 当前排名
        - buy_price — 买入价格
    """

    # 支持的函数映射（单参数的指标函数）
    # 注意：KDJ_K/D/J、MACD_DIF/DEA/HIST、BOLL_upper/lower 支持多参数，
    #       在 _eval_expression 中有特殊处理，不在此处定义
    FUNCTIONS = {
        'MA': lambda df, n: indicators.MA(df['close'], int(n)),
        'EMA': lambda df, n: indicators.EMA(df['close'], int(n)),
        'RSI': lambda df, n: indicators.RSI(df['close'], int(n)),
        'ATR': lambda df, n: indicators.ATR(df['high'], df['low'], df['close'], int(n)),
        'returns': lambda df, n: indicators.returns(df['close'], int(n)),
        'BIAS': lambda df, n: indicators.BIAS(df['close'], int(n)),
        'quality_score': lambda df, n: indicators.quality_score(df['close'], int(n)),
        'BOLL': lambda df, n: indicators.BOLL(df['close'], int(n))['mid'],
        'volatility': lambda df, n: indicators.volatility(df['close'], int(n)),
        'gain_percentile': lambda df, n: indicators.gain_percentile(df['close'], int(n)),
        'volume_percentile': lambda df, n: indicators.volume_percentile(df['volume'], int(n)),
        'RSRS_slope': lambda df, n: indicators.RSRS_slope(df['high'], df['low'], int(n or 18)),
    }

    # 无参数的日历指标函数
    CALENDAR_FUNCTIONS = {
        'year',
        'month',
        'day',
        'weekday',
        'month_end',
    }

    # 字段别名映射
    FIELD_ALIASES = {
        'O': 'open',
        'H': 'high',
        'L': 'low',
        'C': 'close',
        'VOL': 'volume',
    }

    def __init__(self, df: pd.DataFrame, extra_vars: Optional[Dict[str, float]] = None):
        """
        df: 包含OHLCV及预计算指标的DataFrame
        extra_vars: 额外变量（如profit, hold_days, rank）
        """
        self.df = df
        self.extra_vars = extra_vars or {}
        # 不再复制df，extra_vars在_eval_expression中直接查找

    def _get_dates(self) -> pd.DatetimeIndex:
        """获取日期索引，兼容 index 和列两种形式"""
        if isinstance(self.df.index, pd.DatetimeIndex):
            return self.df.index
        if 'date' in self.df.columns:
            return pd.to_datetime(self.df['date'])
        return pd.to_datetime(self.df.index)

    def _eval_calendar_function(self, func_name: str) -> pd.Series:
        """评估日历指标函数，返回与 self.df 等长的 Series"""
        dates = self._get_dates()
        if func_name == 'year':
            result = dates.year
        elif func_name == 'month':
            result = dates.month
        elif func_name == 'day':
            result = dates.day
        elif func_name == 'weekday':
            wd = dates.weekday + 1
            result = wd
        elif func_name == 'month_end':
            dates_series = pd.Series(dates.values, index=self.df.index)
            next_dates = dates_series.shift(-1)
            result = dates_series.dt.month != next_dates.dt.month
            result = result.fillna(True)
        else:
            raise ValueError(f"未知日历函数: {func_name}")
        return pd.Series(result.values, index=self.df.index, name=func_name)

    def evaluate(self, expression: str) -> pd.Series:
        """
        评估表达式，返回布尔Series（条件判断）或数值Series（排名打分）
        """
        expr = expression.strip()

        # 处理逻辑运算 AND / OR
        if ' AND ' in expr:
            parts = expr.split(' AND ')
            result = self.evaluate(parts[0])
            for p in parts[1:]:
                result = result & self.evaluate(p)
            return result

        if ' OR ' in expr:
            parts = expr.split(' OR ')
            result = self.evaluate(parts[0])
            for p in parts[1:]:
                result = result | self.evaluate(p)
            return result

        if expr.startswith('NOT '):
            return ~self.evaluate(expr[4:])

        # 处理比较运算
        comp_pattern = r'^(.+?)\s*(>=|<=|!=|==|>|<)\s*(.+)$'
        match = re.match(comp_pattern, expr)
        if match:
            left = self._eval_expression(match.group(1).strip())
            op = match.group(2)
            right = self._eval_expression(match.group(3).strip())

            if op == '>':
                return left > right
            elif op == '<':
                return left < right
            elif op == '>=':
                return left >= right
            elif op == '<=':
                return left <= right
            elif op == '==':
                return left == right
            elif op == '!=':
                return left != right

        # 没有比较符，直接返回数值表达式结果（用于排名打分）
        return self._eval_expression(expr)

    def _eval_expression(self, expr: str) -> pd.Series:
        """递归计算数值表达式"""
        expr = expr.strip()

        # 先检查extra_vars（特殊变量）
        if expr in self.extra_vars:
            return pd.Series(self.extra_vars[expr], index=self.df.index)

        # 处理括号
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    break
            else:
                return self._eval_expression(expr[1:-1])

        # 处理加减乘除（按优先级，从右到左扫描）
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            c = expr[i]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
            elif depth == 0 and c in ('+', '-'):
                if i > 0 and expr[i-1] not in ('(', ',', '*', '/', '+', '-'):
                    left = self._eval_expression(expr[:i])
                    right = self._eval_expression(expr[i+1:])
                    if c == '+':
                        return left + right
                    else:
                        return left - right

        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            c = expr[i]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
            elif depth == 0 and c in ('*', '/'):
                left = self._eval_expression(expr[:i])
                right = self._eval_expression(expr[i+1:])
                if c == '*':
                    return left * right
                else:
                    return left / right

        # 处理函数调用：MA(20), RSI(6), returns(20), BOLL(20), KDJ_K(9,3,3), MACD_DIF(12,26,9) 等
        # 支持带数字参数、多参数和无参数的函数调用
        func_pattern = r'^(\w+)\(([^)]*)\)$'
        match = re.match(func_pattern, expr)
        if match:
            func_name = match.group(1)
            param_str = match.group(2).strip()

            # 无参数函数调用
            if param_str == '':
                # 日历指标函数
                if func_name in self.CALENDAR_FUNCTIONS:
                    return self._eval_calendar_function(func_name)
                # 无参数的技术指标函数（如 MACD_DIF()）
                if func_name in self.FUNCTIONS:
                    return self.FUNCTIONS[func_name](self.df, None)
                # 多参数函数无参数调用时用默认值
                if func_name in ('KDJ_K', 'KDJ_D', 'KDJ_J'):
                    kdj = indicators.KDJ(self.df['high'], self.df['low'], self.df['close'], 9, 3, 3)
                    return kdj[func_name.split('_')[1]]
                if func_name in ('MACD_DIF', 'MACD_DEA', 'MACD_HIST'):
                    macd = indicators.MACD(self.df['close'], 12, 26, 9)
                    key = {'MACD_DIF': 'DIF', 'MACD_DEA': 'DEA', 'MACD_HIST': 'MACD'}[func_name]
                    return macd[key]
                if func_name in ('BOLL_upper', 'BOLL_lower'):
                    boll = indicators.BOLL(self.df['close'], 20, 2)
                    key = 'upper' if func_name == 'BOLL_upper' else 'lower'
                    return boll[key]
                # RSRS系列无参数调用时用默认值
                if func_name == 'RSRS_slope':
                    return indicators.RSRS_slope(self.df['high'], self.df['low'], 18)
                if func_name in ('RSRS_zscore', 'RSRS_right_zscore'):
                    slope = indicators.RSRS_slope(self.df['high'], self.df['low'], 18)
                    if func_name == 'RSRS_zscore':
                        return indicators.RSRS_zscore(slope, 600)
                    else:
                        return indicators.RSRS_right_zscore(slope, 600)

            # 解析参数（支持int和float）
            def _parse_param(p):
                p = p.strip()
                try:
                    return int(p)
                except ValueError:
                    try:
                        return float(p)
                    except ValueError:
                        return p
            params = [_parse_param(p) for p in param_str.split(',')]

            # penalty惩罚项：penalty(days, threshold, penalty_value)
            if func_name == 'penalty':
                days = int(params[0]) if len(params) > 0 else 3
                threshold = float(params[1]) if len(params) > 1 else -0.05
                penalty_value = float(params[2]) if len(params) > 2 else -300
                return indicators.penalty_score(self.df['close'], days, threshold, penalty_value)

            # RSRS系列：需要先算slope再算zscore（两步依赖）
            if func_name in ('RSRS_zscore', 'RSRS_right_zscore'):
                slope_period = int(params[0]) if len(params) > 0 else 18
                zscore_period = int(params[1]) if len(params) > 1 else 600
                slope = indicators.RSRS_slope(self.df['high'], self.df['low'], slope_period)
                if func_name == 'RSRS_zscore':
                    return indicators.RSRS_zscore(slope, zscore_period)
                else:
                    return indicators.RSRS_right_zscore(slope, zscore_period)

            # 多参数函数：KDJ_K(9,3,3), KDJ_D(9,3,3), KDJ_J(9,3,3), MACD_DIF(12,26,9), MACD_DEA(12,26,9), MACD_HIST(12,26,9)
            if func_name in ('KDJ_K', 'KDJ_D', 'KDJ_J'):
                n = int(params[0]) if len(params) > 0 else 9
                m1 = int(params[1]) if len(params) > 1 else 3
                m2 = int(params[2]) if len(params) > 2 else 3
                kdj = indicators.KDJ(self.df['high'], self.df['low'], self.df['close'], n, m1, m2)
                return kdj[func_name.split('_')[1]]

            if func_name in ('MACD_DIF', 'MACD_DEA', 'MACD_HIST'):
                fast = int(params[0]) if len(params) > 0 else 12
                slow = int(params[1]) if len(params) > 1 else 26
                signal = int(params[2]) if len(params) > 2 else 9
                macd = indicators.MACD(self.df['close'], fast, slow, signal)
                key = {'MACD_DIF': 'DIF', 'MACD_DEA': 'DEA', 'MACD_HIST': 'MACD'}[func_name]
                return macd[key]

            if func_name in ('BOLL_upper', 'BOLL_lower'):
                n = int(params[0]) if len(params) > 0 else 20
                std_dev = int(params[1]) if len(params) > 1 else 2
                boll = indicators.BOLL(self.df['close'], n, std_dev)
                key = 'upper' if func_name == 'BOLL_upper' else 'lower'
                return boll[key]

            # 单参数函数
            param = int(params[0]) if isinstance(params[0], (int, float)) else params[0]
            if func_name in self.FUNCTIONS:
                return self.FUNCTIONS[func_name](self.df, param)
            else:
                col_name = f"{func_name}_{param}"
                if col_name in self.df.columns:
                    return self.df[col_name]
                raise ValueError(f"未知函数: {func_name}({param})")

        # 处理历史引用：close[20]
        hist_pattern = r'^(\w+)\[(\d+)\]$'
        match = re.match(hist_pattern, expr)
        if match:
            col_name = match.group(1)
            shift = int(match.group(2))
            if col_name in self.df.columns:
                return self.df[col_name].shift(shift)
            raise ValueError(f"未知列: {col_name}")

        # 处理带下划线的列名（如 MACD_DIF, ATR_26, returns_20 等）
        if expr in self.df.columns:
            return self.df[expr]

        # 处理字段别名：O->open, H->high, L->low, C->close, VOL->volume
        if expr in self.FIELD_ALIASES:
            alias = self.FIELD_ALIASES[expr]
            if alias in self.df.columns:
                return self.df[alias]
            raise ValueError(f"别名 '{expr}' 指向的列 '{alias}' 不存在")

        # 处理函数调用带多个参数：MACD_DIF(12,26,9) -> 映射到 MACD_DIF 列
        multi_func_pattern = r'^(\w+)\((\d+(?:,\d+)*)\)$'
        multi_match = re.match(multi_func_pattern, expr)
        if multi_match:
            func_name = multi_match.group(1)
            params_str = multi_match.group(2)
            candidates = [
                func_name,
                f"{func_name}_{params_str.replace(',', '_')}",
            ]
            for candidate in candidates:
                if candidate in self.df.columns:
                    return self.df[candidate]
            raise ValueError(f"未知函数/列: {func_name}({params_str})")

        # 处理数字常量
        try:
            val = float(expr)
            return pd.Series(val, index=self.df.index)
        except ValueError:
            pass

        raise ValueError(f"无法解析表达式: '{expr}'")


def _fast_evaluate_simple(condition: str, extra_vars: dict) -> bool:
    """
    快速求值简单条件：只支持 var op number 形式
    返回 bool 或 None（无法解析时）
    """
    m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|!=|==|>|<)\s*([-\d.]+)$', condition.strip())
    if not m:
        return None
    
    var = m.group(1)
    op = m.group(2)
    val = float(m.group(3))
    
    if var not in extra_vars:
        return None
    
    actual = extra_vars[var]
    if op == '>': return actual > val
    elif op == '<': return actual < val
    elif op == '>=': return actual >= val
    elif op == '<=': return actual <= val
    elif op == '==': return actual == val
    elif op == '!=': return actual != val
    return None


def evaluate_condition(expression: str, df: pd.DataFrame,
                       extra_vars: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    便捷函数：评估条件表达式
    返回布尔Series
    """
    parser = ExpressionParser(df, extra_vars)
    return parser.evaluate(expression)


def evaluate_score(expression: str, df: pd.DataFrame,
                   extra_vars: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    便捷函数：评估打分表达式（无比较符时返回数值）
    返回数值Series
    """
    parser = ExpressionParser(df, extra_vars)
    result = parser.evaluate(expression)
    if result.dtype == bool:
        return result.astype(float)
    return result
