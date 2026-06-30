# -*- coding: utf-8 -*-
"""
策略表达式预编译模块
把策略公式/条件预编译成可执行的列名读取，避免回测时重复解析字符串
"""
import re
import pandas as pd
import numpy as np
import indicators as ind


def _extract_functions(expr):
    """提取表达式中所有函数调用和基础字段"""
    pattern = r'(\w+)\s*\(([^)]*)\)'
    found = []
    for m in re.finditer(pattern, expr):
        func_name = m.group(1)
        args_str = m.group(2).strip()
        args = [int(a.strip()) if a.strip().isdigit() else a.strip()
                for a in args_str.split(',') if a.strip()]
        found.append((func_name, args))
    return found


def _precompute_expr(df, expr):
    """
    预计算一个表达式（不涉及特殊变量）
    返回 Series，和 df 等长
    """
    # 简单情况：直接列名
    if expr in df.columns:
        return df[expr].copy()
    
    # 基础字段
    if expr in ('close', 'open', 'high', 'low', 'volume', 'amount'):
        return df[expr].copy()
    
    # 数字常量
    try:
        val = float(expr)
        return pd.Series(val, index=df.index)
    except ValueError:
        pass
    
    # 处理括号：先去掉最外层括号
    e = expr.strip()
    if e.startswith('(') and e.endswith(')'):
        depth = 0
        for i, c in enumerate(e):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(e) - 1:
                break
        else:
            return _precompute_expr(df, e[1:-1])
    
    # 处理加减乘除（从右到左扫描）
    depth = 0
    for i in range(len(e) - 1, -1, -1):
        c = e[i]
        if c == ')':
            depth += 1
        elif c == '(':
            depth -= 1
        elif depth == 0 and c in ('+', '-'):
            if i > 0 and e[i-1] not in ('(', ',', '*', '/', '+', '-'):
                left = _precompute_expr(df, e[:i])
                right = _precompute_expr(df, e[i+1:])
                return left + right if c == '+' else left - right
    
    depth = 0
    for i in range(len(e) - 1, -1, -1):
        c = e[i]
        if c == ')':
            depth += 1
        elif c == '(':
            depth -= 1
        elif depth == 0 and c in ('*', '/'):
            left = _precompute_expr(df, e[:i])
            right = _precompute_expr(df, e[i+1:])
            return left * right if c == '*' else left / right
    
    # 处理函数调用
    func_match = re.match(r'^(\w+)\(([^)]*)\)$', e)
    if func_match:
        func_name = func_match.group(1)
        param_str = func_match.group(2).strip()
        def _parse_p(p):
            p = p.strip()
            try:
                return int(p)
            except ValueError:
                try:
                    return float(p)
                except ValueError:
                    return p
        params = [_parse_p(p) for p in param_str.split(',') if p.strip()]
        
        c = df['close']
        h = df['high']
        l = df['low']
        v = df['volume']
        
        if func_name == 'penalty':
            days = int(params[0]) if len(params) > 0 else 3
            threshold = float(params[1]) if len(params) > 1 else -0.05
            penalty_value = float(params[2]) if len(params) > 2 else -300
            return ind.penalty_score(c, days, threshold, penalty_value)
        elif func_name == 'MA':
            return ind.MA(c, params[0])
        elif func_name == 'EMA':
            return ind.EMA(c, params[0])
        elif func_name == 'RSI':
            return ind.RSI(c, params[0])
        elif func_name == 'ATR':
            return ind.ATR(h, l, c, params[0])
        elif func_name == 'returns':
            return ind.returns(c, params[0])
        elif func_name == 'BIAS':
            return ind.BIAS(c, params[0])
        elif func_name == 'volatility':
            return ind.volatility(c, params[0])
        elif func_name == 'quality_score':
            return ind.quality_score(c, params[0])
        elif func_name == 'gain_percentile':
            return ind.gain_percentile(c, params[0])
        elif func_name == 'volume_percentile':
            return ind.volume_percentile(v, params[0])
        elif func_name == 'MACD_DIF':
            fast = params[0] if len(params) > 0 else 12
            slow = params[1] if len(params) > 1 else 26
            signal = params[2] if len(params) > 2 else 9
            ema_f = ind.EMA(c, fast)
            ema_s = ind.EMA(c, slow)
            return ema_f - ema_s
        elif func_name == 'MACD_DEA':
            fast = params[0] if len(params) > 0 else 12
            slow = params[1] if len(params) > 1 else 26
            signal = params[2] if len(params) > 2 else 9
            ema_f = ind.EMA(c, fast)
            ema_s = ind.EMA(c, slow)
            dif = ema_f - ema_s
            return ind.EMA(dif, signal)
        elif func_name == 'MACD_HIST':
            fast = params[0] if len(params) > 0 else 12
            slow = params[1] if len(params) > 1 else 26
            signal = params[2] if len(params) > 2 else 9
            ema_f = ind.EMA(c, fast)
            ema_s = ind.EMA(c, slow)
            dif = ema_f - ema_s
            dea = ind.EMA(dif, signal)
            return (dif - dea) * 2
        elif func_name == 'BOLL':
            n = params[0] if len(params) > 0 else 20
            return ind.MA(c, n)
        elif func_name == 'BOLL_upper':
            n = params[0] if len(params) > 0 else 20
            std = params[1] if len(params) > 1 else 2
            return ind.MA(c, n) + std * c.rolling(window=n).std()
        elif func_name == 'BOLL_lower':
            n = params[0] if len(params) > 0 else 20
            std = params[1] if len(params) > 1 else 2
            return ind.MA(c, n) - std * c.rolling(window=n).std()
        elif func_name == 'KDJ_K':
            n = params[0] if len(params) > 0 else 9
            m1 = params[1] if len(params) > 1 else 3
            m2 = params[2] if len(params) > 2 else 3
            return ind.KDJ(h, l, c, n, m1, m2)['K']
        elif func_name == 'KDJ_D':
            n = params[0] if len(params) > 0 else 9
            m1 = params[1] if len(params) > 1 else 3
            m2 = params[2] if len(params) > 2 else 3
            return ind.KDJ(h, l, c, n, m1, m2)['D']
        elif func_name == 'KDJ_J':
            n = params[0] if len(params) > 0 else 9
            m1 = params[1] if len(params) > 1 else 3
            m2 = params[2] if len(params) > 2 else 3
            return ind.KDJ(h, l, c, n, m1, m2)['J']
        elif func_name == 'RSRS_slope':
            return ind.RSRS_slope(h, l, params[0] if params else 18)
        elif func_name == 'RSRS_zscore':
            slope = ind.RSRS_slope(h, l, 18)
            return ind.RSRS_zscore(slope, params[0] if params else 600)
        elif func_name == 'RSRS_right_zscore':
            slope = ind.RSRS_slope(h, l, 18)
            return ind.RSRS_right_zscore(slope, params[0] if params else 600)
    
    raise ValueError(f"无法预计算表达式: {expr}")


SPECIAL_VARS = {'rank', 'profit', 'hold_days', 'buy_price'}


def has_special_var(expr):
    """检查表达式是否包含特殊变量"""
    if not expr:
        return False
    for sv in SPECIAL_VARS:
        if sv in expr:
            return True
    return False


def precompute_condition(df, expr):
    """
    预计算条件表达式（不涉及特殊变量）
    返回布尔 Series
    """
    # 处理 AND/OR
    if ' AND ' in expr:
        parts = expr.split(' AND ')
        result = precompute_condition(df, parts[0])
        for p in parts[1:]:
            result = result & precompute_condition(df, p)
        return result
    if ' OR ' in expr:
        parts = expr.split(' OR ')
        result = precompute_condition(df, parts[0])
        for p in parts[1:]:
            result = result | precompute_condition(df, p)
        return result
    
    # 比较运算
    comp_pattern = r'^(.*?)\s*(>=|<=|!=|==|>|<)\s*(.*)$'
    m = re.match(comp_pattern, expr.strip())
    if m:
        left = _precompute_expr(df, m.group(1).strip())
        op = m.group(2)
        right = _precompute_expr(df, m.group(3).strip())
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
    
    # 没有比较符，返回数值
    return _precompute_expr(df, expr)


def precompute_strategy(df, strategy):
    """
    预计算策略中所有不涉及特殊变量的表达式
    返回：添加了预计算列的 df
    """
    df = df.copy()
    
    # 1. 排序公式
    rank_formula = strategy.get('rank_formula', 'returns(20)')
    if not has_special_var(rank_formula):
        try:
            df['__rank_score'] = precompute_condition(df, rank_formula)
        except Exception:
            pass
    
    # 2. 买入规则
    buy_rules = strategy.get('buy_rules', [])
    for i, rule in enumerate(buy_rules):
        cond = rule.get('condition', '') if isinstance(rule, dict) else str(rule)
        if not has_special_var(cond):
            try:
                df[f'__buy_{i}'] = precompute_condition(df, cond)
            except Exception:
                pass
    
    # 3. 卖出规则
    sell_rules = strategy.get('sell_rules', [])
    for i, rule in enumerate(sell_rules):
        cond = rule.get('condition', '') if isinstance(rule, dict) else str(rule)
        if not has_special_var(cond):
            try:
                df[f'__sell_{i}'] = precompute_condition(df, cond)
            except Exception:
                pass
    
    return df


def evaluate_fast(df, expr, extra_vars=None, precomputed_prefix=None):
    """
    快速求值：优先读取预计算列，否则回退到动态解析
    """
    extra_vars = extra_vars or {}
    
    # 1. 检查是否有预计算列
    if precomputed_prefix:
        pc_col = precomputed_prefix
        if pc_col in df.columns:
            return df[pc_col].copy()
    
    # 2. 处理特殊变量
    for sv in SPECIAL_VARS:
        if sv in expr and sv not in extra_vars:
            # 特殊变量未提供，无法预计算
            pass
    
    # 3. 回退到动态解析
    from expression_parser import evaluate_condition
    return evaluate_condition(expr, df, extra_vars)
