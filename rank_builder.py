def custom_rank_indicator_builder(key_prefix):
    """
    自定义排序指标构建器（类似APP中的"修改指标"功能）
    
    支持：
    - 添加多个指标变量（A=MACD_DIF(12,26,9), B=ATR(26)）
    - 输入指标公式：(A/B)*100
    - 指标命名：DIFv
    """
    st.markdown("**📊 自定义排序指标构建器**")
    
    # 初始化session_state
    if f"{key_prefix}_vars" not in st.session_state:
        st.session_state[f"{key_prefix}_vars"] = {}
    if f"{key_prefix}_formula" not in st.session_state:
        st.session_state[f"{key_prefix}_formula"] = "(A/B)*100"
    if f"{key_prefix}_name" not in st.session_state:
        st.session_state[f"{key_prefix}_name"] = "DIFv"
    
    vars_dict = st.session_state[f"{key_prefix}_vars"]
    
    # 显示已有变量
    if vars_dict:
        st.markdown("**指标变量：**")
        for var_name, var_expr in vars_dict.items():
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(f"<code>{var_name}</code> = {var_expr}", unsafe_allow_html=True)
            with c2:
                if st.button("🗑️", key=f"{key_prefix}_del_var_{var_name}"):
                    del vars_dict[var_name]
                    st.session_state[f"{key_prefix}_vars"] = vars_dict
                    st.rerun()
    
    # 添加新变量
    with st.expander("➕ 添加指标变量"):
        # 自动分配变量名：A, B, C...
        var_letters = [chr(ord('A') + i) for i in range(26)]
        used = set(vars_dict.keys())
        next_var = next((l for l in var_letters if l not in used), 'X')
        
        var_name = st.text_input("变量名", value=next_var, key=f"{key_prefix}_var_name", max_chars=1)
        
        ind = st.selectbox("选择指标", list(INDICATORS.keys()), 
                          format_func=lambda x: f"{INDICATORS[x]['name']} ({x})", 
                          key=f"{key_prefix}_var_ind")
        expr = render_indicator_params(f"{key_prefix}_var", ind)
        
        if st.button("添加变量", key=f"{key_prefix}_add_var"):
            if var_name and var_name not in used:
                vars_dict[var_name] = expr
                st.session_state[f"{key_prefix}_vars"] = vars_dict
                st.rerun()
            else:
                st.error(f"变量名 '{var_name}' 已存在或无效")
    
    # 指标公式
    st.markdown("**指标公式**")
    formula = st.text_input("公式（用A/B/C等变量组合）", 
                             value=st.session_state[f"{key_prefix}_formula"],
                             key=f"{key_prefix}_formula_input",
                             help="例如：(A/B)*100")
    st.session_state[f"{key_prefix}_formula"] = formula
    
    # 指标命名
    st.markdown("**指标命名**")
    name = st.text_input("指标名称", 
                        value=st.session_state[f"{key_prefix}_name"],
                        key=f"{key_prefix}_name_input",
                        help="例如：DIFv")
    st.session_state[f"{key_prefix}_name"] = name
    
    # 生成最终排序公式（将变量替换为实际指标表达式）
    final_formula = formula
    for var_name, var_expr in vars_dict.items():
        final_formula = final_formula.replace(var_name, f"({var_expr})")
    
    st.markdown("**生成的排序公式：**")
    st.code(final_formula, language="python")
    
    return final_formula
