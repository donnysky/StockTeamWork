import streamlit as st

def display_all_elements():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stock_year = st.text_input("股票年份")
        if stock_year:
            st.write("获取股票年份: "+stock_year+"的交易日期。")
    with col2:
        stock_year1 = st.text_input("股票年份1")
        if stock_year1:
            st.write("获取股票年份: " + stock_year1 + "的交易日期。")
    with col3:
        stock_year2 = st.text_input("股票年份2")
        if stock_year2:
            st.write("获取股票年份: " + stock_year2 + "的交易日期。")
    with col4:
        stock_year3 = st.text_input("股票年份3")
        if stock_year3:
            st.write("获取股票年份: " + stock_year3 + "的交易日期。")

st.subheader("基础数据")
st.markdown("<hr/>", unsafe_allow_html=True)
cl1, cl2, cl3 = st.columns(3)
with cl1:
    st.header("第一列")
with cl2:
    st.header("第2列")
with cl3:
    st.header("第3列")
display_all_elements()