import uuid
import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import date
# 柱状面积图
import matplotlib.pyplot as plt
# 散点图：altair_chart
import altair as alt
# 三维柱状图：pydeck_chart
import pydeck as pdk
# 上传文件
from io import StringIO
from PIL import Image

import altair as alt


# 动态添加utils文件夹的路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'pages'))

from page1 import Page1
import get_stock_data
import page2
from homepage import homepage



# Set page layout to wide
st.set_page_config(layout="wide", page_title="Stock quant research", page_icon="📈")

#start_date_key = str(uuid.uuid4())
#start_date = st.sidebar.date_input("Start date", date(2018, 1, 1), key=start_date_key)
#end_date = st.sidebar.date_input("End date", date.today())

# Header
st.markdown("<h1 style='text-align: center;'>Stock quant Research 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b style='color: red'>Stock Quant . </b><b style='color: orange'>Research</b> is a simple web app for stock price prediction and backtesting using the <a href='https://www.backtrader.com/'>Backtrader</a> library.</p>", unsafe_allow_html=True)
st.write("<hr/>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h5 style='text-align: center; font-size: 20px;'><b style='color: red'>Stock Quant . </b><b style='color: orange'>Research</b><br/><b style='color: grey'>股  票  量  化  研  究</b></h5>", unsafe_allow_html=True)
# st.sidebar.subheader("量化股票研究")
st.sidebar.markdown("1.获取基础数据")
st.sidebar.markdown("2.选股策略")
st.sidebar.markdown("3.交易策略")
st.sidebar.markdown("4.交易预测")
st.sidebar.markdown("5.结果分析")
# 自定义导航链接
#st.sidebar.page_link("home.py", label="主页")
#st.sidebar.page_link("pages/page1.py", label="数据分析")
st.sidebar.markdown("<a href='pages/page1.py'>数据分析</a>", unsafe_allow_html=True)

st.markdown("""
<style>
/* 按钮样式 background: linear-gradient(45deg, #6a11cb, #2575fc);*/
.stButton>button {
    background: linear-gradient(45deg, #ebebeb, #c5c5c5);
    color: black;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    width:280px;
}
</style>
""", unsafe_allow_html=True)
 
# st.button("渐变按钮")
# st.text_input("带悬停效果的输入框")

# 在侧边栏中添加一个按钮
if st.sidebar.button("基础数据"):
    Page1.page1()
if st.sidebar.button("量化股票研究"):
    st.write("量化股票研究首页被点击了！")
    homepage()