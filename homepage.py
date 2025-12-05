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


def homepage():
	'''
	def plot_stock(code,title,start,end):
	    """
	    再定义一个画图函数，对相应股票（指数）在某期间的价格走势和累计收益进行可视化。
	    """
	    st.subheader(f"收盘价走势 - {code}")
	    df = pd.read_csv('./data/k_data_'+code+'.csv', encoding="utf-8")
	    df.index = pd.to_datetime(df.date)
	    print(str(df.index[1])+"  "+str(df.index[-100])+" "+str(df.close.max()))
	    df = df[['open','high','low','close']]
	    st.line_chart(df)

	plot_stock('sh.600000','浦发银行','2024-01-01','2024-12-31')

	num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
	num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

	indices = np.linspace(0, 1, num_points)
	theta = 2 * np.pi * num_turns * indices
	radius = indices

	x = radius * np.cos(theta)
	y = radius * np.sin(theta)

	df = pd.DataFrame({
	    "x": x,
	    "y": y,
	    "idx": indices,
	    "rand": np.random.randn(num_points),
	})

	st.altair_chart(alt.Chart(df, height=700, width=700)
	    .mark_point(filled=True)
	    .encode(
		x=alt.X("x", axis=None),
		y=alt.Y("y", axis=None),
		color=alt.Color("idx", legend=None, scale=alt.Scale()),
		size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
	    ))
	'''