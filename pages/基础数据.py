import os
import sys

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt

# 自建的module包所在路径不在PYTHONPATH下,使用sys.append()命令把报警包的所在文件夹路径加入到PYTHONPATH
# sys.path.append(os.path.join(os.path.dirname(__file__), 'program'))
# st.write("os.path.abspath(__file__)"+os.path.abspath(__file__))
# st.write("os.path.abspath(__file__)"+os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/pages")
from pathlib import Path
# st.write(Path(__file__).parent.parent.resolve())
# project_root = Path(__file__).parent.parent.resolve()

from program import StockBackProgram as sbpm

data_root_path = str(Path(__file__).parent.parent.resolve())


def get_file_size(file_path, KB=False, MB=False):
    """获取文件大小"""
    size = os.path.getsize(file_path)
    if KB:
        size = round(size / 1024, 2)
    elif MB:
        size = round(size / 1024 * 1024, 2)
    else:
        size = size

def list_files(root_dir):
    """遍历文件"""
    names = []
    # 如果是文件夹，则遍历
    for f in os.listdir(root_dir):
        # 拼接路径
        file_path = os.path.join(root_dir, f)
        if os.path.isfile(file_path):
            file_name = os.path.split(file_path)[-1]
            names.append(file_name)
            # 如果是一个文件
            # size = get_file_size(file_path, KB=True)
    # st.write("names:", names)
    files = pd.DataFrame(names, columns=['file_name'])
    # st.dataframe(files)
    st.write("显示股票交易日历文件")
    tddf = files[files['file_name'].str.contains('trade_date')]
    # st.dataframe(tddf, use_container_width=True)

    tddf_selt = tddf.copy()
    #tddf[['file_name']]
    tddf_selt["link"] = "http://localhost:8501/data/"+tddf_selt['file_name']
    tddf_selt["selected"] = False
    # ["selected"] = False
    # st.write(tddf_selt)
    edited_tddf = st.data_editor(tddf_selt, use_container_width=True)
    # favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]  .loc[0]
    # favorite_command = edited_df.query("is_widget == True")["command"].loc[0]  ["file_name"]
    etdf = edited_tddf.query("selected == True").copy()
    # print("edited_tddf-------", type(etdf))
    if not etdf.empty:
        for index, row in etdf.iterrows():
            fnm_tddf = row["file_name"]
            # st.dataframe(fnm_tddf)
            st.write(" 显示文件【"+fnm_tddf+"】内容")
            # st.session_state.fnm_tddf = fnm_tddf
            ftddf = pd.read_csv(f"{data_root_path}/data/{fnm_tddf}", encoding="utf-8")
            st.dataframe(ftddf, use_container_width=True)


    st.write("显示股票行业分类文件")
    idstdf = files[files['file_name'].str.contains('industry')]
    st.dataframe(idstdf, use_container_width=True)

    st.write("显示指数交易数据文件")
    idxdf = files[files['file_name'].str.contains('index_kdata')]
    st.dataframe(idxdf, use_container_width=True)

    st.write("显示某一天可交易股票文件")
    dfdaystooks = files[files['file_name'].str.contains('stock_all')]
    st.dataframe(dfdaystooks, use_container_width=True)

    st.write("显示股票个股交易数据文件")
    dfsg = files[files['file_name'].str.contains('day_k_data')]
    st.dataframe(dfsg, use_container_width=True)

    st.write("显示按年合并的股票数据文件，并增加列[行业],[股票名称]")
    dfmeg = files[files['file_name'].str.contains('total_kday_data')]
    st.dataframe(dfmeg, use_container_width=True)

    st.write("显示某分类sector下的股票")
    sector = files[files['file_name'].str.contains('stock_sector')]
    st.dataframe(sector, use_container_width=True)

def quant_stat():
    # 获取沪深300指数日线数据
    df = pd.read_csv("stock_file_hs300.csv", encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # 计算收益率（QuantStats要求收益率序列）
    returns = df["pct_chg"].dropna() / 100  # Tushare返回的是百分比，需转换为小数

# 初始化会话状态
def init_session_state():
    if 'fnm_tddf' not in st.session_state:
        st.session_state.fnm_tddf = None


def display_file_data():
    if st.session_state.fnm_tddf is None:
        st.write("")
        # st.write("st.session_state.fnm_tddf is None")
    else:
        df = pd.read_csv(f"{data_root_path}/data/{st.session_state.fnm_tddf}", encoding="utf-8")
        st.dataframe(df)

def base_data_app():
    # --------------------------
    # 侧边栏导航
    # --------------------------
    st.sidebar.title("📊 股票数据准备")
    st.sidebar.markdown("---")

    year = st.sidebar.selectbox(
        "股票数据年份",
        [2024]
    )
    # st.sidebar.write("You selected:", options)
    if st.sidebar.button("查看数据"):
        list_files(f'{data_root_path}/data')

    if st.sidebar.button("下载数据"):
        st.write("按钮【下载数据】被点击了!", year)
        st.write(f"下载{year}年股票交易日历。")
        trade_yeadr = f"{data_root_path}/data/trade_date_{year}.csv"
        sbpm.download_trade_date(trade_yeadr, f'{year}-01-01', f'{year}-12-31')
        st.write(f"下载{year}年股票交易日历完成。")
        dfty = pd.read_csv(trade_yeadr, encoding="utf-8")
        st.dataframe(dfty)

    display_file_data()

    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

    # st.line_chart(chart_data, use_container_width=True)
    # st.markdown("---")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

    # st.area_chart(chart_data)

    # st.markdown("---")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["a", "b", "c"])

    # st.bar_chart(chart_data)
    # st.markdown("---")
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)

    # st.pyplot(fig)

    st.markdown("---")

    df = pd.DataFrame(
        [
            {"command": "st.selectbox", "rating": 4, "is_widget": True},
            {"command": "st.balloons", "rating": 5, "is_widget": False},
            {"command": "st.time_input", "rating": 3, "is_widget": True},
        ]
    )
    # edited_df = st.data_editor(df, num_rows="dynamic")

    # favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
    # favorite_command = edited_df.query("is_widget == True")["command"].loc[0]
    # #st.dataframe(favorite_command)
    # st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

    # 创建示例数据
    data = pd.DataFrame({
        '姓名': ['张三', '李四', '王五'],
        '年龄': [25, 30, 35],
        '城市': ['北京', '上海', '广州']
    })

    # 显示数据表
    # st.dataframe(data)

    # 或使用表格格式显示
    # st.table(data)

    # 高亮显示特定行 Pandas requires version '3.0.0' or newer of 'jinja2' (version '2.11.1' currently installed).
    # st.dataframe(data.style.highlight_max(axis=0))

    st.sidebar.info(
        "📌 系统说明\n"
        "- 基于Streamlit 1.23.1开发\n"
        "- 集成Backtrader 1.9.78.123回测框架\n"
        "- 支持从baostock获取基础数据"
    )



if __name__ == '__main__':
    init_session_state()
    base_data_app()
    list_files(f'{data_root_path}/data')