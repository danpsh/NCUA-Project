import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NCUA Dashboard", layout="wide")

# This helper replaces the missing 'loader.py'
@st.cache_data
def load_data():
    if os.path.exists("ncua_data.parquet"):
        return pd.read_parquet("ncua_data.parquet")
    return None

st.title("🏦 NCUA Credit Union Dashboard")

df = load_data()

if df is None:
    st.error("Data not found! Please run 'python ingest.py' first.")
else:
    # 1. Selection
    cu_list = sorted(df['cu_name'].unique())
    selected_cu = st.selectbox("Select a Credit Union", cu_list)
    
    # 2. Filter Data
    row = df[df['cu_name'] == selected_cu].iloc[0]
    
    # 3. Simple Metrics
    assets = row.get('acct_010', 0)
    net_worth = row.get('acct_891', 0)
    
    col1, col2 = st.columns(2)
    col1.metric("Total Assets", f"${assets:,.0f}")
    col2.metric("Net Worth Ratio", f"{(net_worth/assets)*100:.2f}%" if assets > 0 else "0%")

    st.write("### Profile Details")
    st.write(f"**City:** {row.get('city', 'N/A')}")
    st.write(f"**State:** {row.get('state', 'N/A')}")
