import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NCUA Dashboard", layout="wide")

# ── STEP 1: AUTOMATIC INGEST (The "Background Cook") ────────────────────────
def auto_ingest():
    if not os.path.exists("ncua_data.parquet"):
        with st.status("First-time setup: Processing NCUA files...", expanded=True) as status:
            try:
                # 1. Load Files
                st.write("Reading Financials (FS220.txt)...")
                df_fs = pd.read_csv("FS220.txt", sep="|", encoding="latin-1", low_memory=False)
                
                st.write("Reading Profiles (FOICU.txt)...")
                df_pro = pd.read_csv("FOICU.txt", sep="|", encoding="latin-1", low_memory=False)
                
                # 2. Clean Column Names (The Fix)
                # This forces 'CU_NUMBER ' or 'CU_Number' to become 'cu_number'
                df_fs.columns = [str(c).lower().strip() for c in df_fs.columns]
                df_pro.columns = [str(c).lower().strip() for c in df_pro.columns]
                
                # 3. Merge
                st.write("Merging files on Charter Number...")
                # We check if the column exists before merging to avoid the error again
                if 'cu_number' in df_fs.columns and 'cu_number' in df_pro.columns:
                    final_df = pd.merge(df_fs, df_pro, on="cu_number", how="inner")
                    
                    st.write("Saving high-speed data file...")
                    final_df.to_parquet("ncua_data.parquet")
                    status.update(label="✅ Data ready!", state="complete", expanded=False)
                else:
                    st.error(f"Could not find 'cu_number' column. Available: {list(df_fs.columns[:5])}")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.stop()

# ── STEP 2: LOAD & DISPLAY ──────────────────────────────────────────────────
st.title("🏦 NCUA Credit Union Dashboard")

auto_ingest()

@st.cache_data
def load_data():
    return pd.read_parquet("ncua_data.parquet")

df = load_data()

if df is not None:
    # Use 'cu_name' or 'name' depending on what NCUA provides in FOICU
    name_col = 'cu_name' if 'cu_name' in df.columns else 'name'
    
    cu_list = sorted(df[name_col].unique())
    selected_cu = st.selectbox("Select a Credit Union", cu_list)
    
    row = df[df[name_col] == selected_cu].iloc[0]
    
    # Financial metrics (Assets = 010, Net Worth = 891)
    # Using .get() to avoid errors if the specific acct column is missing
    assets = row.get('acct_010', 0)
    net_worth = row.get('acct_891', 0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", f"${assets:,.0f}")
    col2.metric("Net Worth", f"${net_worth:,.0f}")
    
    nw_ratio = (net_worth/assets)*100 if assets > 0 else 0
    col3.metric("Net Worth Ratio", f"{nw_ratio:.2f}%")

    st.markdown("---")
    st.write(f"### Details for {selected_cu}")
    st.write(f"**City:** {row.get('city', 'N/A')}")
    st.write(f"**State:** {row.get('state', 'N/A')}")
