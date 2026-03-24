import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NCUA Dashboard", layout="wide")

# ── STEP 1: AUTOMATIC INGEST (The "Background Cook") ────────────────────────
def auto_ingest():
    """Checks if the data file exists; if not, creates it from the .txt files."""
    if not os.path.exists("ncua_data.parquet"):
        with st.status("First-time setup: Processing NCUA files...", expanded=True) as status:
            try:
                # Look for the two main files in the same folder
                st.write("Reading FS220.txt...")
                df_fs = pd.read_csv("FS220.txt", sep="|", encoding="latin-1", low_memory=False)
                
                st.write("Reading FOICU.txt...")
                df_pro = pd.read_csv("FOICU.txt", sep="|", encoding="latin-1", low_memory=False)
                
                # Normalize column names
                df_fs.columns = [c.lower().strip() for c in df_fs.columns]
                df_pro.columns = [c.lower().strip() for c in df_pro.columns]
                
                st.write("Merging data...")
                final_df = pd.merge(df_fs, df_pro, on="cu_number", how="inner")
                
                st.write("Saving high-speed data file...")
                final_df.to_parquet("ncua_data.parquet")
                status.update(label="✅ Data ready!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.stop()

# ── STEP 2: LOAD & DISPLAY ──────────────────────────────────────────────────
st.title("🏦 NCUA Credit Union Dashboard")

# Run the auto-ingest check
auto_ingest()

@st.cache_data
def load_data():
    return pd.read_parquet("ncua_data.parquet")

df = load_data()

if df is not None:
    # Selection UI
    cu_list = sorted(df['cu_name'].unique())
    selected_cu = st.selectbox("Select a Credit Union", cu_list)
    
    # Filter to selected CU
    row = df[df['cu_name'] == selected_cu].iloc[0]
    
    # Display Key Metrics
    assets = row.get('acct_010', 0)
    net_worth = row.get('acct_891', 0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", f"${assets:,.0f}")
    col2.metric("Net Worth", f"${net_worth:,.0f}")
    col3.metric("Net Worth Ratio", f"{(net_worth/assets)*100:.2f}%" if assets > 0 else "0%")

    st.markdown("---")
    st.write(f"### Profile for {selected_cu}")
    st.write(f"**Location:** {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
    st.write(f"**Charter Number:** {row.get('cu_number', 'N/A')}")
