import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="NCUA Dashboard", page_icon="🏦", layout="wide")

def auto_ingest():
    if not os.path.exists("ncua_data.parquet"):
        with st.status("First-time setup: Processing NCUA files...", expanded=True) as status:
            try:
                # 1. Load Files - Removing 'low_memory' to satisfy the Python engine
                st.write("Reading Financials (FS220.txt)...")
                df_fs = pd.read_csv("FS220.txt", sep=None, engine='python', encoding="latin-1")
                
                st.write("Reading Profiles (FOICU.txt)...")
                df_pro = pd.read_csv("FOICU.txt", sep=None, engine='python', encoding="latin-1")
                
                # 2. Clean Column Names (Removes quotes and spaces)
                df_fs.columns = [str(c).lower().strip().replace('"', '') for c in df_fs.columns]
                df_pro.columns = [str(c).lower().strip().replace('"', '') for c in df_pro.columns]
                
                # 3. Merge
                st.write("Merging files on Charter Number...")
                if 'cu_number' in df_fs.columns and 'cu_number' in df_pro.columns:
                    final_df = pd.merge(df_fs, df_pro, on="cu_number", how="inner")
                    
                    st.write("Saving high-speed data file...")
                    final_df.to_parquet("ncua_data.parquet")
                    status.update(label="✅ Data ready!", state="complete", expanded=False)
                else:
                    st.error(f"Still can't find 'cu_number'. Found: {list(df_fs.columns[:3])}")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.stop()

# --- MAIN APP ---
st.title("🏦 NCUA Credit Union Dashboard")

auto_ingest()

@st.cache_data
def load_data():
    if os.path.exists("ncua_data.parquet"):
        return pd.read_parquet("ncua_data.parquet")
    return None

df = load_data()

if df is not None:
    # Handle naming columns
    name_col = 'cu_name' if 'cu_name' in df.columns else 'name'
    
    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("Filters")
        state_filter = st.multiselect("Filter by State", sorted(df['state'].dropna().unique()), default=["MO"])
    
    # Filter data based on state
    filtered_df = df[df['state'].isin(state_filter)] if state_filter else df
    
    # Selection UI
    cu_list = sorted(filtered_df[name_col].unique())
    
    # Default to BLUCURRENT if it exists in the list
    default_index = 0
    for i, name in enumerate(cu_list):
        if "BLUCURRENT" in str(name).upper():
            default_index = i
            break
            
    selected_cu = st.selectbox("Select a Credit Union", cu_list, index=default_index)
    
    # --- UI Layout ---
    row = filtered_df[filtered_df[name_col] == selected_cu].iloc[0]
    
    # Convert acct codes to numbers (Assets=010, Net Worth=891)
    assets = pd.to_numeric(row.get('acct_010', 0), errors='coerce')
    net_worth = pd.to_numeric(row.get('acct_891', 0), errors='coerce')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Assets", f"${assets:,.0f}")
    with col2:
        st.metric("Net Worth", f"${net_worth:,.0f}")
    with col3:
        nw_ratio = (net_worth/assets)*100 if assets > 0 else 0
        st.metric("Net Worth Ratio", f"{nw_ratio:.2f}%")

    st.markdown("---")
    st.subheader(f"Institutional Profile: {selected_cu}")
    
    p_col1, p_col2 = st.columns(2)
    p_col1.write(f"**City/State:** {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
    p_col1.write(f"**Charter Number:** {row.get('cu_number', 'N/A')}")
    
    p_col2.write(f"**Cycle Date:** {row.get('cycle_date', 'N/A')}")
    p_col2.write(f"**Join Number:** {row.get('join_number', 'N/A')}")
