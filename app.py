import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="NCUA Executive Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --- Data Engine (Auto-Ingest) ---
def auto_ingest():
    """Combines FS220 and FOICU into a high-speed Parquet file if missing."""
    if not os.path.exists("ncua_data.parquet"):
        with st.status("🚀 Processing NCUA files...", expanded=True) as status:
            try:
                # 1. Load Files with 'sniffing' for commas or pipes
                df_fs = pd.read_csv("FS220.txt", sep=None, engine='python', encoding="latin-1")
                df_pro = pd.read_csv("FOICU.txt", sep=None, engine='python', encoding="latin-1")
                
                # 2. Clean Headers (Lowercase, strip spaces, remove quotes)
                df_fs.columns = [str(c).lower().strip().replace('"', '') for c in df_fs.columns]
                df_pro.columns = [str(c).lower().strip().replace('"', '') for c in df_pro.columns]
                
                # 3. Merge
                if 'cu_number' in df_fs.columns and 'cu_number' in df_pro.columns:
                    final_df = pd.merge(df_fs, df_pro, on="cu_number", how="inner")
                    
                    # 4. FORCE Numeric Conversion for key accounts
                    # Assets (010) and Net Worth (891)
                    for col in ['acct_010', 'acct_891']:
                        if col in final_df.columns:
                            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                    
                    final_df.to_parquet("ncua_data.parquet")
                    status.update(label="✅ Data Ready!", state="complete", expanded=False)
                else:
                    st.error("Missing 'cu_number' in source files.")
                    st.stop()
            except Exception as e:
                st.error(f"Ingest Error: {e}")
                st.stop()

# --- Load Data ---
auto_ingest()

@st.cache_data
def get_cached_data():
    return pd.read_parquet("ncua_data.parquet")

df = get_cached_data()

# --- Main Dashboard ---
st.title("🏦 NCUA Credit Union Dashboard")

if df is not None:
    # 1. Sidebar Filters
    with st.sidebar:
        st.header("Filters")
        all_states = sorted(df['state'].dropna().unique())
        state_filter = st.multiselect("Select State(s)", all_states, default=["MO"])
    
    filtered_df = df[df['state'].isin(state_filter)] if state_filter else df
    
    # 2. CU Selection
    name_col = 'cu_name' if 'cu_name' in df.columns else 'name'
    cu_list = sorted(filtered_df[name_col].unique())
    
    # Default to BluCurrent if it exists
    default_idx = 0
    for i, name in enumerate(cu_list):
        if "BLUCURRENT" in str(name).upper():
            default_idx = i
            break

    selected_cu = st.selectbox("Select Institution", cu_list, index=default_idx)
    row = filtered_df[filtered_df[name_col] == selected_cu].iloc[0]

    # 3. Safe Metric Extraction (Prevents KeyError)
    # Using .get() means if 'acct_010' is missing, it returns 0 instead of crashing
    assets = row.get('acct_010', 0)
    net_worth = row.get('acct_891', 0)
    
    # Ensure they are treated as numbers
    assets = float(assets) if assets else 0.0
    net_worth = float(net_worth) if net_worth else 0.0
    
    nw_ratio = (net_worth / assets * 100) if assets > 0 else 0

    # 4. Display KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", f"${assets:,.0f}")
    col2.metric("Net Worth", f"${net_worth:,.0f}")
    col3.metric("Net Worth Ratio", f"{nw_ratio:.2f}%")

    # 5. Peer Distribution Chart
    st.divider()
    st.subheader(f"Peer Comparison: Net Worth Ratio ({', '.join(state_filter) if state_filter else 'US'})")
    
    chart_df = filtered_df.copy()
    # Calculate ratio for the whole group for the chart
    chart_df['nw_ratio_calc'] = (pd.to_numeric(chart_df.get('acct_891', 0), errors='coerce') / 
                                 pd.to_numeric(chart_df.get('acct_010', 1), errors='coerce')) * 100
    
    # Clean up the chart data
    chart_df = chart_df[chart_df['nw_ratio_calc'].between(0, 30)].dropna(subset=['nw_ratio_calc'])

    fig = px.histogram(
        chart_df, 
        x="nw_ratio_calc", 
        nbins=40,
        labels={'nw_ratio_calc': 'Net Worth Ratio (%)'},
        template="plotly_dark",
        color_discrete_sequence=['#58a6ff']
    )
    
    fig.add_vline(x=nw_ratio, line_dash="dash", line_color="#ff7b72")
    fig.add_annotation(x=nw_ratio, text=f" {selected_cu}", showarrow=False, xanchor="left", font_color="#ff7b72")
    
    st.plotly_chart(fig, use_container_width=True)

    # 6. Profile Info
    with st.expander("🔍 View Institutional Details"):
        st.write(f"**City:** {row.get('city', 'N/A')}")
        st.write(f"**Charter Number:** {row.get('cu_number', 'N/A')}")
        st.write(f"**Cycle Date:** {row.get('cycle_date', 'N/A')}")
