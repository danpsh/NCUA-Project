import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="NCUA Executive Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 15px; margin-bottom: 10px;
    }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; }
    .metric-value { font-size: 24px; font-weight: 700; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# --- Data Engine (Auto-Ingest) ---
def auto_ingest():
    """Combines FS220 and FOICU into a high-speed Parquet file if missing."""
    if not os.path.exists("ncua_data.parquet"):
        with st.status("🚀 First-time setup: Processing NCUA files...", expanded=True) as status:
            try:
                st.write("Loading Financials (FS220.txt)...")
                df_fs = pd.read_csv("FS220.txt", sep=None, engine='python', encoding="latin-1")
                
                st.write("Loading Profiles (FOICU.txt)...")
                df_pro = pd.read_csv("FOICU.txt", sep=None, engine='python', encoding="latin-1")
                
                # Normalize headers
                df_fs.columns = [str(c).lower().strip().replace('"', '') for c in df_fs.columns]
                df_pro.columns = [str(c).lower().strip().replace('"', '') for c in df_pro.columns]
                
                st.write("Merging data on Charter Number...")
                if 'cu_number' in df_fs.columns and 'cu_number' in df_pro.columns:
                    final_df = pd.merge(df_fs, df_pro, on="cu_number", how="inner")
                    
                    # Convert key financial strings to numbers for math
                    for col in ['acct_010', 'acct_891']:
                        if col in final_df.columns:
                            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
                    
                    st.write("Optimizing for speed...")
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
    # 1. Sidebar Controls
    with st.sidebar:
        st.header("Global Filters")
        all_states = sorted(df['state'].dropna().unique())
        state_filter = st.multiselect("Select State(s)", all_states, default=["MO"])
        
        st.divider()
        st.caption("Data Source: NCUA 5300 Call Report")

    # 2. Filter Logic
    filtered_df = df[df['state'].isin(state_filter)] if state_filter else df
    
    # 3. CU Selection
    name_col = 'cu_name' if 'cu_name' in df.columns else 'name'
    cu_list = sorted(filtered_df[name_col].unique())
    
    # Default selection logic for BluCurrent
    default_idx = 0
    for i, name in enumerate(cu_list):
        if "BLUCURRENT" in str(name).upper():
            default_idx = i
            break

    selected_cu = st.selectbox("Select Institution", cu_list, index=default_idx)
    row = filtered_df[filtered_df[name_col] == selected_cu].iloc[0]

    # 4. Key Performance Indicators (KPIs)
    assets = row['acct_010']
    net_worth = row['acct_891']
    nw_ratio = (net_worth / assets * 100) if assets > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Assets</div><div class="metric-value">${assets:,.0f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Net Worth</div><div class="metric-value">${net_worth:,.0f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Net Worth Ratio</div><div class="metric-value">{nw_ratio:.2f}%</div></div>', unsafe_allow_html=True)

    # 5. Peer Distribution Chart
    st.divider()
    st.subheader(f"Peer Comparison: Net Worth Ratio ({', '.join(state_filter) if state_filter else 'US'})")
    
    # Prep chart data
    chart_df = filtered_df.copy()
    chart_df['nw_ratio_calc'] = (chart_df['acct_891'] / chart_df['acct_010']) * 100
    chart_df = chart_df[chart_df['nw_ratio_calc'].between(0, 30)] # Filter outliers for better visual

    fig = px.histogram(
        chart_df, 
        x="nw_ratio_calc", 
        nbins=40,
        labels={'nw_ratio_calc': 'Net Worth Ratio (%)'},
        template="plotly_dark",
        color_discrete_sequence=['#30363d']
    )
    
    # Add indicator for selected CU
    fig.add_vline(x=nw_ratio, line_dash="dash", line_color="#58a6ff")
    fig.add_annotation(x=nw_ratio, text=f" {selected_cu}", showarrow=False, xanchor="left", font_color="#58a6ff")
    
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True)

    # 6. Profile Table
    with st.expander("🔍 View Full Institutional Profile"):
        st.table(pd.DataFrame({
            "Field": ["City", "State", "Charter Number", "Cycle Date", "Join Number"],
            "Value": [row.get('city'), row.get('state'), row.get('cu_number'), row.get('cycle_date'), row.get('join_number')]
        }))
