##### factor_contributions
##### --- Section 1 - URL Parsing and Dataset Loading ---

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from urllib.parse import urlparse, urlencode

# ---- Page config ----
st.set_page_config(page_title="Factor Contributions", layout="wide")

# ---- Read query params from URL ----
query_params = st.query_params
user = query_params.get("user", None)
csv_filename = query_params.get("csv", None)
survey_url = query_params.get("surveyurl", None)
dzc_file = query_params.get("dzc", None)

# ---- Page title and description ----
st.title("🔍 Factor Contribution Explorer")
st.markdown("**Assess how combinations of attributes contribute to outcome accuracy in a survey dataset.**")

# ---- Diagnostics and file loading ----
with st.expander("⚙️ Diagnostics and Input Info", expanded=False):
    st.markdown(f"🧪 <span style='font-size: 0.85em;'>**Streamlit version:** {st.__version__}</span>", unsafe_allow_html=True)
    st.markdown(f"👤 <span style='font-size: 0.85em;'>**User:** {user}</span>", unsafe_allow_html=True)
    st.markdown(f"📂 <span style='font-size: 0.85em;'>**CSV File:** {csv_filename}</span>", unsafe_allow_html=True)

    if not csv_filename or not survey_url:
        st.error("❌ Missing `csv` filename or `surveyurl` in query parameters.")
        st.stop()

    parsed = urlparse(survey_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}/surveys/"
    csv_url = base_url + csv_filename
    st.markdown(f"🔗 <span style='font-size: 0.85em;'>Trying URL: `{csv_url}`</span>", unsafe_allow_html=True)

    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"❌ Could not fetch CSV from SuAVE: {e}")
        st.stop()

    df.columns = df.columns.str.strip()
    st.markdown("📋 First few rows of data:")
    st.write(df.head(3))
    st.markdown(f"✅ Dataset loaded with **{df.shape[0]} rows** and **{df.shape[1]} columns**")


##### --- Section 2 - Variable Selection UI ---

# ---- Section 2: Select Target and Explanatory Variables ----
st.markdown("## 🎯 Step 2: Define Target and Explanatory Variables")

categorical_cols = df.select_dtypes(include='object').columns.tolist()
target_variable = st.selectbox("🎯 Target Variable (A)", categorical_cols, key="target_var")
target_value = st.selectbox(f"🎯 Value in {target_variable} to Explain (A1)", df[target_variable].unique().tolist(), key="target_val")

independent_vars = st.multiselect("📋 Explanatory Variables (B, C, D, etc.)", [c for c in categorical_cols if c != target_variable], max_selections=4)



##### --- Section 3 - Compute Accuracy and Factor Contributions ---

# ---- Section 3: Generate Explanation Table ----

st.markdown("### 🔎 Optional Filters")

col1, col2, col3 = st.columns([1, 2, 2])
with col1:
    min_n = st.number_input("Min rows (n)", value=5, min_value=0)
with col2:
    min_acc = st.slider("Min accuracy", 0.0, 1.0, 0.0, 0.01)
with col3:
    min_contrib = st.slider("Min contribution", 0.0, 1.0, 0.0, 0.01)



if st.button("🔍 Generate Factor Contribution Table") and target_variable and target_value and independent_vars:
    from itertools import product

    combinations = list(product(*[df[var].unique() for var in independent_vars]))
    rows = []

    for combo in combinations:
        combo_dict = dict(zip(independent_vars, combo))
        condition = (df[list(combo_dict)] == pd.Series(combo_dict)).all(axis=1)
        subset = df[condition]

        if len(subset) == 0:
            continue

        acc_full = (subset[target_variable] == target_value).mean()
        rule_str = " AND ".join([f"{k}={v}" for k, v in combo_dict.items()])
        row = {
            'Rule': rule_str,
            'n': len(subset),
            f'Accuracy if Rule → {target_value}': acc_full
        }

        for var in independent_vars:
            reduced = {k: v for k, v in combo_dict.items() if k != var}
            reduced_df = df[(df[list(reduced)] == pd.Series(reduced)).all(axis=1)]
            if len(reduced_df) > 0:
                acc_reduced = (reduced_df[target_variable] == target_value).mean()
                row[f'Contribution of {var}'] = acc_full - acc_reduced
            else:
                row[f'Contribution of {var}'] = None

        rows.append(row)

        result_df = pd.DataFrame(rows)

        # Apply filters
        filtered_df = result_df.copy()
        filtered_df = filtered_df[filtered_df['n'] >= min_n]
        filtered_df = filtered_df[filtered_df[f'Accuracy if Rule → {target_value}'] >= min_acc]

        if min_contrib > 0:
            contrib_cols = [col for col in result_df.columns if col.startswith("Contribution of")]
            mask = (filtered_df[contrib_cols].abs() >= min_contrib).any(axis=1)
            filtered_df = filtered_df[mask]

        sort_col = st.selectbox("📊 Sort by column", options=filtered_df.columns.tolist(), index=0)
        sort_ascending = st.radio("⬆️ Sort order", ["Descending", "Ascending"]) == "Ascending"
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_ascending)


        if len(filtered_df) == 0:
            st.warning("⚠️ No rows matched the filter criteria.")
        else:
            st.success(f"✅ Showing {len(filtered_df)} matching rule(s)")
            st.session_state.modified_df = filtered_df.copy()
            st.dataframe(filtered_df)
    
    
    
    st.session_state.modified_df = result_df.copy()
    st.dataframe(result_df)

	
##### --- Section 4 - Download Results Table ---

st.markdown("---")
st.subheader("💾 Download Factor Contribution Table")

if "modified_df" in st.session_state and not st.session_state.modified_df.empty:
    csv_out = io.StringIO()
    st.session_state.modified_df.to_csv(csv_out, index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_out.getvalue(),
        file_name="factor_contributions.csv",
        mime="text/csv"
    )
else:
    st.warning("⚠️ No data available to download yet. Please run the analysis first.")


# ---- Return to Home button ----
param_str = urlencode({k: v[0] if isinstance(v, list) else v for k, v in query_params.items()})
launcher_url = "https://suave-launcher.streamlit.app"

button_css = """
<style>
.back-button {
    display: inline-block;
    padding: 0.6em  1.2em;
    margin-top: 2em;
    font-size: 1.1em;
    font-weight: bold;
    color: white !important;
    background-color: #1f77b4;
    border: none;
    border-radius: 8px;
    text-decoration: none;
}
.back-button:hover {
    background-color: #16699b;
    color: white !important;
}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)
st.markdown(f'<a href="{launcher_url}/?{param_str}" class="back-button">⬅️ Return to Home</a>', unsafe_allow_html=True)
