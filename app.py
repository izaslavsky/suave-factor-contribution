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
min_n = st.number_input("Minimum number of matching rows (n)", value=5, min_value=0)
min_acc = st.slider(f"Minimum accuracy of rule → {target_value}", 0.0, 1.0, 0.0, 0.01)
min_contrib = st.slider("Minimum absolute contribution of any factor", 0.0, 1.0, 0.0, 0.01)



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

	
	
##### --- Section 4 - Optional Save and SuAVE Upload ---

st.markdown("---")
st.subheader("📤 Publish Factor Contribution to SuAVE")

from suave_uploader import upload_to_suave

# Add contribution column if it doesn't exist yet
if contrib is not None and "factor_contribution#number" not in df.columns:
    df["factor_contribution#number"] = None
    df.loc[df_ab.index, "factor_contribution#number"] = contrib
    st.session_state.modified_df = df.copy()
    st.session_state.last_new_var = "factor_contribution#number"

auth_user = st.text_input("🔐 SuAVE Login:")
auth_pass = st.text_input("🔑 SuAVE Password:", type="password")

base_name = csv_filename.replace(".csv", "").split("_", 1)[-1]
suggested_name = f"{base_name}_contrib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
survey_name = st.text_input("📛 Name for New Survey:", value=suggested_name)

if st.button("📦 Upload to SuAVE"):
    if not survey_name or not auth_user or not auth_pass:
        st.warning("⚠️ Please fill in all fields before uploading.")
    else:
        parsed = urlparse(survey_url)
        referer = survey_url.split("/main")[0] + "/"
        df_to_upload = st.session_state.get("modified_df", df)

        if "last_new_var" in st.session_state:
            st.info(f"🔄 The variable **{st.session_state.last_new_var}** will be included in the uploaded survey.")

        success, message, new_url = upload_to_suave(
            df_to_upload,
            survey_name,
            auth_user,
            auth_pass,
            referer,
            dzc_file=query_params.get("dzc", None)
        )

        if success:
            st.success(message)
            st.markdown(f"🔗 [Open New Survey in SuAVE]({new_url})")
        else:
            st.error(f"❌ {message}")

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
