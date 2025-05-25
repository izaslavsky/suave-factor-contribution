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

st.markdown("---")
st.subheader("🧠 Define Rule: If A and B then C")

st.markdown(
    "You can evaluate how combinations of two factors (A and B) predict an outcome (C). "
    "We'll compute the conditional accuracy of 'If A and B then C' and assess the unique contribution of A."
)

# --- Column selection ---
non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()
categorical_cols = [col for col in non_numeric_cols if df[col].nunique() < 20]

if len(categorical_cols) < 3:
    st.warning("⚠️ Please ensure your dataset includes at least 3 categorical columns with fewer than 20 unique values.")
    st.stop()

# -- Select A, B, and C columns --
col_A = st.selectbox("🔷 Select A (1st condition column)", categorical_cols, index=0)
val_A = st.selectbox(f"Value for A (from column `{col_A}`)", sorted(df[col_A].dropna().unique().tolist()))

col_B = st.selectbox("🔶 Select B (2nd condition column)", [col for col in categorical_cols if col != col_A])
val_B = st.selectbox(f"Value for B (from column `{col_B}`)", sorted(df[col_B].dropna().unique().tolist()))

col_C = st.selectbox("🔴 Select C (Outcome column)", [col for col in categorical_cols if col not in [col_A, col_B]])
val_C = st.selectbox(f"Value for C (from column `{col_C}`)", sorted(df[col_C].dropna().unique().tolist()))

# Save in session state for later
st.session_state.condition_def = {
    "col_A": col_A, "val_A": val_A,
    "col_B": col_B, "val_B": val_B,
    "col_C": col_C, "val_C": val_C
}


##### --- Section 3 - Compute Accuracy and Factor Contributions ---

st.markdown("---")
st.subheader("📊 Compute Accuracy and Factor Contribution")

if "condition_def" not in st.session_state:
    st.warning("Please define A, B, and C conditions above.")
    st.stop()

cd = st.session_state.condition_def

# --- Filters ---
df_ab = df[(df[cd["col_A"]] == cd["val_A"]) & (df[cd["col_B"]] == cd["val_B"])]
df_b = df[df[cd["col_B"]] == cd["val_B"]]

# --- Accuracy Calculations ---
def safe_accuracy(subset, col_c, val_c):
    if len(subset) == 0:
        return None
    return (subset[col_c] == val_c).mean()

acc_ab = safe_accuracy(df_ab, cd["col_C"], cd["val_C"])
acc_b = safe_accuracy(df_b, cd["col_C"], cd["val_C"])
contrib = acc_ab - acc_b if acc_ab is not None and acc_b is not None else None

# --- Display results ---
st.markdown("### 🧮 Results")

def fmt(x):
    return f"{x:.3f}" if x is not None else "N/A"

st.markdown(f"**Accuracy if A and B then C**: {fmt(acc_ab)}")
st.markdown(f"**Accuracy if B then C**: {fmt(acc_b)}")
st.markdown(f"**Estimated Contribution of A**: {fmt(contrib)}")

if contrib is not None:
    st.success(f"✅ Contribution of A is **{fmt(contrib)}**")
else:
    st.error("❌ Cannot compute contribution: one of the required subsets is empty.")
	
	
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
