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


##### --- Section 2: Define Target and Explanatory Variables ---

st.markdown("### 🎯 Define Target and Explanatory Variables")

import matplotlib.pyplot as plt
import re

original_df = df.copy()
numeric_cols = [col for col in original_df.columns if "#number" in col]
categorical_cols = df.select_dtypes(include='object').columns.tolist()
all_vars = categorical_cols + numeric_cols

rebinned_columns = {}

# Step 2: Choose target variable
target_variable_raw = st.selectbox("🎯 Target Variable (A)", all_vars, key="target_var")
target_variable = target_variable_raw  # Will change to binned name if needed

# Step 3: Define binning logic
def interactive_rebin(var_name, label_prefix, default_bins=3):
    raw = original_df[var_name].dropna()
    clean_name = var_name.replace("#number", "")

    with st.expander(f"🔧 Adjust bins for {label_prefix} Variable: {clean_name}", expanded=True):
        # Layout: histogram on the left (60%), UI on the right (40%)
        col1, col2 = st.columns([3, 2])

        # Plot in left column
        with col1:
            fig, ax = plt.subplots(figsize=(2.2, 1.2))
            ax.hist(raw, bins=30, edgecolor='black')
            ax.set_title(f"Histogram of {clean_name}", fontsize=4)
            ax.tick_params(labelsize=3)
            st.pyplot(fig, use_container_width=True)

        # Bin controls in right column
        with col2:
            n_bins = st.slider(f"Number of bins for {clean_name}", 2, 10, default_bins)
            min_val, max_val = float(raw.min()), float(raw.max())
            edges = np.linspace(min_val, max_val, n_bins + 1)

            new_edges = []
            for i, edge in enumerate(edges):
                new_edge = st.number_input(
                    f"Edge {i+1}", min_val, max_val, edge,
                    step=(max_val - min_val) / 100,
                    key=f"edge_{var_name}_{i}"
                )
                new_edges.append(new_edge)

        cleaned_edges = sorted(set(new_edges))
        if len(cleaned_edges) >= 2:
            binned = pd.cut(raw, bins=cleaned_edges, include_lowest=True)
            new_col = f"__binned__{clean_name}"
            
            def format_bin_label(interval):
                if isinstance(interval, pd._libs.interval.Interval):
                    left, right = interval.left, interval.right
                    # Use adaptive precision based on bin width
                    width = right - left
                    if width < 0.001:
                        return f"[{left:.5f}, {right:.5f})"
                    elif width < 0.01:
                        return f"[{left:.4f}, {right:.4f})"
                    elif width < 0.1:
                        return f"[{left:.3f}, {right:.3f})"
                    elif width < 1:
                        return f"[{left:.2f}, {right:.2f})"
                    elif width < 10:
                        return f"[{left:.1f}, {right:.1f})"
                    else:
                        return f"[{left:.0f}, {right:.0f})"
                return str(interval)

            df[new_col] = pd.Series(binned.map(format_bin_label), index=raw.index)

            
            rebinned_columns[var_name] = new_col
            st.success(f"{label_prefix} binned into {len(cleaned_edges) - 1} intervals.")
            return new_col
        else:
            st.warning("You must specify at least two distinct bin edges.")
            return None

# Step 4: Rebin if numeric target
if "#number" in target_variable_raw:
    binned_col = interactive_rebin(target_variable_raw, label_prefix="Target")
    if binned_col:
        target_variable = binned_col

# Step 5: Confirm target value
if target_variable and target_variable in df.columns:
    value_counts = df[target_variable].value_counts(sort=False)

    def safe_bin_key(x):
        try:
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(x))
            return float(match.group()) if match else float('inf')
        except Exception:
            return float('inf')

    sorted_bins = sorted(value_counts.index.tolist(), key=safe_bin_key)
    display_options = [f"{val} ({value_counts[val]})" for val in sorted_bins]

    selected_display = st.selectbox(
        f"🎯 Value in {target_variable} to Explain",
        display_options,
        key="target_val"
    )

    target_value = selected_display.split(" (")[0]
else:
    target_value = None

# Step 6: Select explanatory variables
candidates = [col for col in all_vars if col != target_variable_raw]
selected_explanatory = st.multiselect("📋 Explanatory Variables (B, C, D, etc.)", options=candidates, max_selections=4)

independent_vars = []
label_lookup = {}
for var in selected_explanatory:
    if "#number" in var:
        binned = rebinned_columns.get(var)
        if not binned:
            binned = interactive_rebin(var, label_prefix="Explanatory")
        if binned and binned in df.columns:
            independent_vars.append(binned)
            label_lookup[binned] = binned.replace("__binned__", "")
    else:
        independent_vars.append(var)
        label_lookup[var] = var

# Also record a cleaned label for the target
if "__binned__" in target_variable:
    label_lookup[target_variable] = target_variable.replace("__binned__", "")
else:
    label_lookup[target_variable] = target_variable

# Trigger computation if ready
if target_variable and target_value and independent_vars:
    pass  # Section 3 will proceed with computation automatically



##### --- Section 3 - Compute Accuracy and Factor Contributions ---

st.markdown("### 🧮 Rules and Factor Contributions")

# Sliders for dynamic filtering – compact layout
with st.container():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        min_n = st.number_input("Min rows (n)", value=5, min_value=0, key="min_n")
    with c2:
        min_acc = st.slider("Min accuracy", 0.0, 1.0, 0.0, 0.01, key="min_acc")
    with c3:
        min_contrib = st.slider("Min |contrib|", 0.0, 1.0, 0.0, 0.01, key="min_contrib")

# Generate combinations and compute contributions
from itertools import product

if target_variable and target_value and independent_vars:
    combinations = list(product(*[df[var].unique() for var in independent_vars]))
    rows = []

    for combo in combinations:
        combo_dict = dict(zip(independent_vars, combo))
        condition = (df[list(combo_dict)] == pd.Series(combo_dict)).all(axis=1)
        subset = df[condition]

        if len(subset) == 0:
            continue

        acc_full = (subset[target_variable] == target_value).mean()
        rule_str = " AND ".join([f"{label_lookup.get(k, k)}={v}" for k, v in combo_dict.items()])
        rule_str = rule_str.replace("__binned__", "")
        row = {
            'Rule': rule_str,
            'n': len(subset),
            f'Accuracy → {target_value}': round(acc_full, 2)
        }

        for var in independent_vars:
            reduced = {k: v for k, v in combo_dict.items() if k != var}
            reduced_df = df[(df[list(reduced)] == pd.Series(reduced)).all(axis=1)]
            acc_reduced = (reduced_df[target_variable] == target_value).mean() if len(reduced_df) > 0 else None
            clean_label = label_lookup.get(var, var).replace("__binned__", "")
            row[f'Contrib of {clean_label}'] = round(acc_full - acc_reduced, 2) if acc_reduced is not None else None


        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.reset_index(drop=True, inplace=True)

    # Apply filters dynamically
    contrib_cols = [c for c in result_df.columns if c.startswith("Contrib of")]
    filtered_df = result_df[
        (result_df['n'] >= min_n) &
        (result_df[f'Accuracy → {target_value}'] >= min_acc)
    ]

    if min_contrib > 0:
        mask = (filtered_df[contrib_cols].abs() >= min_contrib).any(axis=1)
        filtered_df = filtered_df[mask]

    # Sort controls
    st.markdown("#### 🔃 Sort and View")
    with st.container():
        s1, s2 = st.columns([2, 1])
        with s1:
            sort_col = st.selectbox("📊 Sort by", options=filtered_df.columns.tolist(), index=0)
        with s2:
            sort_ascending = st.radio("⬆️ Order", ["Descending", "Ascending"]) == "Ascending"

    filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_ascending)

    # Apply conditional styling
    def highlight(row):
        style = [''] * len(row)
        if row[f'Accuracy → {target_value}'] > 0.8:
            style = ['background-color: lightgreen'] * len(row)
        for i, col in enumerate(row.index):
            if col.startswith("Contrib of") and abs(row[col]) > 0.2:
                style[i] = 'background-color: khaki'
        return style

    # Display table
    if filtered_df.empty:
        st.warning("⚠️ No rules matched the filters.")
    else:
        st.success(f"✅ Showing {len(filtered_df)} rules")
        st.dataframe(
            filtered_df.style
                .apply(highlight, axis=1)
                .format(precision=2),
            use_container_width=True,
            hide_index=True
)

        st.session_state.modified_df = filtered_df.copy()

        # CSV export
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Table as CSV",
            data=csv,
            file_name="factor_contributions.csv",
            mime="text/csv"
        )





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