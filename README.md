# SuAVE Factor Contribution App

This Streamlit app allows you to explore how combinations of categorical, boolean, or numeric variables contribute to predicting an outcome in a SuAVE survey. It computes the added predictive value of each variable using rules like "If A and B then C" vs. "If B then C", where A and B are input conditions and C is the target outcome.

## Features

- Load any SuAVE dataset via URL query parameters (`?csv=...&surveyurl=...`)
- Interactive variable selection (target and explanatory)
- Full support for numeric variables via dynamic binning:
  - Interactive histogram visualization of distributions
  - Adjustable bin count and bin edges for numeric columns
  - Uses bin labels as categorical inputs for rule generation
- Generates all rule combinations and computes:
  - Rule count (n)
  - Conditional accuracy of "if A and B then C"
  - Contribution of each variable to prediction accuracy
- Dynamic filters:
  - Minimum rule count
  - Minimum accuracy
  - Minimum absolute contribution
- Visual output:
  - Clean, readable HTML table
  - Highlighted contributions and accuracy
  - Wrapped variable names and line breaks in rule display
  - Emphasized values in rules (blue italic)
- CSV download of filtered results

## How to Launch

ou can use this app on Streamlit Community Cloud:

1. Visit: https://suave-factor.streamlit.app
2. Add your dataset in the URL like this:

```
?user=suavedemos&csv=demo_dataset.csv&surveyurl=https://suave-net.sdsc.edu/main/file=suavedemos_demo_dataset.csv
```

## Deployment

### Requirements

See [requirements.txt](./requirements.txt)

### Deployment Steps

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/suave-factor.git
   cd suave-factor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**

   ```bash
   streamlit run app.py
   ```

4. **Deploy to Streamlit Community Cloud**

   - Push your repo to GitHub
   - Go to https://streamlit.io/cloud
   - Click **"New App"**
   - Select your repo and `app.py`
   - Click **"Deploy"**

---

MIT License · SDSC · Developed by I. Zaslavsky