# SuAVE Factor Contribution App

This Streamlit app allows you to explore conditional accuracy contributions from categorical and boolean variables in a SuAVE survey. The contribution is measured as the difference between the accuracy of rules like "If A and B then C" and "If B then C", where A, B, and C are logical conditions over survey variables.

## Features

- Load any SuAVE dataset by passing `?csv=...&surveyurl=...` in the URL
- Select sets of binary/categorical variables to serve as conditions A, B, and target C
- Automatically compute and display:
  - Count of matches for A, B, and C combinations
  - Accuracy of "if A and B then C"
  - Contribution of A to prediction
- Visual preview of selected variable breakdowns
- Publish a new SuAVE dataset with computed variables

## How to Launch

You can use this app on Streamlit Community Cloud:

1. Go to the deployed app URL: https://suave-factor.streamlit.app
2. Append the dataset as query parameters:

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

   Create a virtual environment or install with:

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
   - Choose your repo and `app.py`
   - Click **"Deploy"**

---

MIT License · SDSC · Developed by I. Zaslavsky
