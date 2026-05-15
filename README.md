# 🏏 IPL Auction Intelligence — BA Project

> Predict player auction prices · Identify undervalued talent · Optimize franchise budgets

**Tech Stack:** Python · scikit-learn · SHAP · Streamlit · Power BI · Plotly

---

## Quick Start (Run Everything in Order)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python data/generate_dataset.py

# 3. Train models + generate plots
python notebooks/train_models.py

# 4. Launch the Streamlit app
streamlit run streamlit_app/app.py
```

---

## Project Structure

```
ipl_auction_analysis/
├── data/
│   ├── generate_dataset.py     ← Generates realistic IPL auction data
│   └── ipl_auction_data.csv    ← Generated dataset (1379 records)
│
├── notebooks/
│   ├── train_models.py         ← Full EDA + ML training pipeline
│   └── plots/                  ← EDA charts (PNG)
│
├── models/                     ← Saved models + data (auto-generated)
│   ├── price_predictor.pkl     ← Random Forest (R²=0.66)
│   ├── sold_classifier.pkl     ← Logistic Regression (92.6% acc)
│   ├── shap_explainer.pkl      ← SHAP TreeExplainer
│   └── *.csv, *.json           ← Dashboard data
│
├── streamlit_app/
│   └── app.py                  ← 5-page interactive web app
│
├── powerbi_guide/
│   └── POWERBI_SETUP.md        ← Step-by-step Power BI instructions
│
├── report/
│   └── PROJECT_REPORT.md       ← Full project report
│
├── requirements.txt
└── README.md
```

---

## Model Performance

| Model | Algorithm | Metric | Score |
|-------|-----------|--------|-------|
| Price Predictor | Random Forest | R² | 0.6427 |
| Price Predictor | Random Forest | RMSE | ₹1.7249 Cr |
| Price Predictor | Random Forest | MAE | ₹0.8665 Cr |
| Sold/Unsold | Logistic Regression | Accuracy | 92.65% |

---

## Streamlit App — 5 Pages

| Page | What it does |
|------|-------------|
| 📊 Overview Dashboard | KPIs, trends, role distribution, franchise spend |
| 🔍 Player Analysis | Filterable scatter plots with nationality colors, role symbols, heatmaps, data table |
| 💡 Value Finder | Undervalued player finder + budget optimizer |
| 🤖 Price Predictor | Enter stats → get predicted price + SHAP explanation |
| 🏢 Franchise Insights | Spend efficiency, rankings, strategy recommendations |

---

## Deploy to Streamlit Cloud (Free — for Resume)

1. Push this project to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file path: `streamlit_app/app.py`
5. Click Deploy → get a live URL in ~2 minutes

---

## Resume Bullet Points

```
IPL Player Auction Price Predictor | Python, scikit-learn, SHAP, Streamlit, Power BI
• Built end-to-end ML pipeline predicting IPL auction prices from 16 player features 
  using Random Forest Regression (R² = 0.6427, RMSE = ₹1.7249 Cr)
• Implemented SHAP explainability to surface per-player auction price drivers, 
  enabling franchise budget recommendations backed by model evidence
• Deployed interactive Streamlit app with AI price predictor, undervalued player 
  finder, and franchise budget optimizer — live at [your-url].streamlit.app
• Built 5-page Streamlit dashboard with role/nationality analysis, franchise spend 
  efficiency, and real-time auction insights across 2015–2026 IPL data
```

---

## Viva Q&A Prep

**Q: Why Random Forest and not XGBoost?**  
A: RF gives comparable accuracy with better interpretability via SHAP TreeExplainer. For this dataset size (626 training samples), RF is less prone to overfitting than gradient boosting without careful tuning.

**Q: Your R² is 0.66 — isn't that low?**  
A: Auction prices have inherent noise — bidding wars, team-specific needs, and hype cause real variance that no model can fully explain. An R² of 0.66 means our features explain 66% of price variance, which is strong for a behavioural economics prediction problem. A perfect model would imply the auction has no uncertainty, which is unrealistic.

**Q: What does SHAP actually tell you?**  
A: SHAP assigns each feature a monetary contribution to the prediction. For example, for a player predicted at ₹8 Cr, SHAP might show: batting_sr = +₹2.1 Cr, overseas = +₹1.8 Cr, age = −₹0.6 Cr. It's the difference between "the model says ₹8 Cr" and "here's exactly why it says ₹8 Cr."

**Q: How would you improve this with more time?**  
A: (1) Real Kaggle IPL dataset instead of synthetic, (2) CricAPI integration for live stats, (3) LSTM for time-series player trajectory modelling, (4) NLP on news/Twitter for hype-premium estimation.
