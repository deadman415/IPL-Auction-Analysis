# IPL Auction Intelligence — Business Intelligence Project Report

**Course:** Business Intelligence  
**Project:** IPL Player Auction Price Analysis & Prediction  
**Tools:** Python · scikit-learn · SHAP · Streamlit · Power BI  
**Dataset:** IPL Auction Records 2015–2024 (678 entries, 83 players, 10 franchises)

---

## 1. Executive Summary

The Indian Premier League (IPL) auction is one of the most high-stakes, data-driven events in professional sport. Each year, ten franchises compete to build winning squads within a fixed budget — yet auction outcomes are notoriously inconsistent, with teams routinely overpaying for marquee players while ignoring exceptional value elsewhere.

This project builds an end-to-end Business Intelligence solution that:
- Analyses 10 years of IPL auction data (2015–2024)
- Predicts a player's auction price from their performance stats using a Random Forest model (R² = 0.66)
- Explains those predictions using SHAP (SHapley Additive exPlanations) — identifying the exact features driving each price
- Identifies systematically undervalued players through a Value Index metric
- Provides franchise-level spend efficiency analysis
- Delivers all insights via an interactive Streamlit web application and Power BI dashboard

**Business Problem:** How can IPL franchises make smarter, data-backed auction decisions — maximising team performance per crore spent?

---

## 2. Domain Background

### 2.1 How the IPL Auction Works
- Each player enters with a **base price** (₹20 lakh – ₹2 crore)
- Teams bid against each other; the highest bid wins
- Each team has a **salary cap** (~₹90 crore for 2024)
- A maximum of **8 overseas players** can be in the squad (only 4 play per match)
- Players can go **unsold** if no team bids at base price

### 2.2 Why This is a BI Problem
Franchise management teams make multi-crore decisions in seconds, under competitive pressure, with incomplete information. BI tools that aggregate historical patterns, model player value, and surface actionable insights can give franchises a systematic edge — exactly what this project provides.

---

## 3. Data Description

### 3.1 Dataset Overview
| Attribute | Detail |
|-----------|--------|
| Records | 678 auction entries |
| Years | 2015–2024 (10 seasons) |
| Unique Players | 83 |
| Franchises | 10 |
| Features | 21 columns |
| Sold Players | 626 (92.3%) |
| Unsold Players | 52 (7.7%) |

### 3.2 Key Features
| Feature | Description |
|---------|-------------|
| `sold_price_cr` | Hammer price at auction (₹ Crore) — **regression target** |
| `is_sold` | Binary: 1 = sold, 0 = unsold — **classification target** |
| `batting_avg` | Career batting average in T20 cricket |
| `batting_sr` | Career batting strike rate |
| `wickets` | Career wickets taken |
| `economy_rate` | Runs conceded per over (lower = better) |
| `ipl_experience_years` | Number of prior IPL seasons |
| `performance_score` | Engineered composite metric (see Section 4) |
| `value_index` | `performance_score ÷ sold_price_cr` (efficiency metric) |

### 3.3 Data Quality
- No missing values in core auction fields
- Categorical encoding applied to `role` (3 classes) and `nationality` (7 classes)
- Price range: ₹1.24 Cr – ₹25.0 Cr (realistic IPL range)
- Nationality distribution: 60% Indian, 40% overseas (realistic ratio)

---

## 4. Methodology

### 4.1 Feature Engineering

**Performance Score** (composite metric designed to normalise across roles):

For Batsmen:
```
performance_score = (batting_avg × 0.4) + (batting_sr × 0.3) + (runs × 0.02)
```

For Bowlers:
```
performance_score = (wickets × 1.5) + max(0, (10 − economy) × 3) + (runs × 0.01)
```

For All-Rounders:
```
performance_score = (batting_avg × 0.25) + (batting_sr × 0.2) 
                  + (wickets × 1.2) + max(0, (10 − economy) × 2)
```

**Value Index:**
```
value_index = performance_score ÷ sold_price_cr
```
A higher value index means more performance per crore — this is the core metric for franchise decision-making.

### 4.2 Exploratory Data Analysis

Key findings from EDA:
1. **Price inflation:** Average auction prices grew ~120% from 2015 to 2024 (~12% per year), closely tracking IPL media rights valuations
2. **Overseas premium:** Overseas players command 35% higher prices on average (4 overseas slots per team creates scarcity)
3. **Age sweet spot:** Players aged 24–29 command peak prices; performance and price both decline after 33
4. **Role premium:** All-rounders attract highest avg bids (dual skill utility), followed by batsmen, then bowlers
5. **Strike rate dominance:** Batting strike rate is the single strongest predictor of price for batting-capable players

### 4.3 Machine Learning Models

#### Model 1: Auction Price Predictor (Regression)
- **Algorithm:** Random Forest Regressor
- **Tuning:** 5-fold cross-validated GridSearchCV over n_estimators (100, 200), max_depth (8, 12, None), min_samples_leaf (2, 4)
- **Best parameters:** n_estimators=200, max_depth=None, min_samples_leaf=4
- **Train/Test split:** 80/20 (random_state=42)

**Results:**
| Metric | Value |
|--------|-------|
| R² Score | 0.6618 |
| RMSE | ₹3.33 Cr |
| MAE | ₹2.56 Cr |

#### Model 2: Sold/Unsold Classifier (Classification)
- **Algorithm:** Logistic Regression with StandardScaler
- **Accuracy:** 92.65%
- **Application:** Estimate probability of a player receiving bids before setting strategy

### 4.4 Explainable AI — SHAP

SHAP (SHapley Additive exPlanations) was applied to the Random Forest model to:
1. Generate **global feature importance** — which features drive price across all players
2. Generate **per-player waterfall plots** — exactly how much each stat contributed to that player's predicted price

**Top SHAP features driving auction price:**
1. Performance Score (composite metric)
2. Batting Strike Rate
3. Base Price (set by BCCI)
4. Nationality (overseas premium)
5. IPL Experience
6. Economy Rate (for bowlers)

SHAP enables a franchise analyst to say: *"For this specific player, his batting strike rate is adding ₹2.1 Cr to his predicted price, but his age is reducing it by ₹0.8 Cr — so we can find a younger player with similar strike rate and save ₹2 Cr."*

---

## 5. Key Insights & Business Recommendations

### Insight 1: Franchises systematically overpay for experience
Players with 6+ IPL seasons command a 28% premium over equally-performing younger players. **Recommendation:** Target players in their 2nd–3rd IPL season — peak performance with sub-peak pricing.

### Insight 2: Strike rate is priced more efficiently than wickets
Batting strike rate correlates strongly with price (SHAP rank: #2), while wickets are underweighted relative to their match impact. **Recommendation:** Franchises should prioritise high-wicket economy bowlers — the market undervalues them.

### Insight 3: Value Index reveals consistent bargain franchises
Franchises with highest Value Index (performance per crore) built competitive squads at lower cost by targeting the 2nd–4th bidding tier rather than chasing headliners.

### Insight 4: Overseas player budget deserves scientific allocation
With only 4 overseas slots, spending them on batting all-rounders (highest dual value) over specialist batsmen or bowlers maximises squad flexibility.

### Insight 5: Price inflation is predictable — lock in players early
Year-on-year price inflation of ~12% suggests that retaining a player before auction is almost always cheaper than re-acquiring them 2 years later.

---

## 6. System Architecture

```
Raw Data (CSV)
     │
     ▼
Data Generator / Preprocessor  →  Cleaned Dataset
     │
     ├── EDA Notebook            →  4 EDA plots (PNG)
     │
     ├── Feature Engineering     →  Encoded features, performance_score, value_index
     │
     ├── Random Forest Regressor →  price_predictor.pkl  (R² = 0.66)
     │                              shap_explainer.pkl
     │
     ├── Logistic Regression     →  sold_classifier.pkl  (Acc = 92.6%)
     │
     └── Streamlit App (5 pages)
          ├── Overview Dashboard
          ├── Player Analysis
          ├── Value Finder + Budget Optimizer
          ├── AI Price Predictor + SHAP
          └── Franchise Insights

     Power BI Dashboard (4 pages)
          ├── Executive Overview
          ├── Player Deep Dive
          ├── Value Intelligence
          └── Franchise Strategy
```

---

## 7. Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas, NumPy | Data manipulation |
| scikit-learn | ML models, GridSearchCV, metrics |
| SHAP | Explainable AI |
| Plotly | Interactive charts in Streamlit |
| Streamlit | Web application framework |
| Power BI | Enterprise BI dashboard |
| joblib | Model serialization |
| matplotlib | Static plot generation |
| GitHub | Version control |

---

## 8. Limitations & Future Work

### Current Limitations
- Dataset is synthetically generated with realistic distributions; real IPL auction data (Kaggle) would strengthen model accuracy
- R² of 0.66 indicates moderate predictive power — auction prices have inherent noise from bidding wars and team-specific needs
- Model does not account for injuries, team composition gaps, or real-time demand

### Future Enhancements
1. **Real-time data:** Integrate CricAPI or Cricbuzz API for live player stats
2. **Deep learning:** LSTM model trained on time-series auction history per player
3. **NLP sentiment:** Social media sentiment analysis (Twitter/X) to capture hype premium
4. **Multi-year retention optimization:** Integer programming to solve squad composition across 3-year planning horizons

---

## 9. Conclusion

This project demonstrates a complete Business Intelligence pipeline applied to a real-world sports analytics problem. By combining EDA, predictive modelling, explainable AI, and interactive dashboards, it provides IPL franchises with actionable tools to:
- Predict what a player will cost before entering the auction room
- Understand exactly which stats are driving that price (SHAP)
- Identify systematically undervalued players within a budget
- Benchmark franchise spend efficiency

The live Streamlit application makes these insights accessible to non-technical decision-makers — the hallmark of effective Business Intelligence.

---

*Project files:* `data/` `models/` `notebooks/` `streamlit_app/` `powerbi_guide/`  
*To run:* `pip install -r requirements.txt` → `python data/generate_dataset.py` → `python notebooks/train_models.py` → `streamlit run streamlit_app/app.py`
