"""
IPL Auction Intelligence Dashboard
Streamlit App — Full BI Solution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, json, os, shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Auction Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

# ─── LOAD ASSETS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf        = joblib.load(f"{MODEL_DIR}/price_predictor.pkl")
    classifier = joblib.load(f"{MODEL_DIR}/sold_classifier.pkl")
    scaler    = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
    le_role   = joblib.load(f"{MODEL_DIR}/le_role.pkl")
    le_nat    = joblib.load(f"{MODEL_DIR}/le_nat.pkl")
    return rf, classifier, scaler, explainer, le_role, le_nat

@st.cache_data
def load_data():
    df      = pd.read_csv(f"{MODEL_DIR}/processed_data.csv")
    top_val = pd.read_csv(f"{MODEL_DIR}/top_value_players.csv")
    fran    = pd.read_csv(f"{MODEL_DIR}/franchise_efficiency.csv")

    rename_map = {
        "playername": "player_name",
        "iplexperienceyears": "ipl_experience_years",
        "battingavg": "batting_avg",
        "battingsr": "batting_sr",
        "runsscored": "runs_scored",
        "economyrate": "economy_rate",
        "basepricecr": "base_price_cr",
        "soldpricecr": "sold_price_cr",
        "performancescore": "performance_score",
        "valueindex": "value_index",
        "bowlingavg": "bowling_avg",
        "bowlingsr": "bowling_sr",
    }

    df      = df.rename(columns=rename_map)
    top_val = top_val.rename(columns=rename_map)

    with open(f"{MODEL_DIR}/feature_names.json") as f:
        features = json.load(f)
    with open(f"{MODEL_DIR}/regression_metrics.json") as f:
        metrics = json.load(f)
    with open(f"{MODEL_DIR}/summary_stats.json") as f:
        stats = json.load(f)
    return df, top_val, fran, features, metrics, stats

rf, classifier, scaler, explainer, le_role, le_nat = load_models()
df, top_val, fran_eff, FEATURES, reg_metrics, stats = load_data()

ROLES      = sorted([r for r in df['role'].unique() if pd.notna(r)])
NATIONS    = sorted([n for n in df['nationality'].unique() if pd.notna(n)])
FRANCHISES = sorted([f for f in df['franchise'].unique() if f != 'Unsold'])
YEARS      = sorted(df['year'].unique())

# ─── COLOR PALETTE ──────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "quaternary": "#d62728",
    "neutral": "#7f7f7f",
    "blue": "#1f77b4",
    "coral": "#ff7f0e",
    "green": "#2ca02c",
    "purple": "#9467bd",
    "gold": "#bcbd22"
}

FRANCHISE_COLORS = {fran: PALETTE["primary"] for fran in FRANCHISES}
ROLE_COLORS = {
    "Batsman": PALETTE["blue"],
    "Bowler": PALETTE["coral"],
    "All-Rounder": PALETTE["green"]
}
ROLE_COLORS = {role: ROLE_COLORS.get(role, PALETTE["neutral"]) for role in ROLES}
NATIONALITY_COLORS = {nation: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                      for i, nation in enumerate(NATIONS)}

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stMetric { background-color: white; border-radius: 5px; padding: 10px; }
    .insight-box { background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 10px; margin: 10px 0; }
    .warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; margin: 10px 0; }
    .success-box { background-color: #d4edda; border-left: 5px solid #28a745; padding: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏏 IPL Auction Dashboard")
    st.markdown("📊 *Real-time Analytics & ML Insights*")
    st.divider()
    
    page = st.radio("Navigate", [
        "📊 Overview Dashboard",
        "🔍 Player Analysis",
        "💡 Value Finder",
        "🤖 Price Predictor (AI)",
        "🏢 Franchise Insights",
    ])
    
    st.divider()
    st.markdown(f"""
    **📈 Dataset Stats:**
    - {stats['total_records']} auction entries
    - Years: 2015–2026
    - Model R²: {reg_metrics['r2']:.4f}
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.title("📊 Overview Dashboard")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Entries", f"{stats['total_records']}")
    c2.metric("Players Sold", f"{stats['total_sold']}")
    c3.metric("Avg Sale Price", f"₹{stats['avg_price_overall']} Cr")
    c4.metric("Highest Bid", f"₹{stats['max_price']} Cr")
    c5.metric("Unsold Players", f"{stats['total_unsold']}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Avg Auction Price Trend")
        yearly = df.groupby('year')['sold_price_cr'].agg(['mean', 'median']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['mean'].round(2),
                                  mode='lines+markers', name='Mean Price',
                                  line=dict(color=PALETTE['blue'], width=2.5),
                                  marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['median'].round(2),
                                  mode='lines+markers', name='Median Price',
                                  line=dict(color=PALETTE['coral'], width=2, dash='dash'),
                                  marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=list(yearly['year']) + list(yearly['year'][::-1]),
                                  y=list(yearly['mean']) + list(yearly['median'][::-1]),
                                  fill='toself', fillcolor='rgba(55,138,221,0.08)',
                                  line=dict(color='rgba(0,0,0,0)'), showlegend=False))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="₹ Crore", xaxis_title="Year",
                           legend=dict(x=0.02, y=0.98))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price Distribution by Role")
        fig = go.Figure()
        for role in ROLES:
            data = df[df['role'] == role]['sold_price_cr']
            fig.add_trace(go.Histogram(x=data, name=role, opacity=0.7,
                                        marker_color=ROLE_COLORS[role], nbinsx=20))
        fig.update_layout(barmode='overlay', height=320,
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Sold Price (₹ Cr)", yaxis_title="Count",
                           legend=dict(x=0.7, y=0.98))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Franchise Total Spend by Year")
        pivot = df.groupby(['year', 'franchise'])['sold_price_cr'].sum().reset_index()
        pivot_top = pivot[pivot['franchise'].isin(FRANCHISES[:6])]
        fig = px.line(pivot_top, x='year', y='sold_price_cr', color='franchise',
                       markers=True, height=320)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="Total Spend (₹ Cr)", xaxis_title="Year",
                           legend=dict(x=0, y=1))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Nationality Breakdown of Sold Players")
        nat_counts = df['nationality'].value_counts().reset_index()
        nat_counts.columns = ['nationality', 'count']
        fig = px.pie(nat_counts, names='nationality', values='count',
                      color_discrete_sequence=list(PALETTE.values()), height=320)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Key insights
    st.subheader("📌 Key Insights")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        best_value_team = fran_eff.sort_values('avg_value_index', ascending=False).iloc[0]
        st.markdown(f"""<div class="success-box">
        <b>Best Value Franchise</b><br>
        {best_value_team['franchise']} gets the most performance per ₹ crore spent
        (Value Index: {best_value_team['avg_value_index']:.1f})
        </div>""", unsafe_allow_html=True)
    with ic2:
        biggest_spender = fran_eff.sort_values('total_spend', ascending=False).iloc[0]
        st.markdown(f"""<div class="warning-box">
        <b>Biggest Spender</b><br>
        {biggest_spender['franchise']} spent ₹{biggest_spender['total_spend']:.0f} Cr total
        across all auctions — ₹{biggest_spender['avg_price']:.1f} Cr avg per player
        </div>""", unsafe_allow_html=True)
    with ic3:
        price_growth = ((df[df['year']==2024]['sold_price_cr'].mean() /
                          df[df['year']==2015]['sold_price_cr'].mean() - 1) * 100)
        st.markdown(f"""<div class="insight-box">
        <b>Price Inflation</b><br>
        Avg auction prices grew <b>{price_growth:.0f}%</b> from 2015 to 2024 —
        roughly 12% per year, driven by media rights escalation
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Player Analysis":
    st.title("🔍 Player Analysis")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_role = st.selectbox("Filter by Role", ["All"] + ROLES)
    with col_f2:
        sel_nat = st.selectbox("Filter by Nationality", ["All"] + NATIONS)
    with col_f3:
        year_range = st.select_slider("Year Range", options=YEARS, value=(YEARS[0], YEARS[-1]))

    fdf = df.copy()
    if sel_role != "All":   fdf = fdf[fdf['role'] == sel_role]
    if sel_nat  != "All":   fdf = fdf[fdf['nationality'] == sel_nat]
    fdf = fdf[(fdf['year'] >= year_range[0]) & (fdf['year'] <= year_range[1])]

    st.markdown(f"*Showing {len(fdf)} auction entries matching filters*")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Score vs Auction Price")
        fig = px.scatter(fdf, x='performance_score', y='sold_price_cr',
                          color='nationality', symbol='role', size='ipl_experience_years',
                          hover_data=['player_name', 'year', 'franchise'],
                          color_discrete_map=NATIONALITY_COLORS,
                          symbol_map={
                              'Batsman': 'circle',
                              'Bowler': 'square',
                              'All-Rounder': 'diamond'
                          },
                          height=380,
                          labels={'sold_price_cr': 'Sold Price (₹ Cr)',
                                  'performance_score': 'Performance Score'})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age Distribution & Price Heatmap")
        fig = px.density_heatmap(fdf, x='age', y='sold_price_cr',
                                  nbinsx=15, nbinsy=15, height=380,
                                  color_continuous_scale='Blues',
                                  labels={'sold_price_cr': 'Sold Price (₹ Cr)', 'age': 'Age'})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Career Batting Strike Rate vs Price")
    fig = px.scatter(fdf[fdf['role'] != 'Bowler'], x='batting_sr', y='sold_price_cr',
                      color='role', trendline='ols', hover_data=['player_name','year'],
                      color_discrete_map=ROLE_COLORS, height=320,
                      labels={'sold_price_cr': 'Price (₹ Cr)', 'batting_sr': 'Batting SR'})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Player Data Table")
    display_cols = ['player_name','role','nationality','year','age','batting_avg',
                    'batting_sr','wickets','economy_rate','base_price_cr','sold_price_cr',
                    'franchise','value_index']
    st.dataframe(fdf[display_cols].sort_values('sold_price_cr', ascending=False).reset_index(drop=True),
                 use_container_width=True, height=320)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VALUE FINDER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Value Finder":
    st.title("💡 Value Finder")

    col1, col2 = st.columns([1, 2])
    with col1:
        budget = st.slider("Max Price per Player (₹ Cr)", 1.0, 15.0, 8.0, 0.5)
        role_filter = st.multiselect("Roles", ROLES, default=ROLES)
        nat_filter  = st.multiselect("Nationality", NATIONS, default=NATIONS)
        top_n = st.slider("Show Top N Players", 5, 20, 10)

    filtered = df[
        (df['sold_price_cr'] <= budget) &
        (df['role'].isin(role_filter)) &
        (df['nationality'].isin(nat_filter))
    ].copy()

    top_players = filtered.nlargest(top_n, 'value_index').reset_index(drop=True)

    with col2:
        st.subheader(f"Top {top_n} Value Picks (≤ ₹{budget} Cr)")
        if len(top_players) == 0:
            st.warning("No players match your filters. Try adjusting the budget or roles.")
        else:
            fig = go.Figure()
            colors = [ROLE_COLORS[r] for r in top_players['role']]
            fig.add_trace(go.Bar(
                y=top_players['player_name'],
                x=top_players['value_index'],
                orientation='h',
                marker_color=colors,
                text=[f"₹{p:.1f}Cr | {s:.0f} pts" for p, s in
                      zip(top_players['sold_price_cr'], top_players['performance_score'])],
                textposition='inside'
            ))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Value Index (performance ÷ price)",
                               yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    if len(top_players) > 0:
        st.subheader("Budget Optimizer — Best XI within Cap")
        cap = st.number_input("Total Team Budget (₹ Cr)", value=80.0, min_value=20.0, max_value=200.0)

        # Greedy optimizer: sort by value index, pick within cap
        pool = filtered.sort_values('value_index', ascending=False).copy()
        selected, total_spend = [], 0
        role_counts = {"Batsman": 0, "Bowler": 0, "All-Rounder": 0}
        role_limits  = {"Batsman": 5, "Bowler": 5, "All-Rounder": 4}

        for _, row in pool.iterrows():
            if len(selected) >= 11: break
            if total_spend + row['sold_price_cr'] <= cap:
                if role_counts[row['role']] < role_limits[row['role']]:
                    selected.append(row)
                    total_spend += row['sold_price_cr']
                    role_counts[row['role']] += 1

        if selected:
            sel_df = pd.DataFrame(selected)[
                ['player_name','role','nationality','sold_price_cr','performance_score','value_index']
            ].reset_index(drop=True)
            sel_df.index += 1

            c1, c2, c3 = st.columns(3)
            c1.metric("Players Selected", len(selected))
            c2.metric("Total Spend", f"₹{total_spend:.1f} Cr")
            c3.metric("Budget Remaining", f"₹{cap - total_spend:.1f} Cr")

            st.dataframe(sel_df, use_container_width=True)

            st.markdown(f"""<div class="success-box">
            <b>Optimizer recommendation:</b> This XI costs <b>₹{total_spend:.1f} Cr</b> out of
            your ₹{cap} Cr budget, leaving ₹{cap-total_spend:.1f} Cr for retention or mid-season trades.
            Avg value index: {sel_df['value_index'].mean():.1f} — significantly above auction average of
            {df['value_index'].mean():.1f}.
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRICE PREDICTOR (AI)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Price Predictor (AI)":
    st.title("🤖 Price Predictor (AI)")

    st.markdown(f"""<div class="insight-box">
    Model: Random Forest Regressor | <b>R² = {reg_metrics['r2']}</b> |
    RMSE = ₹{reg_metrics['rmse']} Cr | MAE = ₹{reg_metrics['mae']} Cr
    </div>""", unsafe_allow_html=True)

    with st.form("predictor_form"):
        st.subheader("Player Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            role        = st.selectbox("Role", ROLES)
            nationality = st.selectbox("Nationality", NATIONS)
            age         = st.slider("Age", 18, 42, 26)
            experience  = st.slider("IPL Experience (years)", 0, 12, 3)
            base_price  = st.selectbox("Base Price (₹ Cr)", [0.20, 0.50, 1.0, 1.5, 2.0])
            year        = st.selectbox("Auction Year", YEARS[::-1])

        with c2:
            st.markdown("**Batting Stats**")
            bat_avg  = st.slider("Batting Average", 5.0, 70.0, 30.0, 0.5)
            bat_sr   = st.slider("Batting Strike Rate", 80.0, 220.0, 130.0, 1.0)
            runs     = st.slider("Runs Scored (career)", 0, 1000, 250, 10)
            fifties  = st.slider("Fifties", 0, 20, 3)
            hundreds = st.slider("Hundreds", 0, 5, 0)

        with c3:
            st.markdown("**Bowling Stats**")
            wickets  = st.slider("Wickets (career)", 0, 40, 8)
            economy  = st.slider("Economy Rate", 6.0, 13.0, 8.5, 0.1)
            bowl_avg = st.slider("Bowling Average", 15.0, 60.0, 30.0, 0.5)
            bowl_sr  = st.slider("Bowling Strike Rate", 10.0, 40.0, 20.0, 0.5)

        submitted = st.form_submit_button("🎯 Predict Auction Price", type="primary",
                                           use_container_width=True)

    if submitted:
        # Performance score (same formula as training)
        if role == "Batsman":
            perf = (bat_avg * 0.4) + (bat_sr * 0.3) + (runs * 0.02)
        elif role == "Bowler":
            perf = (wickets * 1.5) + max(0, (10 - economy) * 3) + (runs * 0.01)
        else:
            perf = (bat_avg * 0.25) + (bat_sr * 0.2) + (wickets * 1.2) + max(0, (10 - economy) * 2)

        role_enc = int(le_role.transform([role])[0])
        nat_enc  = int(le_nat.transform([nationality])[0])

        input_data = pd.DataFrame([{
            'role_enc': role_enc, 'nationality_enc': nat_enc, 'age': age,
            'iplexperienceyears': experience, 'battingavg': bat_avg,
            'battingsr': bat_sr, 'runsscored': runs, 'fifties': fifties,
            'hundreds': hundreds, 'wickets': wickets, 'economyrate': economy,
            'bowlingavg': bowl_avg, 'bowlingsr': bowl_sr,
            'basepricecr': base_price, 'performancescore': round(perf, 2),
            'year': year
        }])

        predicted_price = float(rf.predict(input_data)[0])
        predicted_price = max(base_price, predicted_price)
        ci_low  = max(base_price, predicted_price * 0.75)
        ci_high = predicted_price * 1.30

        # Sold probability
        input_scaled = scaler.transform(input_data)
        sold_prob = float(classifier.predict_proba(input_scaled)[0][1])

        st.divider()
        st.subheader("Prediction Results")

        r1, r2_col, r3, r4 = st.columns(4)
        r1.metric("Predicted Price", f"₹{predicted_price:.2f} Cr")
        r2_col.metric("Price Range", f"₹{ci_low:.1f} – ₹{ci_high:.1f} Cr")
        r3.metric("Sale Probability", f"{sold_prob*100:.0f}%")
        r4.metric("Performance Score", f"{perf:.1f}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_price,
            delta={'reference': df['sold_price_cr'].mean(), 'suffix': ' vs avg'},
            gauge={
                'axis': {'range': [0, 25], 'ticksuffix': 'Cr'},
                'bar': {'color': PALETTE['blue']},
                'steps': [
                    {'range': [0, 5],  'color': '#EAF3DE'},
                    {'range': [5, 12], 'color': '#FAEEDA'},
                    {'range': [12, 25],'color': '#FAECE7'}
                ],
                'threshold': {'line': {'color': PALETTE['coral'], 'width': 3},
                              'thickness': 0.75, 'value': predicted_price}
            },
            title={'text': "Predicted Auction Price (₹ Cr)"}
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # SHAP explanation
        st.subheader("🔍 Why this price? (SHAP Explanation)")
        st.markdown("*Each bar shows how much that feature pushes the price up (green) or down (red)*")

        shap_vals = explainer.shap_values(input_data)[0]
        feat_names_display = [
            'Role', 'Nationality', 'Age', 'IPL Experience', 'Batting Avg',
            'Batting SR', 'Runs', 'Fifties', 'Hundreds', 'Wickets',
            'Economy', 'Bowling Avg', 'Bowling SR', 'Base Price', 'Perf. Score', 'Year'
        ]
        shap_df = pd.DataFrame({
            'Feature': feat_names_display,
            'SHAP Value': shap_vals,
            'Direction': ['Increases price' if v > 0 else 'Decreases price' for v in shap_vals]
        }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

        fig = go.Figure(go.Bar(
            y=shap_df['Feature'][::-1],
            x=shap_df['SHAP Value'][::-1],
            orientation='h',
            marker_color=[PALETTE['green'] if v > 0 else PALETTE['coral']
                          for v in shap_df['SHAP Value'][::-1]],
            text=[f"+₹{v:.2f}Cr" if v > 0 else f"₹{v:.2f}Cr" for v in shap_df['SHAP Value'][::-1]],
            textposition='outside'
        ))
        fig.add_vline(x=0, line_dash='solid', line_color='gray', line_width=1)
        fig.update_layout(height=360, margin=dict(l=10, r=80, t=10, b=10),
                           xaxis_title="Price Impact (₹ Cr)")
        st.plotly_chart(fig, use_container_width=True)

        # Plain-English explanation
        top_pos = shap_df[shap_df['SHAP Value'] > 0].iloc[0] if len(shap_df[shap_df['SHAP Value'] > 0]) > 0 else None
        top_neg = shap_df[shap_df['SHAP Value'] < 0].iloc[0] if len(shap_df[shap_df['SHAP Value'] < 0]) > 0 else None

        explanation = f"The model predicts <b>₹{predicted_price:.2f} Cr</b> for this player. "
        if top_pos is not None:
            explanation += f"The biggest price driver is <b>{top_pos['Feature']}</b> (+₹{top_pos['SHAP Value']:.2f} Cr). "
        if top_neg is not None:
            explanation += f"The main factor pulling the price down is <b>{top_neg['Feature']}</b> (₹{top_neg['SHAP Value']:.2f} Cr)."

        st.markdown(f'<div class="insight-box">{explanation}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FRANCHISE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Franchise Insights":
    st.title("🏢 Franchise Strategy Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spend vs Avg Performance")
        fig = px.scatter(fran_eff, x='total_spend', y='avg_performance',
                          size='players_bought', color='avg_value_index',
                          hover_data=['franchise'],
                          text='franchise',
                          color_continuous_scale='RdYlGn',
                          labels={'total_spend': 'Total Spend (₹ Cr)',
                                  'avg_performance': 'Avg Player Performance',
                                  'avg_value_index': 'Value Index'},
                          height=380)
        fig.update_traces(textposition='top center', textfont_size=10)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Franchise Efficiency Ranking")
        eff_sorted = fran_eff.sort_values('avg_value_index', ascending=True)
        colors = [PALETTE['green'] if v >= eff_sorted['avg_value_index'].median()
                  else PALETTE['coral'] for v in eff_sorted['avg_value_index']]
        fig = go.Figure(go.Bar(
            y=eff_sorted['franchise'],
            x=eff_sorted['avg_value_index'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}" for v in eff_sorted['avg_value_index']],
            textposition='inside'
        ))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Avg Value Index (higher = more efficient)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Franchise Spending Over the Years")
    yearly_fran = df.groupby(['year', 'franchise'])['sold_price_cr'].sum().reset_index()
    fig = px.area(yearly_fran, x='year', y='sold_price_cr', color='franchise',
                   height=340,
                   labels={'sold_price_cr': 'Total Spend (₹ Cr)', 'year': 'Year'})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Franchise Stats")
    display_fran = fran_eff.rename(columns={
        'franchise': 'Franchise', 'total_spend': 'Total Spend (₹ Cr)',
        'avg_price': 'Avg Price (₹ Cr)', 'avg_value_index': 'Avg Value Index',
        'players_bought': 'Players Bought', 'avg_performance': 'Avg Performance Score'
    }).sort_values('Avg Value Index', ascending=False).reset_index(drop=True)
    display_fran.index += 1
    st.dataframe(display_fran, use_container_width=True)

    best = display_fran.iloc[0]
    worst = display_fran.iloc[-1]
    st.markdown(f"""<div class="success-box">
    <b>Actionable Insight:</b> {best['Franchise']} leads in value efficiency (index: {best['Avg Value Index']:.1f}),
    spending ₹{best['Avg Price (₹ Cr)']:.1f} Cr avg per player. Compare this to {worst['Franchise']}
    (index: {worst['Avg Value Index']:.1f}, ₹{worst['Avg Price (₹ Cr)']:.1f} Cr avg) — a gap of
    {best['Avg Value Index'] - worst['Avg Value Index']:.1f} value index points. Closing this gap
    could save ₹{(worst['Avg Price (₹ Cr)'] - best['Avg Price (₹ Cr)']):.1f} Cr per player.
    </div>""", unsafe_allow_html=True)
