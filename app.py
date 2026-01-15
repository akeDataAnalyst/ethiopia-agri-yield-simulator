#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# Load model and data
# ────────────────────────────────────────────────
model = joblib.load('yield_model.pkl')
df = pd.read_csv('ethiopia_agri_data.csv')

# Precompute some EDA aggregates for tabs
yearly_yield = df.groupby('Year')['Yield_ton_ha'].mean().reset_index()
region_yield = df.groupby('Region')['Yield_ton_ha'].mean().sort_values(ascending=False).reset_index()

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(page_title="Ethiopia Agri Yield Simulator & Insights", layout="wide")
st.title("Ethiopia Agricultural Yield Simulator & Insights")
st.markdown("Interactive dashboard for exploring productivity interventions in Ethiopia. Portfolio project by Aklilu Abera for Dalberg Advisors Analyst role.")

# ────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────
tab_sim, tab_eda, tab_model = st.tabs(["🔧 Yield Simulator", "📊 EDA Insights (Phase 2)", "📈 Model & Insights (Phase 3)"])

# ────────────────────────────────────────────────
# Sidebar – shared inputs
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Controls")

    region = st.selectbox("Focus Region", 
                          ["Afar (Baseline)", "Tigray", "Amhara", "Oromia", 
                           "Central Ethiopia", "Sidama", "South Ethiopia", "South West Ethiopia"])

    crop = st.selectbox("Crop (for context)", 
                        ["Maize", "Sorghum", "Wheat", "Barley", "Teff"])

    rainfall = st.slider("Effective Rainfall (mm/year)", 200, 2400, 500, 50)
    fertilizer = st.slider("Fertilizer Use (kg/ha)", 20, 200, 60, 10)
    market_access = st.slider("Market Access Index (0–1)", 0.3, 1.0, 0.50, 0.05)

    st.subheader("Economic Assumptions")
    farm_size_ha = st.number_input("Avg Farm Size (ha)", value=1.5, step=0.1)
    households = st.number_input("Households Targeted", value=50000, step=1000, format="%d")
    price_per_ton = st.number_input("Price per Ton (USD)", value=400, step=50)
    cost_per_ha = st.number_input("Intervention Cost per ha (USD)", value=150, step=10)

# ────────────────────────────────────────────────
# Prediction logic (used in simulator tab)
# ────────────────────────────────────────────────
input_data = np.array([[rainfall, fertilizer, market_access]])
predicted_yield = model.predict(input_data)[0].round(2)

baseline_input = np.array([[500, 60, 0.50]])
baseline_yield = model.predict(baseline_input)[0].round(2)

uplift_pct = ((predicted_yield - baseline_yield) / baseline_yield * 100).round(1) if baseline_yield > 0 else 0

# Economic calculations – use built-in round() on floats
extra_yield_ha   = predicted_yield - baseline_yield
extra_production = round(extra_yield_ha * farm_size_ha * households, 0)
extra_value      = round(extra_production * price_per_ton, 0)
total_cost       = round(cost_per_ha * farm_size_ha * households, 0)
roi = round(((extra_value - total_cost) / total_cost * 100), 0) if total_cost > 0 else 0

# ────────────────────────────────────────────────
# TAB 1: Yield Simulator
# ────────────────────────────────────────────────
with tab_sim:
    st.subheader(f"Simulator – {region} | {crop}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Yield (t/ha)", f"{predicted_yield}")
    col2.metric("Baseline Yield (Afar-like)", f"{baseline_yield}")
    col3.metric("Yield Uplift", f"{uplift_pct}%", delta_color="normal")

    st.subheader("Estimated Scaled Impact")
    colA, colB, colC = st.columns(3)
    colA.metric("Extra Production (tons)", f"{extra_production:,}")
    colB.metric("Added Value (USD)", f"${extra_value:,}")
    colC.metric("ROI", f"{roi}%")

    st.markdown("**Your Inputs vs Baseline**")
    scenario_df = pd.DataFrame({
        "Parameter": ["Rainfall (mm)", "Fertilizer (kg/ha)", "Market Access", "Predicted Yield (t/ha)", "Uplift (%)"],
        "Value": [rainfall, fertilizer, market_access, predicted_yield, uplift_pct],
        "Baseline": [500, 60, 0.50, baseline_yield, 0.0]
    })
    st.dataframe(scenario_df, hide_index=True, use_container_width=True)

    # Yield comparison bar
    chart_data = pd.DataFrame({"Yield": [baseline_yield, predicted_yield]}, index=["Baseline", "Scenario"])
    st.bar_chart(chart_data, color=["#d3d3d3", "#4CAF50"])

# ────────────────────────────────────────────────
# TAB 2: EDA Insights (Phase 2 Recap)
# ────────────────────────────────────────────────
with tab_eda:
    st.subheader("National Yield Trend (2000–2023)")
    st.line_chart(yearly_yield.set_index('Year')['Yield_ton_ha'], use_container_width=True)

    st.subheader("Average Yield by Region (t/ha)")
    st.bar_chart(region_yield.set_index('Region')['Yield_ton_ha'].sort_values(ascending=True), 
                 color="#2196F3", use_container_width=True)

    st.subheader("Key Phase 2 Highlights")
    st.markdown("""
    - National avg yield: ~2.95 t/ha  
    - Highest: South West Ethiopia (3.50), South Ethiopia (3.49)  
    - Lowest: Afar (1.57), Tigray (2.15)  
    - Strongest driver: Rainfall (corr 0.77)  
    - No clear time trend → stagnation without intervention
    """)

# ────────────────────────────────────────────────
# TAB 3: Model & Insights (Phase 3 Recap)
# ────────────────────────────────────────────────
with tab_model:
    st.subheader("Model Performance & Coefficients")
    st.markdown("""
    **R²**: 0.7137 (explains ~71% of yield variation)  
    **RMSE**: 0.3924 t/ha  

    **Coefficients**:
    - Rainfall_mm: 0.000856  
    - Fertilizer_Use_kg_ha: 0.003023  
    - Market_Access_Index: 1.429643  
    - Intercept: 0.5803
    """)

    st.subheader("Full Package Scenario Recap")
    st.markdown("""
    - Baseline yield: ~1.90 t/ha  
    - Full intervention: ~2.93 t/ha  
    - Uplift: **+54.2%**  
    - Scaled (50k households): **$30.9M** added value, **175% ROI**
    """)

    st.info("Recommendation: Focus integrated packages (irrigation + inputs + market linkages) on vulnerable regions for maximum inclusive impact.")

# Footer
st.markdown("---")
st.caption("Portfolio Project | Aklilu Abera")


# In[ ]:




