#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Load model and data
# -------------------------------
model = joblib.load('yield_model.pkl')

# Load your dataset for reference visuals (optional)
df = pd.read_csv('ethiopia_agri_data.csv')

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Ethiopia Agri Yield Simulator", layout="wide")
st.title("Ethiopia Agricultural Yield Prediction & Intervention Simulator")
st.markdown("""
Interactive tool to explore how interventions (rainfall via irrigation, fertilizer, market access)  
could boost crop yields in low-performing regions like Afar and Tigray.
""")

# -------------------------------
# Sidebar: User Inputs
# -------------------------------
st.sidebar.header("Simulation Parameters")

# Region selector (for context/display)
region = st.sidebar.selectbox("Focus Region", 
                              ["Afar (Baseline)", "Tigray", "Amhara", "Oromia", 
                               "Central Ethiopia", "Sidama", "South Ethiopia", "South West Ethiopia"])

# Sliders for what-if
rainfall = st.sidebar.slider("Effective Rainfall (mm/year)", 
                             min_value=200, max_value=2400, value=500, step=50)

fertilizer = st.sidebar.slider("Fertilizer Use (kg/ha)", 
                               min_value=20, max_value=200, value=60, step=10)

market_access = st.sidebar.slider("Market Access Index (0–1)", 
                                  min_value=0.3, max_value=1.0, value=0.50, step=0.05)

# Economic assumptions (editable)
st.sidebar.subheader("Economic Assumptions")
farm_size_ha = st.sidebar.number_input("Avg Farm Size (ha)", value=1.5, step=0.1)
households = st.sidebar.number_input("Households Targeted", value=50000, step=1000)
price_per_ton = st.sidebar.number_input("Price per Ton (USD)", value=400, step=50)
cost_per_ha = st.sidebar.number_input("Intervention Cost per ha (USD)", value=150, step=10)

# -------------------------------
# Make Prediction
# -------------------------------
input_data = np.array([[rainfall, fertilizer, market_access]])
predicted_yield = model.predict(input_data)[0]

# Baseline (using your Phase 3 Afar-like: 500mm, 60kg, 0.5 market)
baseline_input = np.array([[500, 60, 0.50]])
baseline_yield = model.predict(baseline_input)[0]

uplift_pct = ((predicted_yield - baseline_yield) / baseline_yield) * 100 if baseline_yield > 0 else 0

# Economic calc
extra_yield_ha = predicted_yield - baseline_yield
extra_production = extra_yield_ha * farm_size_ha * households
extra_value = extra_production * price_per_ton
total_cost = cost_per_ha * farm_size_ha * households
roi = ((extra_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0

# -------------------------------
# Main Content: Results
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Yield")
    st.metric("Predicted Yield (t/ha)", f"{predicted_yield:.2f}")
    st.metric("Baseline Yield (Afar-like)", f"{baseline_yield:.2f}")
    st.metric("Yield Uplift", f"{uplift_pct:.1f}%")

with col2:
    st.subheader("Estimated Impact")
    st.metric("Extra Production (tons)", f"{extra_production:,.0f}")
    st.metric("Added Economic Value (USD)", f"${extra_value:,.0f}")
    st.metric("ROI", f"{roi:.0f}%")

# Scenario comparison table
st.subheader("Scenario vs Baseline")
scenario_df = pd.DataFrame({
    "Metric": ["Rainfall (mm)", "Fertilizer (kg/ha)", "Market Access", "Predicted Yield (t/ha)", "Uplift (%)"],
    "Scenario": [rainfall, fertilizer, market_access, round(predicted_yield, 2), round(uplift_pct, 1)],
    "Baseline (Afar-like)": [500, 60, 0.50, round(baseline_yield, 2), 0.0]
})
st.table(scenario_df)

# Simple bar chart for uplift
fig, ax = plt.subplots()
ax.bar(["Baseline", "Your Scenario"], [baseline_yield, predicted_yield], color=['gray', 'green'])
ax.set_ylabel("Yield (t/ha)")
ax.set_title("Yield Comparison")
st.pyplot(fig)

# -------------------------------
# Insights
# -------------------------------
st.subheader("Key Insights Recap")
st.markdown("""
- **Model R²**: 0.71 → Strong explanatory power  
- **Biggest Driver**: Market Access (coef ~1.43)  
- **Full Package Potential**: Up to +54% yield in vulnerable regions  
- **ROI Example**: 175% for 50k households in pilot  
""")

st.markdown("**Recommendation**: Prioritize integrated interventions (irrigation + inputs + markets) in Afar/Tigray for high-impact, equity-centered growth.")

# Footer
st.markdown("---")
st.caption("Portfolio Project by Aklilu Abera")


# In[ ]:




