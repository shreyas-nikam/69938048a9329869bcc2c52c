import streamlit as st
import pandas as pd
import numpy as np
from source import *

st.set_page_config(page_title="QuLab: Lab 50: Portfolio Optimization with AI Predictions", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 50: Portfolio Optimization with AI Predictions")
st.divider()

# Initialize Session State
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'alpha_scores' not in st.session_state:
    st.session_state['alpha_scores'] = None
if 'historical_returns' not in st.session_state:
    st.session_state['historical_returns'] = None
if 'optimization_result' not in st.session_state:
    st.session_state['optimization_result'] = None
if 'bl_result' not in st.session_state:
    st.session_state['bl_result'] = None
if 'backtest_result' not in st.session_state:
    st.session_state['backtest_result'] = None

# Navigation
navigation_options = [
    "1. Application Overview",
    "2. Input Data Loading",
    "3. Mean-Variance Optimization",
    "4. Turnover Penalization",
    "5. Black-Litterman Integration",
    "6. Walk-Forward Backtest",
    "7. Performance Evaluation"
]

selected_page = st.sidebar.selectbox("Navigate to Step", navigation_options)

# Page 1: Application Overview
if selected_page == "1. Application Overview":
    st.markdown(f"### Purpose of the Application")
    st.markdown(f"This application simulates the process of integrating machine learning (ML) insights into portfolio construction. Designed for investment professionals, it allows users to apply ML alpha scores within a constrained Mean-Variance Optimization (MVO) framework, assess turnover costs, blend views using the Black-Litterman model, and validate strategies via walk-forward backtesting.")

    st.markdown(f"### High-Level Story Flow")
    st.markdown(f"1. **Input Data Loading:** Load ML alpha scores and historical stock returns.\n2. **Mean-Variance Optimization (MVO):** Construct portfolios balancing ML-predicted returns against risk.\n3. **Turnover Penalization:** Analyze the trade-off between alpha capture and transaction costs.\n4. **Black-Litterman Integration:** Incorporate confidence levels to blend ML views with market equilibrium.\n5. **Walk-Forward Backtest:** Simulate real-world performance over time.\n6. **Performance Evaluation:** Visualize the efficient frontier and attribute performance sources.")

# Page 2: Input Data Loading
elif selected_page == "2. Input Data Loading":
    st.header("Input Data Loading")
    st.markdown(f"Upload the Machine Learning Alpha Scores and Historical Returns CSV files to begin the analysis. If no files are uploaded, sample data can be generated for demonstration purposes.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_alpha = st.file_uploader("Upload ML Alpha Scores (CSV)", type=['csv'])
    with col2:
        uploaded_returns = st.file_uploader("Upload Historical Returns (CSV)", type=['csv'])

    if st.button("Load and Prepare Data"):
        try:
            # Simulate loading from upload or default
            alpha_df = pd.read_csv(uploaded_alpha) if uploaded_alpha else None
            returns_df = pd.read_csv(uploaded_returns) if uploaded_returns else None
            
            # Call source function
            data_inputs = prepare_optimization_inputs(alpha_df, returns_df)
            
            st.session_state['alpha_scores'] = data_inputs['alpha_scores']
            st.session_state['historical_returns'] = data_inputs['historical_returns']
            st.session_state['data_loaded'] = True
            
            st.success("Data successfully loaded and prepared!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.warning("Please ensure the CSV files are formatted correctly.")

    if st.session_state['data_loaded']:
        st.markdown(f"#### Data Preview")
        st.write("**ML Alpha Scores:**")
        st.dataframe(st.session_state['alpha_scores'].head())
        st.write("**Historical Returns:**")
        st.dataframe(st.session_state['historical_returns'].head())

# Page 3: Mean-Variance Optimization
elif selected_page == "3. Mean-Variance Optimization":
    st.header("Mean-Variance Optimization (MVO)")
    
    if not st.session_state['data_loaded']:
        st.warning("Please load data in Step 2 first.")
    else:
        st.markdown(f"The objective is to find portfolio weights that maximize the utility function, balancing expected return from ML alphas against portfolio variance.")
        
        st.markdown(r"$$ \max_{w} \left( w^T \mu_{ML} - \frac{\lambda}{2} w^T \Sigma w \right) $$")
        st.markdown(r"$$ \text{s.t. } \sum w_i = 1, \quad w_{min} \le w_i \le w_{max} $$")

        col1, col2 = st.columns(2)
        with col1:
            risk_aversion = st.slider("Risk Aversion ($\lambda$)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
        with col2:
            constraint_type = st.selectbox("Constraint Type", options=["Long Only", "Long/Short"], index=0)

        if st.button("Optimize Portfolio"):
            try:
                results = optimize_portfolio(
                    returns=st.session_state['historical_returns'],
                    alphas=st.session_state['alpha_scores'],
                    risk_aversion=risk_aversion,
                    constraint_type=constraint_type
                )
                st.session_state['optimization_result'] = results
                st.success("Optimization Complete")
                
                st.markdown(f"### Optimization Results")
                st.write("**Optimal Weights:**")
                st.dataframe(results['weights'])
                st.write(f"**Expected Portfolio Return:** {results['expected_return']:.4f}")
                st.write(f"**Portfolio Volatility:** {results['volatility']:.4f}")
                
                # Histogram of weights
                st.bar_chart(results['weights'])
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")

# Page 4: Turnover Penalization
elif selected_page == "4. Turnover Penalization":
    st.header("Turnover Sensitivity Analysis")
    
    if not st.session_state['data_loaded']:
        st.warning("Please load data in Step 2 first.")
    else:
        st.markdown(f"High turnover can erode alpha due to transaction costs. We introduce a penalty term to the objective function to control rebalancing frequency.")
        
        st.markdown(r"$$ \max_{w} \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w - \gamma ||w - w_{prev}||_1 \right) $$")
        st.markdown(r"$$ \text{where } \gamma \text{ is the turnover penalty parameter.} $$")

        penalty_gamma = st.slider("Turnover Penalty ($\gamma$)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
        
        if st.button("Run Sensitivity Analysis"):
            try:
                sensitivity_results = turnover_sensitivity(
                    returns=st.session_state['historical_returns'],
                    alphas=st.session_state['alpha_scores'],
                    penalty_gamma=penalty_gamma
                )
                
                st.markdown(f"### Sensitivity Results")
                st.line_chart(sensitivity_results['turnover_vs_return'])
                st.write("The chart above illustrates how increasing the turnover penalty affects the net portfolio return.")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# Page 5: Black-Litterman Integration
elif selected_page == "5. Black-Litterman Integration":
    st.header("Black-Litterman Model")
    
    if not st.session_state['data_loaded']:
        st.warning("Please load data in Step 2 first.")
    else:
        st.markdown(f"The Black-Litterman model allows us to blend the market equilibrium returns (Prior) with our ML-based alpha scores (Views), weighted by our confidence in the ML model.")
        
        st.markdown(r"$$ E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q] $$")
        
        confidence_level = st.slider("Confidence in ML Views (0 = Market, 1 = Full ML)", 0.0, 1.0, 0.5)
        
        if st.button("Calculate Posterior Returns"):
            try:
                bl_results = black_litterman(
                    returns=st.session_state['historical_returns'],
                    alphas=st.session_state['alpha_scores'],
                    confidence=confidence_level
                )
                st.session_state['bl_result'] = bl_results
                
                st.markdown(f"### Posterior Returns Distribution")
                st.bar_chart(bl_results['posterior_returns'])
                
                st.markdown(f"This chart compares the blended expected returns against the original market equilibrium implied returns.")
                
            except Exception as e:
                st.error(f"Calculation failed: {e}")

# Page 6: Walk-Forward Backtest
elif selected_page == "6. Walk-Forward Backtest":
    st.header("Walk-Forward Backtest")
    
    if not st.session_state['data_loaded']:
        st.warning("Please load data in Step 2 first.")
    else:
        st.markdown(f"Simulate the strategy performance over time using a rolling window approach. This tests the robustness of the ML signals and optimization constraints in a dynamic market environment.")
        
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.number_input("Rolling Window Size (Days)", min_value=30, max_value=365, value=60)
        with col2:
            rebalance_freq = st.selectbox("Rebalance Frequency", options=["Daily", "Weekly", "Monthly"], index=2)
            
        if st.button("Run Backtest"):
            with st.spinner("Running Walk-Forward Simulation..."):
                try:
                    backtest_res = walk_forward_optimized_backtest(
                        returns=st.session_state['historical_returns'],
                        alphas=st.session_state['alpha_scores'],
                        window=window_size,
                        frequency=rebalance_freq
                    )
                    st.session_state['backtest_result'] = backtest_res
                    st.success("Backtest Complete")
                    
                    st.markdown(f"### Cumulative Performance")
                    st.line_chart(backtest_res['equity_curve'])
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

# Page 7: Performance Evaluation
elif selected_page == "7. Performance Evaluation":
    st.header("Performance Evaluation")
    
    if st.session_state['backtest_result'] is None:
        st.warning("Please run the Backtest in Step 6 first.")
    else:
        st.markdown(f"Analyze the risk-adjusted performance of the AI-optimized portfolio compared to benchmarks.")
        
        try:
            st.markdown(f"### Efficient Frontier")
            fig_frontier = plot_efficient_frontier(st.session_state['backtest_result'])
            st.pyplot(fig_frontier)
            
            st.markdown(f"### Performance Attribution")
            fig_attrib = plot_performance_attribution(st.session_state['backtest_result'])
            st.pyplot(fig_attrib)
            
            st.markdown(f"**Key Metrics:**")
            metrics = st.session_state['backtest_result'].get('metrics', {})
            st.json(metrics)
            
        except Exception as e:
            st.error(f"Visualization failed: {e}")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
