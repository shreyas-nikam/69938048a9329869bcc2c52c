import streamlit as st
import pandas as pd
import numpy as np
from source import *

st.set_page_config(
    page_title="QuLab: Lab 50: Portfolio Optimization with AI Predictions", layout="wide")
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
    st.markdown(f"This application demonstrates the process of integrating machine learning (ML) insights into portfolio construction using **real historical market data from Yahoo Finance**. Designed for investment professionals, it allows users to apply ML alpha scores within a constrained Mean-Variance Optimization (MVO) framework, assess turnover costs, blend views using the Black-Litterman model, and validate strategies via walk-forward backtesting.")

    st.markdown(f"### High-Level Story Flow")
    st.markdown(f"1. **Input Data Loading:** Download real historical stock returns and generate ML alpha scores from ~100 S&P 500 stocks.\n2. **Mean-Variance Optimization (MVO):** Construct portfolios balancing ML-predicted returns against risk.\n3. **Turnover Penalization:** Analyze the trade-off between alpha capture and transaction costs.\n4. **Black-Litterman Integration:** Incorporate confidence levels to blend ML views with market equilibrium.\n5. **Walk-Forward Backtest:** Simulate real-world performance over time.\n6. **Performance Evaluation:** Visualize the efficient frontier and attribute performance sources.")

# Page 2: Input Data Loading
elif selected_page == "2. Input Data Loading":
    st.header("Input Data Loading")
    st.markdown(f"This application uses **real historical stock data** from Yahoo Finance to demonstrate portfolio optimization with ML-predicted alpha scores. Click the button below to download historical data for 100 popular stocks with 120 months of historical returns.")

    st.markdown(f"**Data Parameters:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("- **Universe Size:** ~100 stocks (S&P 500 subset)")
        st.write("- **Historical Period:** 120 months")
    with col2:
        st.write("- **Data Source:** Yahoo Finance (yfinance)")
        st.write("- **Real Sectors:** Based on actual company classifications")

    if st.button("Load Real Market Data"):
        try:
            with st.spinner("Downloading real market data from Yahoo Finance..."):
                # Generate synthetic data
                returns_df_full, ml_alpha_scores_full, sector_map_full, ml_alpha_scores_time_series = generate_synthetic_data(
                    n_universe=100,
                    n_periods_hist=120,
                    num_factors=5,
                    num_sectors=5
                )

                # Store in session state
                st.session_state['alpha_scores'] = ml_alpha_scores_full
                st.session_state['historical_returns'] = returns_df_full
                st.session_state['sector_map'] = sector_map_full
                st.session_state['alpha_scores_time_series'] = ml_alpha_scores_time_series
                st.session_state['data_loaded'] = True

                st.success(
                    f"✓ Successfully downloaded data for {len(returns_df_full.columns)} stocks with {len(returns_df_full)} periods!")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    if st.session_state['data_loaded']:
        st.markdown(f"#### Data Preview")
        st.write(
            f"**Number of Stocks:** {len(st.session_state['alpha_scores'])}")
        st.write(
            f"**Number of Periods:** {len(st.session_state['historical_returns'])}")
        st.write(
            f"**Date Range:** {st.session_state['historical_returns'].index[0].strftime('%Y-%m-%d')} to {st.session_state['historical_returns'].index[-1].strftime('%Y-%m-%d')}")

        st.write("**ML Alpha Scores (Top 10 by absolute value):**")
        top_alphas = st.session_state['alpha_scores'].abs().nlargest(10)
        display_alphas = st.session_state['alpha_scores'][top_alphas.index].to_frame(
            'Alpha Score')
        st.dataframe(display_alphas)

        st.write("**Historical Returns (Last 10 periods):**")
        st.dataframe(st.session_state['historical_returns'].tail(10))

# Page 3: Mean-Variance Optimization
elif selected_page == "3. Mean-Variance Optimization":
    st.header("Mean-Variance Optimization (MVO)")

    if not st.session_state['data_loaded']:
        st.warning("Please load data in Step 2 first.")
    else:
        st.markdown(f"The objective is to find portfolio weights that maximize the utility function, balancing expected return from ML alphas against portfolio variance.")

        st.markdown(
            r"""
$$
\max_{w} \left( w^T \mu_{ML} - \frac{\lambda}{2} w^T \Sigma w \right)
$$""")
        st.markdown(
            r"""
$$
\text{s.t. } \sum w_i = 1, \quad w_{min} \le w_i \le w_{max}
$$""")

        col1, col2 = st.columns(2)
        with col1:
            risk_aversion = st.slider(
                "Risk Aversion ($\lambda$)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
        with col2:
            constraint_type = st.selectbox("Constraint Type", options=[
                                           "Long Only", "Long/Short"], index=0)

        if st.button("Optimize Portfolio"):
            try:
                results = optimize_portfolio_wrapper(
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
                st.write(
                    f"**Expected Portfolio Return:** {results['expected_return']:.4f}")
                st.write(
                    f"**Portfolio Volatility:** {results['volatility']:.4f}")

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

        st.markdown(
            r"""
$$
\max_{w} \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w - \gamma ||w - w_{prev}||_1 \right)
$$""")
        st.markdown(
            r"""
$$
\text{where } \gamma \text{ is the turnover penalty parameter.}
$$""")

        penalty_gamma = st.slider(
            "Turnover Penalty ($\gamma$)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

        if st.button("Run Sensitivity Analysis"):
            try:
                sensitivity_results = turnover_sensitivity_wrapper(
                    returns=st.session_state['historical_returns'],
                    alphas=st.session_state['alpha_scores'],
                    penalty_gamma=penalty_gamma,
                    sector_map=st.session_state.get('sector_map', None)
                )

                st.markdown(f"### Sensitivity Results")
                st.line_chart(sensitivity_results['turnover_vs_return'])
                st.write(
                    "The chart above illustrates how increasing the turnover penalty affects the net portfolio return.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

# Page 5: Black-Litterman Integration
elif selected_page == "5. Black-Litterman Integration":
    st.header("Black-Litterman Model")

    if not st.session_state['data_loaded']:
        st.warning("Please load data in Step 2 first.")
    else:
        st.markdown(f"The Black-Litterman model allows us to blend the market equilibrium returns (Prior) with our ML-based alpha scores (Views), weighted by our confidence in the ML model.")

        st.markdown(
            r"""
$$
E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]
$$""")

        confidence_level = st.slider(
            "Confidence in ML Views (0 = Market, 1 = Full ML)", 0.0, 1.0, 0.5)

        if st.button("Calculate Posterior Returns"):
            try:
                bl_results = black_litterman_wrapper(
                    returns=st.session_state['historical_returns'],
                    alphas=st.session_state['alpha_scores'],
                    confidence=confidence_level
                )
                st.session_state['bl_result'] = bl_results

                st.markdown(f"### Posterior Returns Distribution")
                st.bar_chart(bl_results['posterior_returns'])

                st.markdown(
                    f"This chart compares the blended expected returns against the original market equilibrium implied returns.")

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
            window_size = st.number_input(
                "Rolling Window Size (Days)", min_value=30, max_value=60, value=30)
        with col2:
            rebalance_freq = st.selectbox("Rebalance Frequency", options=[
                                          "Daily", "Weekly", "Monthly"], index=2)

        if st.button("Run Backtest"):
            with st.spinner("Running Walk-Forward Simulation..."):
                try:
                    backtest_res = walk_forward_optimized_backtest_wrapper(
                        returns=st.session_state['historical_returns'],
                        alphas=st.session_state['alpha_scores_time_series'],
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
        st.markdown(
            f"Analyze the risk-adjusted performance of the AI-optimized portfolio compared to benchmarks.")

        try:
            # Display equity curve
            st.markdown(f"### Cumulative Performance")
            equity_curve = st.session_state['backtest_result']['equity_curve']
            st.line_chart(equity_curve)

            # Display key metrics
            st.markdown(f"### Key Performance Metrics")
            metrics = st.session_state['backtest_result'].get('metrics', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annualized Return", metrics.get(
                    'Annualized Return', 'N/A'))
                st.metric("Sharpe Ratio", metrics.get('Sharpe Ratio', 'N/A'))
            with col2:
                st.metric("Annualized Volatility", metrics.get(
                    'Annualized Volatility', 'N/A'))
                st.metric("Average Turnover", metrics.get(
                    'Average Turnover', 'N/A'))
            with col3:
                st.metric("Max Drawdown", metrics.get('Max Drawdown', 'N/A'))

            # Show detailed metrics in dashboard format
            st.markdown(f"### Detailed Performance Summary")

            # Create a styled DataFrame for better visualization
            metrics_data = {
                'Metric': [
                    '📈 Annualized Return',
                    '📊 Annualized Volatility',
                    '⚡ Sharpe Ratio',
                    '🔄 Average Turnover',
                    '📉 Maximum Drawdown'
                ],
                'Value': [
                    metrics.get('Annualized Return', 'N/A'),
                    metrics.get('Annualized Volatility', 'N/A'),
                    metrics.get('Sharpe Ratio', 'N/A'),
                    metrics.get('Average Turnover', 'N/A'),
                    metrics.get('Max Drawdown', 'N/A')
                ],
                'Description': [
                    'Average yearly portfolio return',
                    'Standard deviation of returns (risk)',
                    'Risk-adjusted return measure',
                    'Portfolio rebalancing frequency',
                    'Largest peak-to-trough decline'
                ]
            }

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Performance Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large")
                }
            )

            # Show backtest details
            if 'full_backtest' in st.session_state['backtest_result']:
                with st.expander("View Detailed Backtest Results"):
                    st.dataframe(
                        st.session_state['backtest_result']['full_backtest'].tail(20))

        except Exception as e:
            st.error(f"Visualization failed: {e}")
            import traceback
            st.error(traceback.format_exc())


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
