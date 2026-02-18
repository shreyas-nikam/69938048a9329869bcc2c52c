
# Streamlit Application Specification for AI-Optimized Portfolio Construction

## 1. Application Overview

### Purpose of the Application
The application serves to simulate the process of integrating machine learning (ML) insights into portfolio construction. It enables users, primarily CFA Charterholders and Investment Professionals, to apply ML alpha scores with real-world constraints, providing a hands-on experience in constructing and evaluating AI-optimized portfolios.

### High-Level Story Flow
1. **Input Data Loading:** Users load ML alpha scores and historical stock returns.
2. **Mean-Variance Optimization (MVO):** Apply ML returns in an MVO framework with constraints.
3. **Turnover Penalization:** Assess the impact of transaction costs using a turnover penalty.
4. **Black-Litterman Integration:** Combine ML views with market equilibrium.
5. **Walk-Forward Backtest:** Simulate real-world portfolio management over time.
6. **Performance Evaluation:** Visualize efficient frontier, performance metrics, and portfolio attribution.

## 2. Code Requirements

### Import Statement
```python
from source import *
```

### UI Interactions and Function Calls
- **Data Loading**: Upload CSV files for ML alpha scores and historical returns.
  - Calls: `prepare_optimization_inputs()`
- **MVO Execution**: Select constraints and optimization parameters.
  - Calls: `optimize_portfolio()`
- **Turnover Sensitivity Analysis**: Adjust turnover penalty parameter.
  - Calls: `turnover_sensitivity()`
- **Black-Litterman Exploration**: Set view confidence and visualize blending.
  - Calls: `black_litterman()`
- **Backtest Simulation**: Run walk-forward analysis.
  - Calls: `walk_forward_optimized_backtest()`
- **Results Visualization**: Efficient frontier and attribution breakdown.
  - Calls: `plot_efficient_frontier()`, `plot_performance_attribution()`

### Session State Management
- **Initialization**: Check if certain keys exist, else set defaults.
  ```python
  if 'ml_alpha_scores' not in st.session_state:
      st.session_state['ml_alpha_scores'] = None
  ```
- **Update**: Modify the session state when user inputs change.
  ```python
  st.session_state['ml_alpha_scores'] = uploaded_file
  ```
- **Read Across Pages**: Use stored session data to maintain continuity in workflows.
  ```python
  if st.session_state['ml_alpha_scores'] is not None:
      ...
  ```

### Markdown Blocks
- **Application Purpose**: Explain the real-world context of each module.
- **Mathematical Formulations**:
  ```python
  st.markdown(r"$$ \max_{w} \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w - \gamma ||w - w_{prev}||_1 \right) $$")
  st.markdown(r"where $w$ are portfolio weights, $\mu_{ML}$ are ML alpha scores, $\Sigma$ is the covariance matrix ...")
  ```
- **Visual Analyses**:
  ```python
  st.markdown(f"### Comparison of Naive vs. Constrained MVO")
  st.markdown(f"### Efficient Frontier Analysis")
  ```

## Input Context for Development
- The application utilizes Python libraries such as `cvxpy`, `numpy`, `pandas`, `scipy`, and `matplotlib` for core functionalities like optimization and visualization.
- The Streamlit app must provide an interactive experience where users can adjust parameters, visualize results, and perform analysis akin to real-world workflows.

## Visualization Requirements
- **Histograms**: Display weight distribution.
- **Sensitivity Table**: Show impact of different turnover penalties.
- **Bar Charts**: Black-Litterman blend visualizations.
- **Line Plots**: Efficient frontier, walk-forward equity curves.
- **Attribution Visualization**: Demonstrate breakdown of portfolio returns.

## Output Specifications
- **Developer Notes**: Organize the code structure for readability.
- **User Guidance**: Clearly label widgets and inform about their functionality.
- **Handling Errors**: Log warnings and alert users when errors occur.

This specification provides a comprehensive blueprint for building the `app.py` file, detailing both technical and user-experience aspects necessary to create a robust Streamlit application.
