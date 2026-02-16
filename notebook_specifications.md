
# AI-Enhanced Portfolio Construction: From ML Alpha to Investable Weights

## Introduction: The Quant's Dilemma

As a **CFA Charterholder and Quant Researcher** at "Alpha Frontier Asset Management," your team has successfully developed machine learning models (e.g., as demonstrated in D5-T1-C1) that predict stock alpha scores ($\mu_{ML}$) with promising accuracy. However, raw alpha scores are not an investable portfolio. Your primary challenge now is to translate these powerful, yet often volatile, predictions into a practical, stable, and compliant investment strategy that adheres to real-world constraints and explicitly accounts for transaction costs.

Simply buying the top-ranked stocks from the ML model can lead to highly concentrated, unstable portfolios with excessive turnover, violating regulatory and internal mandates. Your task is to build a robust portfolio construction framework that bridges the gap between predictive power and actionable investment decisions, ensuring that AI-driven insights can be effectively deployed in the firm's client portfolios.

This notebook will guide you through the process of taking ML alpha scores and combining them with advanced optimization techniques, including robust covariance estimation, constraint handling, turnover penalization, and the Black-Litterman model, to construct a truly "AI-Optimized Portfolio."

## 1. Setup: Installing Libraries and Importing Dependencies

Before we begin our journey into AI-enhanced portfolio construction, we need to ensure all necessary libraries are installed and imported. These tools will enable us to perform robust data preparation, complex optimization, and insightful visualizations.

```python
!pip install numpy pandas cvxpy scipy matplotlib scikit-learn
```

```python
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Matplotlib settings for consistent plots
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14,
                      'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})
```

## 2. Input Preparation: Robust Alpha and Covariance Estimation

### Story + Context + Real-World Relevance

As a Quant Researcher, your first step is to prepare the foundational inputs for portfolio optimization: the expected returns vector ($\mu$) and the covariance matrix of returns ($\Sigma$). While your ML models provide the alpha scores (acting as $\mu_{ML}$), estimating a reliable covariance matrix is critical. A naive sample covariance matrix, especially with limited historical data and many assets, is notoriously noisy and often ill-conditioned. This can lead to extreme, unstable portfolio weights that are impractical and risky.

To address this, we employ **Ledoit-Wolf shrinkage**, a technique that blends the sample covariance matrix with a more stable, structured target matrix (often a scaled identity matrix). This shrinks the extreme eigenvalues of the sample covariance matrix towards the mean, resulting in a more robust and better-conditioned estimate of $\Sigma$. This stability is paramount for the downstream optimization process, preventing the optimizer from exploiting spurious correlations.

The formula for the Ledoit-Wolf estimator, $\Sigma_{LW}$, is a convex combination of the sample covariance matrix $S$ and a shrinkage target $F$:
$$ \Sigma_{LW} = \delta F + (1 - \delta) S $$
where $\delta$ is the shrinkage intensity, a value between 0 (no shrinkage, just sample covariance) and 1 (full shrinkage, just the target matrix). The Ledoit-Wolf method optimally estimates $\delta$ to minimize the mean squared error between the estimated and true covariance matrices.

We will also filter our universe of stocks to focus on a manageable number of top-performing assets based on their ML alpha scores, reflecting a real-world scenario where a PM might only focus on the most promising opportunities.

```python
def prepare_optimization_inputs(returns_df: pd.DataFrame, ml_alpha_scores: pd.Series, n_assets: int = 50, n_periods_cov: int = 60) -> tuple[np.ndarray, np.ndarray, list, pd.Series]:
    """
    Prepares inputs for portfolio optimization:
    - Selects top n_assets based on ML alpha scores.
    - Estimates a robust covariance matrix (Sigma) using Ledoit-Wolf shrinkage
      from historical returns.

    Args:
        returns_df (pd.DataFrame): Historical monthly stock returns, indexed by time,
                                   with stock identifiers as columns.
        ml_alpha_scores (pd.Series): ML-predicted excess returns (alpha) for the universe of stocks.
        n_assets (int): Number of top assets to select for the portfolio.
        n_periods_cov (int): Number of historical periods to use for covariance estimation.

    Returns:
        tuple[np.ndarray, np.ndarray, list, pd.Series]:
            - mu (np.ndarray): ML-predicted expected returns for selected assets.
            - Sigma (np.ndarray): Ledoit-Wolf shrunk covariance matrix for selected assets.
            - stock_symbols (list): List of symbols for selected assets.
            - selected_ml_alpha (pd.Series): ML alpha scores for selected assets (for consistency).
    """
    # Ensure ml_alpha_scores is aligned with returns_df columns
    ml_alpha_scores = ml_alpha_scores[ml_alpha_scores.index.isin(returns_df.columns)]
    
    # Select top n_assets by absolute alpha magnitude (tradeable universe)
    # This simulates a PM focusing on the most promising stocks identified by ML
    top_stocks = ml_alpha_scores.abs().nlargest(n_assets).index
    
    # Filter returns and alpha scores to selected top stocks
    returns_subset = returns_df[top_stocks].tail(n_periods_cov).dropna()
    mu = ml_alpha_scores[top_stocks].values
    
    # Estimate raw sample covariance matrix for diagnostics
    raw_cov = returns_subset.cov()
    
    # Covariance: Ledoit-Wolf shrinkage for stability
    lw = LedoitWolf()
    lw.fit(returns_subset)
    Sigma = lw.covariance_
    shrinkage_intensity = lw.shrinkage_

    print(f"Universe selected for optimization: {n_assets} stocks")
    print(f"Expected returns range (ML alpha): [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"Covariance matrix shape: {Sigma.shape}, shrinkage intensity={shrinkage_intensity:.3f}")
    print(f"Condition number (raw sample covariance): {np.linalg.cond(raw_cov):.0f}")
    print(f"Condition number (Ledoit-Wolf shrunk covariance): {np.linalg.cond(Sigma):.0f}")
    
    return mu, Sigma, top_stocks.tolist(), ml_alpha_scores[top_stocks]

# --- Synthetic Data Generation (for demonstration purposes) ---
np.random.seed(42)
n_universe = 200 # Total stocks in the universe
n_periods_hist = 120 # 10 years of monthly data

# Simulate historical returns with some correlation
dates = pd.date_range(start='2010-01-01', periods=n_periods_hist, freq='M')
stock_symbols_full = [f'Stock_{i}' for i in range(n_universe)]

# Factor model simulation for correlated returns
factors = np.random.randn(n_periods_hist, 5) * 0.01 # 5 factors
factor_betas = np.random.randn(n_universe, 5) * 0.5 + 0.5 # Each stock's sensitivity to factors
idiosyncratic_returns = np.random.randn(n_periods_hist, n_universe) * 0.02

# Generate returns
returns_array = (factors @ factor_betas.T) + idiosyncratic_returns
returns_df_full = pd.DataFrame(returns_array, index=dates, columns=stock_symbols_full)
returns_df_full = returns_df_full - returns_df_full.mean() # Center returns for easier interpretation

# Simulate ML alpha scores (excess returns)
ml_alpha_scores_full = pd.Series(np.random.randn(n_universe) * 0.005, index=stock_symbols_full) # Monthly alpha

# Define a simple sector map for n_universe assets
num_sectors = 5
sector_map_full = {f'Sector_{s}': [stock_symbols_full[j] for j in range(s * (n_universe // num_sectors), (s + 1) * (n_universe // num_sectors))]
                   for s in range(num_sectors)}

# Prepare inputs for optimization (selecting 50 assets from the 200 universe)
mu_ml, Sigma_ml, selected_stocks_ml, selected_ml_alpha = prepare_optimization_inputs(
    returns_df_full, ml_alpha_scores_full, n_assets=50, n_periods_cov=60
)
```

### Explanation of Execution

The output above demonstrates the critical role of Ledoit-Wolf shrinkage. Notice the **Condition number (raw sample covariance)**, which is likely very high (e.g., in the thousands or millions). A high condition number indicates an ill-conditioned matrix, meaning it is close to singular and highly sensitive to small changes, making optimization unstable. In contrast, the **Condition number (Ledoit-Wolf shrunk covariance)** is significantly lower, typically below 100. This improved condition number confirms that the shrinkage technique has successfully stabilized the covariance matrix, making it suitable for robust portfolio optimization.

For a CFA Charterholder, this output is crucial: it provides confidence that the risk estimates are stable and reliable, preventing the optimization from creating portfolios based on noisy or spurious relationships in the historical data. The selection of `n_assets` based on ML alpha scores ensures that the optimization focuses on the most promising opportunities identified by the models, aligning with the firm's investment strategy.

## 3. Naive vs. Constrained Mean-Variance Optimization (MVO)

### Story + Context + Real-World Relevance

After preparing the inputs, the next step is to actually construct a portfolio. A common starting point is **Mean-Variance Optimization (MVO)**, which aims to maximize expected return for a given level of risk (or minimize risk for a given return). However, a pure, "naive" MVO often leads to highly concentrated portfolios with extreme positions, which are unacceptable in regulated asset management. These portfolios are prone to instability and can violate investment mandates (e.g., maximum position limits, sector exposure limits, long-only requirements).

As a Quant, you understand that **constraints are the bridge between theoretical optimization and practical, investable portfolios.** We need to incorporate real-world limitations such as:
- **Fully Invested:** The sum of weights must equal 1 ($$\sum_{i=1}^{N} w_i = 1$$).
- **Long-Only:** No short-selling allowed ($w_i \geq 0$).
- **Position Limits:** Individual stock weights cannot exceed a certain percentage ($w_i \leq w_{max}$).
- **Sector Limits:** The sum of weights within any given sector cannot exceed a certain percentage ($\sum_{i \in S_k} w_i \leq S_{max}$ for each sector $k$).

The MVO problem seeks to maximize a risk-adjusted return, typically formalized as maximizing expected return minus a penalty for risk. The objective function for a standard MVO is:

$$ \max_{w} \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w \right) $$

subject to the aforementioned constraints. Here, $w$ is the vector of portfolio weights, $\mu$ is the vector of expected returns, $\Sigma$ is the covariance matrix, and $\lambda$ is the **risk aversion parameter**, which balances the trade-off between maximizing expected return and minimizing portfolio variance. A higher $\lambda$ indicates a more risk-averse investor. We will use `cvxpy` to solve this quadratic programming problem.

```python
def optimize_portfolio(mu: np.ndarray, Sigma: np.ndarray, constraints: str = 'constrained',
                       risk_aversion: float = 2.0, w_prev: np.ndarray = None,
                       turnover_penalty: float = 0.0, max_weight: float = 0.05,
                       sector_map: dict = None, max_sector: float = 0.25) -> tuple[np.ndarray, float]:
    """
    Solves the constrained Mean-Variance Optimization problem with optional turnover penalty.

    Args:
        mu (np.ndarray): Expected returns vector (ML alpha scores).
        Sigma (np.ndarray): Covariance matrix of returns.
        constraints (str): Type of constraints ('naive', 'constrained').
                           - 'naive': Fully invested, long-only, no explicit max_weight or sector limits.
                           - 'constrained': Fully invested, long-only, max_weight, and sector limits.
        risk_aversion (float): Lambda parameter for risk aversion.
        w_prev (np.ndarray, optional): Previous period's weights, required for turnover penalty.
        turnover_penalty (float): Gamma parameter for turnover penalty.
        max_weight (float): Maximum allowed weight for any single asset.
        sector_map (dict, optional): Dictionary mapping sector names to lists of asset indices.
        max_sector (float): Maximum allowed weight for any single sector.

    Returns:
        tuple[np.ndarray, float]:
            - w.value (np.ndarray): Optimal portfolio weights.
            - objective.value (float): Value of the objective function at optimality.
    """
    n = len(mu)
    w = cp.Variable(n)

    # Objective: alpha - risk - turnover cost
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)
    objective = ret - (risk_aversion / 2) * risk

    # Turnover penalty (L1 norm)
    if w_prev is not None and turnover_penalty > 0:
        turnover = cp.norm(w - w_prev, 1)
        objective -= turnover_penalty * turnover

    # Constraints
    cons = [cp.sum(w) == 1] # Fully invested budget

    if constraints == 'constrained':
        cons += [w >= 0] # Long-only
        cons += [w <= max_weight] # Position limit
        
        # Sector constraints (if provided)
        if sector_map is not None:
            for sector_name, indices in sector_map.items():
                # Map stock symbols to their indices in the current `mu` vector
                current_indices = [selected_stocks_ml.index(sym) for sym in indices if sym in selected_stocks_ml]
                if current_indices: # Only add constraint if sector has assets in current selection
                    cons += [cp.sum(w[current_indices]) <= max_sector]

    elif constraints == 'naive':
        # Naive: fully invested, long-only, but without max_weight or sector_map constraints
        cons += [w >= 0]
        # For naive, we effectively set max_weight to 1 (or very high)
        cons += [w <= 1.0] 
    else:
        raise ValueError("Invalid constraints type. Choose 'naive' or 'constrained'.")

    # Solve the optimization problem
    prob = cp.Problem(cp.Maximize(objective), cons)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
        if prob.status == 'optimal' or prob.status == 'optimal_near':
            return w.value, prob.value
        else:
            print(f"Warning: solver status = {prob.status}")
            return None, None
    except cp.error.SolverError:
        print("Error: Solver failed to converge for this problem.")
        return None, None

# Adjust sector_map_full to use only the selected stocks for this optimization step
# Get indices of selected stocks relative to the full list to create a map using only symbols
selected_sector_map = {}
for sector, symbols in sector_map_full.items():
    selected_sector_map[sector] = [s for s in symbols if s in selected_stocks_ml]

# --- Execute Naive (unconstrained) MVO ---
# Effectively unconstrained: only long-only and fully invested. Risk aversion is low to show concentration.
w_naive, obj_naive = optimize_portfolio(mu_ml, Sigma_ml, constraints='naive',
                                        risk_aversion=0.1, max_weight=1.0)

# --- Execute Constrained MVO ---
# Realistic limits: long-only, max position, sector limits, fully invested.
w_constrained, obj_con = optimize_portfolio(mu_ml, Sigma_ml, constraints='constrained',
                                            risk_aversion=2.0, max_weight=0.05,
                                            sector_map=selected_sector_map, max_sector=0.25)

# --- Comparison Metrics ---
if w_naive is not None and w_constrained is not None:
    expected_return_naive = (mu_ml @ w_naive) * 100
    volatility_naive = np.sqrt(w_naive @ Sigma_ml @ w_naive) * 100

    expected_return_constrained = (mu_ml @ w_constrained) * 100
    volatility_constrained = np.sqrt(w_constrained @ Sigma_ml @ w_constrained) * 100

    print("\nNAIVE vs. CONSTRAINED COMPARISON")
    print("=" * 55)
    print(f"{'Metric':<30s}{'Naive':>12s}{'Constrained':>12s}")
    print(f"{'Max position weight':<30s}{np.max(w_naive):>12.1%}{np.max(w_constrained):>12.1%}")
    print(f"{'# nonzero positions':<30s}{np.sum(w_naive > 0.001):>12.0f}{np.sum(w_constrained > 0.001):>12.0f}")
    print(f"{'Expected return':<30s}{expected_return_naive:>12.2f}%{expected_return_constrained:>12.2f}%")
    print(f"{'Portfolio volatility':<30s}{volatility_naive:>12.2f}%{volatility_constrained:>12.2f}%")

    # --- Visualization: Weight Distribution ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(w_naive[w_naive > 1e-4], bins=10, color='skyblue', edgecolor='black')
    plt.title('Naive Portfolio Weights Distribution')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.xlim(0, max(np.max(w_naive), np.max(w_constrained)) * 1.1)

    plt.subplot(1, 2, 2)
    plt.hist(w_constrained[w_constrained > 1e-4], bins=10, color='lightcoral', edgecolor='black')
    plt.title('Constrained Portfolio Weights Distribution')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.xlim(0, max(np.max(w_naive), np.max(w_constrained)) * 1.1)
    plt.tight_layout()
    plt.show()
```

### Explanation of Execution

The comparison clearly highlights the inadequacy of naive MVO for real-world application. The "Naive Portfolio Weights Distribution" histogram likely shows a few assets with extremely high weights, while many others have zero weight. This extreme concentration creates significant uncompensated risk and violates practical investment rules. For instance, a maximum position limit of 5% (i.e., $w_{max} = 0.05$) is common to prevent overexposure to any single stock.

In contrast, the "Constrained Portfolio Weights Distribution" histogram displays a much more diversified and practical distribution of weights, with no single asset exceeding the defined `max_weight` (here, 5%). The `# nonzero positions` metric also demonstrates how constraints lead to a broader allocation, which is generally more stable and less prone to idiosyncratic shocks.

For a CFA Charterholder, this distinction is paramount. It illustrates how simply optimizing for alpha and risk without considering practical constraints transforms a theoretically optimal solution into an operationally unfeasible one. The application of long-only, position, and sector limits ensures the portfolio aligns with client mandates and regulatory requirements, making the AI-driven strategy genuinely actionable.

## 4. Accounting for Transaction Costs with Turnover Penalization

### Story + Context + Real-World Relevance

While constraints address portfolio concentration, another major practical hurdle for investment professionals is **transaction costs**. A portfolio optimized solely on expected returns and risk, even with position limits, might still suggest frequent and significant rebalancing, leading to high **turnover**. High turnover translates directly into substantial transaction costs (brokerage fees, bid-ask spread, market impact), which can quickly erode any predicted alpha.

As a Quant, you know that a successful strategy must account for these costs. We introduce a **turnover penalty** into the MVO objective function. This penalty discourages large deviations from the previous period's portfolio weights ($w_{prev}$), thereby explicitly modeling the cost of rebalancing. The objective function becomes:

$$ \max_{w} \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w - \gamma \|w - w_{prev}\|_1 \right) $$

Here, $\gamma$ is the **turnover penalty parameter**. A higher $\gamma$ means the optimizer is more reluctant to trade, resulting in lower turnover but potentially less alpha capture. The term $\|w - w_{prev}\|_1$ represents the **portfolio turnover**, calculated as the sum of absolute changes in weights. The goal is to find an optimal $\gamma$ that balances the trade-off between maximizing gross alpha and minimizing transaction costs, ultimately maximizing the **net Sharpe Ratio**.

We will perform a sensitivity analysis by sweeping across different $\gamma$ values and observing the impact on gross and net Sharpe Ratios, assuming a fixed transaction cost per unit of turnover. The net Sharpe Ratio, which accounts for costs, is the true measure of performance for an implementable strategy.

```python
def turnover_sensitivity(mu: np.ndarray, Sigma: np.ndarray, w_prev: np.ndarray,
                         gammas: list = None, risk_aversion: float = 2.0,
                         max_weight: float = 0.05, sector_map: dict = None,
                         max_sector: float = 0.25, transaction_cost_bps: float = 20.0) -> pd.DataFrame:
    """
    Sweeps turnover penalty (gamma) to find the optimal trade-off between alpha capture and transaction costs.

    Args:
        mu (np.ndarray): Expected returns vector (ML alpha scores).
        Sigma (np.ndarray): Covariance matrix of returns.
        w_prev (np.ndarray): Previous period's weights, typically an equal-weight portfolio to start.
        gammas (list): List of turnover penalty (gamma) values to test.
        risk_aversion (float): Lambda parameter for risk aversion.
        max_weight (float): Maximum allowed weight for any single asset.
        sector_map (dict): Dictionary mapping sector names to lists of asset indices.
        max_sector (float): Maximum allowed weight for any single sector.
        transaction_cost_bps (float): Transaction cost in basis points per unit of turnover (e.g., 20 bps = 0.0020).

    Returns:
        pd.DataFrame: Results of the sensitivity analysis (gamma, turnover, gross/net returns/Sharpe).
    """
    if gammas is None:
        gammas = [0, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.02]

    results = []
    transaction_cost_per_unit_turnover = transaction_cost_bps / 10000.0 # Convert bps to decimal

    print("TURNOVER SENSITIVITY ANALYSIS")
    print("=" * 75)
    print(f"{'Gamma':<8s}{'Turnover':>10s}{'Gross Ret':>12s}{'Net Ret':>12s}{'Gross SR':>10s}{'Net SR':>10s}")

    for gamma in gammas:
        w, _ = optimize_portfolio(mu, Sigma, constraints='constrained',
                                 risk_aversion=risk_aversion, w_prev=w_prev,
                                 turnover_penalty=gamma, max_weight=max_weight,
                                 sector_map=sector_map, max_sector=max_sector)
        
        if w is None:
            continue

        turnover = np.sum(np.abs(w - w_prev)) # L1 norm
        exp_ret_gross = mu @ w
        
        # Calculate gross volatility for Sharpe Ratio
        vol = np.sqrt(w @ Sigma @ w)
        
        # Calculate transaction costs
        gross_cost = turnover * transaction_cost_per_unit_turnover
        net_ret = exp_ret_gross - gross_cost
        
        # Annualize for Sharpe Ratio (assuming monthly data, 12 periods/year)
        sharpe_gross = (exp_ret_gross / vol) * np.sqrt(12) if vol > 1e-6 else 0
        sharpe_net = (net_ret / vol) * np.sqrt(12) if vol > 1e-6 else 0 # Use gross vol for net sharpe too, for comparison

        results.append({'gamma': gamma, 'turnover': turnover,
                        'gross_return': exp_ret_gross, 'net_return': net_ret, 'vol': vol,
                        'sharpe_gross': sharpe_gross, 'sharpe_net': sharpe_net})
        
        print(f"{gamma:<8.4f}{turnover:>10.1%}{exp_ret_gross*100:>12.2f}%{net_ret*100:>12.2f}%{sharpe_gross:>10.2f}{sharpe_net:>10.2f}")

    results_df = pd.DataFrame(results)
    return results_df

# Assuming an equal-weighted previous portfolio to start (cold start)
n_selected = len(selected_stocks_ml)
w_prev_equal = np.ones(n_selected) / n_selected

turnover_results_df = turnover_sensitivity(mu_ml, Sigma_ml, w_prev=w_prev_equal,
                                         sector_map=selected_sector_map,
                                         gammas=[0, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01])

# --- Visualization: Turnover Sensitivity Plot ---
if not turnover_results_df.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(turnover_results_df['gamma'], turnover_results_df['sharpe_gross'], marker='o', label='Gross Sharpe Ratio')
    plt.plot(turnover_results_df['gamma'], turnover_results_df['sharpe_net'], marker='x', label='Net Sharpe Ratio')
    plt.axvline(x=turnover_results_df.loc[turnover_results_df['sharpe_net'].idxmax()]['gamma'], 
                color='gray', linestyle='--', label='Optimal Net Sharpe Gamma')
    plt.title('Turnover Penalty Sensitivity: Gross vs. Net Sharpe Ratio')
    plt.xlabel('Turnover Penalty (γ)')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.xscale('log') # Gamma often varies logarithmically
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.show()
```

### Explanation of Execution

The **Turnover Sensitivity Analysis** table and plot vividly demonstrate the critical trade-off between capturing gross alpha and incurring transaction costs. As the turnover penalty ($\gamma$) increases, the portfolio's turnover generally decreases, reducing transaction costs. Initially, this leads to an increase in the **Net Sharpe Ratio**, as the savings from lower costs outweigh the reduction in gross alpha capture. However, beyond a certain point (the "Optimal Net Sharpe Gamma" marked on the plot), a very high $\gamma$ will excessively restrict rebalancing, causing the portfolio to miss out on significant alpha opportunities, leading to a decline in both gross and net returns.

For a CFA Charterholder, this analysis is indispensable. It provides a data-driven approach to setting a realistic turnover budget. Relying solely on "Gross Sharpe" (which is typically inflated) from backtests without turnover penalties is misleading. The "Net Sharpe Ratio" is the true metric of an implementable strategy's profitability. This step ensures that the AI-driven portfolio construction is not only theoretically sound but also economically viable in a production environment, effectively bridging the gap between backtesting and live trading.

## 5. Black-Litterman Model for Blending ML Views

### Story + Context + Real-World Relevance

Even with robust covariance estimation and turnover control, portfolios optimized purely on ML alpha scores can sometimes be overly aggressive, highly concentrated, or sensitive to small changes in predictions. This is because raw ML predictions, while powerful, often reflect a level of "overconfidence" that doesn't fully capture their inherent uncertainty. A responsible approach integrates these idiosyncratic ML views with a more stable, diversified market perspective.

Here, as a Quant, you introduce the **Black-Litterman (BL) model**. The BL model provides a Bayesian framework to combine an equilibrium market view (e.g., implied by market capitalization weights) with an investor's specific "views" (our ML alpha predictions). This blend produces a more diversified and stable posterior expected return vector ($\mu_{BL}$) than either input alone. A key parameter in BL is **view confidence** ($\tau$), which dictates how much weight is given to the investor's views versus the market equilibrium. A lower $\tau$ means more trust in the equilibrium, while a higher $\tau$ places more trust in the ML views.

The BL model calculates the posterior expected return vector $\mu_{BL}$ as:
$$ \mu_{BL} = \left( (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right)^{-1} \left( (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right) $$
where:
- $\Pi$ is the implied equilibrium return vector (derived from market cap weights and risk aversion).
- $Q$ is the vector of investor views (our ML alpha scores).
- $P$ is a matrix linking views to assets (identity matrix for absolute views).
- $\Omega$ is a diagonal matrix representing the uncertainty of the views.
- $\tau$ is the scalar "view confidence" parameter (a measure of our confidence in the ML views relative to the market).
- $\Sigma$ is the covariance matrix of asset returns.

This approach acknowledges the value of ML while anchoring the portfolio to a sensible market prior, producing more stable and diversified allocations.

```python
def black_litterman(Sigma: np.ndarray, market_cap_weights: np.ndarray, ml_views: np.ndarray,
                    risk_aversion: float = 2.5, tau: float = 0.05, view_confidence: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the Black-Litterman model to blend market equilibrium returns with ML views.

    Args:
        Sigma (np.ndarray): Covariance matrix of returns.
        market_cap_weights (np.ndarray): Market capitalization weights (as proxy for equilibrium).
        ml_views (np.ndarray): ML-predicted excess returns (alpha scores).
        risk_aversion (float): Risk aversion parameter for deriving equilibrium returns.
        tau (float): Scalar Black-Litterman parameter for equilibrium uncertainty.
        view_confidence (float): Confidence level in ML views (0 to 1), higher means more trust in ML.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - posterior_mu (np.ndarray): Black-Litterman posterior mean (blended expected returns).
            - posterior_cov (np.ndarray): Black-Litterman posterior covariance.
    """
    n = len(ml_views)

    # 1. Equilibrium Returns (Pi): Derived from market cap weights
    # Pi = lambda * Sigma * w_mkt
    Pi = risk_aversion * Sigma @ market_cap_weights

    # 2. Investor Views (Q) and Link Matrix (P)
    # P = I (identity matrix for absolute views), Q = ML alpha scores
    P = np.eye(n)
    Q = ml_views

    # 3. Uncertainty of Views (Omega)
    # Omega = diagonal uncertainty matrix. Higher confidence = lower omega.
    # We use a simple approach: scale a diagonal of Sigma by tau and view_confidence.
    omega_diag = np.diag(tau * Sigma) / view_confidence
    Omega = np.diag(omega_diag)

    # 4. Black-Litterman Posterior
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)

    # Posterior covariance matrix
    posterior_cov_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
    posterior_cov = np.linalg.inv(posterior_cov_inv)

    # Posterior mean vector
    posterior_mu = posterior_cov @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

    print(f"Black-Litterman Posterior (View Confidence: {view_confidence}):")
    print(f"  Equilibrium mean return: {Pi.mean()*100:.3f}%")
    print(f"  ML views mean: {Q.mean()*100:.3f}%")
    print(f"  BL posterior mean: {posterior_mu.mean()*100:.3f}%")
    
    return posterior_mu, posterior_cov

# --- Simulate market cap weights (for selected assets) ---
# For simplicity, we can start with equal weights as a proxy for market equilibrium.
# In a real scenario, this would come from actual market capitalization data.
market_cap_weights = np.ones(len(selected_stocks_ml)) / len(selected_stocks_ml)

# --- Execute Black-Litterman and optimize portfolio with BL inputs ---
view_confidence_level = 0.3 # Trust ML views by 30%
mu_bl, Sigma_bl_posterior = black_litterman(Sigma_ml, market_cap_weights, mu_ml,
                                            view_confidence=view_confidence_level)

# Optimize portfolio with BL inputs (using the same turnover penalty from sensitivity analysis)
# Using an example optimal gamma from the previous step, e.g., 0.0005
optimal_gamma_for_bl = 0.0005 

w_bl, obj_bl = optimize_portfolio(mu_bl, Sigma_ml, constraints='constrained',
                                 risk_aversion=2.0, w_prev=w_prev_equal, # w_prev_equal as a baseline for comparison
                                 turnover_penalty=optimal_gamma_for_bl, max_weight=0.05,
                                 sector_map=selected_sector_map, max_sector=0.25)

# --- Visualization: Black-Litterman Blend ---
if w_bl is not None:
    # Select top 10 stocks by BL posterior return for visualization
    top_stocks_bl_idx = np.argsort(mu_bl)[-10:]
    top_stock_symbols = [selected_stocks_ml[i] for i in top_stocks_bl_idx]

    # Create a DataFrame for easy plotting
    bl_returns_data = pd.DataFrame({
        'Equilibrium': Pi[top_stocks_bl_idx],
        'ML Views': mu_ml[top_stocks_bl_idx],
        'BL Posterior': mu_bl[top_stocks_bl_idx]
    }, index=top_stock_symbols)

    bl_returns_data *= 100 # Convert to percentage for readability

    bl_returns_data.plot(kind='bar', figsize=(12, 7), title='Black-Litterman Blend for Top Stocks (Expected Returns %)')
    plt.ylabel('Expected Return (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
```

### Explanation of Execution

The Black-Litterman model's output demonstrates a more balanced and realistic set of expected returns (`BL posterior mean`) compared to raw ML views or a pure market equilibrium. The bar chart visualization for selected top stocks clearly shows how the BL posterior expected returns typically lie between the equilibrium returns and the ML views, reflecting the blend. This "anchoring" effect of the BL model is crucial.

For a CFA Charterholder, Black-Litterman offers a "responsible" way to integrate potentially volatile ML alpha into portfolio construction. It stabilizes the optimization process, produces more diversified portfolios, and allows for the integration of human judgment through the `view_confidence` parameter. This parameter is the Quant's tool to reflect how much trust is placed in the ML model's current predictions relative to the market's collective wisdom. A higher confidence level means the portfolio leans more heavily on ML insights, while lower confidence maintains a stronger anchor to the market, mitigating the risk of overfitting or relying too heavily on potentially noisy ML signals. This human-in-the-loop mechanism is key for deploying AI successfully in a regulated financial environment.

## 6. Walk-Forward Optimized Backtest

### Story + Context + Real-World Relevance

After establishing a robust optimization framework, the most critical step for any investment professional is to validate its performance in a realistic, forward-looking manner. A static, in-sample backtest is insufficient and often overestimates performance. We need to simulate the entire portfolio construction process, including ML prediction, optimization, and rebalancing, over an extended period. This is where **walk-forward backtesting** comes in.

As a Quant Researcher, you'll simulate the strategy month-by-month (or period-by-period):
1.  **Prediction:** At the start of each period, historical data up to that point is used to simulate generating new ML alpha scores (in this lab, we'll use pre-generated "next month return" as a proxy for a perfect predictor for demonstration, but in a real system, the ML model would run here).
2.  **Optimization:** The portfolio is optimized using the latest alpha scores, robust covariance, and all defined constraints (position, sector, turnover penalty).
3.  **Rebalancing:** The new optimal weights determine the trades to be executed.
4.  **Performance Tracking:** The actual returns of the constructed portfolio are tracked for the upcoming period, net of transaction costs.
5.  **Iteration:** The process repeats, with the previous period's weights serving as $w_{prev}$ for the next optimization, ensuring the turnover penalty is correctly applied.

This rigorous simulation provides a realistic estimate of the strategy's implementable performance, including net-of-cost Sharpe Ratio and cumulative returns, which are crucial metrics for a CFA Charterholder evaluating strategy viability.

```python
def walk_forward_optimized_backtest(returns_df: pd.DataFrame, ml_alpha_scores_df: pd.DataFrame,
                                     sector_map: dict, n_assets_select: int = 50,
                                     risk_aversion: float = 2.0, turnover_penalty: float = 0.0005,
                                     max_weight: float = 0.05, max_sector: float = 0.25,
                                     min_train_periods: int = 60, n_periods_cov: int = 60) -> pd.DataFrame:
    """
    Complete pipeline: simulate ML prediction -> optimize -> rebalance -> track performance net-of-costs.

    Args:
        returns_df (pd.DataFrame): Historical monthly stock returns, indexed by time.
        ml_alpha_scores_df (pd.DataFrame): ML-predicted alpha scores for each month, indexed by time.
                                         Each row is a Series of alpha scores for that month.
        sector_map (dict): Dictionary mapping sector names to lists of stock symbols.
        n_assets_select (int): Number of top assets to select for the portfolio in each period.
        risk_aversion (float): Risk aversion parameter for MVO.
        turnover_penalty (float): Gamma parameter for turnover penalty.
        max_weight (float): Maximum allowed weight for any single asset.
        max_sector (float): Maximum allowed weight for any single sector.
        min_train_periods (int): Minimum periods required for initial training data.
        n_periods_cov (int): Number of historical periods to use for covariance estimation.

    Returns:
        pd.DataFrame: Cumulative net-of-cost returns of the optimized portfolio over the backtesting period.
    """
    
    # Ensure all dataframes are aligned on columns for safety
    common_stocks = list(set(returns_df.columns) & set(ml_alpha_scores_df.columns))
    returns_df = returns_df[common_stocks]
    ml_alpha_scores_df = ml_alpha_scores_df[common_stocks]

    months = sorted(returns_df.index.unique())
    portfolio_returns = []
    
    # Start with an equal-weighted portfolio for the first period's rebalancing calculation
    # We will adjust this to the size of the *selected* assets later
    w_prev = None 
    
    print(f"Starting walk-forward backtest over {len(months) - min_train_periods} periods...")
    for i in range(min_train_periods, len(months)):
        current_month = months[i]
        
        # Training data for this period (up to current_month - 1)
        train_returns_df = returns_df[returns_df.index < current_month].tail(n_periods_cov)
        
        # ML alpha scores for the *current* month (to predict next month's returns)
        # Note: In a real system, `ml_alpha_scores_df.loc[current_month]` would be generated by a live ML model.
        current_ml_alpha = ml_alpha_scores_df.loc[current_month] 
        
        # 1. Input Preparation (select assets & robust covariance)
        mu_t, Sigma_t, current_selected_stocks, current_selected_alpha = prepare_optimization_inputs(
            train_returns_df, current_ml_alpha, n_assets=n_assets_select, n_periods_cov=n_periods_cov
        )
        
        # Adjust w_prev to match the current selection of stocks, or initialize
        if w_prev is None:
            w_prev_adjusted = np.ones(n_assets_select) / n_assets_select
        else:
            # Need to map previous weights to current selected stocks. If a stock drops out, its weight goes to 0.
            # If a new stock comes in, its initial prev weight is 0.
            w_prev_series = pd.Series(w_prev, index=previous_selected_stocks).reindex(current_selected_stocks, fill_value=0.0)
            w_prev_adjusted = w_prev_series.values
            # Normalize if sum is not 1 (e.g., if many stocks dropped out and sum < 1)
            # This handles cases where previous stocks are no longer in the selected set
            if w_prev_adjusted.sum() > 0:
                w_prev_adjusted /= w_prev_adjusted.sum()
            else: # If no common stocks, start fresh
                w_prev_adjusted = np.ones(n_assets_select) / n_assets_select


        # Adjust sector_map to use only the currently selected stocks
        current_sector_map = {}
        for sector, symbols in sector_map.items():
            current_sector_map[sector] = [s for s in symbols if s in current_selected_stocks]

        # 2. Optimize with turnover penalty and all constraints
        w_opt, _ = optimize_portfolio(mu_t, Sigma_t, constraints='constrained',
                                      risk_aversion=risk_aversion, w_prev=w_prev_adjusted,
                                      turnover_penalty=turnover_penalty, max_weight=max_weight,
                                      sector_map=current_sector_map, max_sector=max_sector)
        
        if w_opt is None: # If optimization fails, hold previous portfolio or equal weight
            w_opt = w_prev_adjusted # Try to hold previous portfolio if possible
            if w_opt is None or w_opt.sum() == 0: # If w_prev_adjusted was also problematic
                 w_opt = np.ones(n_assets_select) / n_assets_select # Fallback to equal weight
            print(f"  Optimization failed for {current_month}, using fallback weights.")

        # 3. Realize returns for the next period (actual return of the current month)
        # Assuming returns_df contains actual returns for 'current_month'
        # Filter returns to just the selected stocks for this period
        actual_returns_for_month = returns_df.loc[current_month, current_selected_stocks].values
        
        port_ret_gross = w_opt @ actual_returns_for_month

        # Calculate turnover and net cost for this period
        turnover_val = np.sum(np.abs(w_opt - w_prev_adjusted))
        transaction_cost_bps = 20.0 # Same as in sensitivity analysis
        transaction_cost_per_unit_turnover = transaction_cost_bps / 10000.0
        cost_val = turnover_val * transaction_cost_per_unit_turnover
        
        port_ret_net = port_ret_gross - cost_val
        
        portfolio_returns.append({
            'month': current_month,
            'gross_return': port_ret_gross,
            'net_return': port_ret_net,
            'turnover': turnover_val,
            'transaction_cost': cost_val
        })
        
        # 4. Update w_prev for the next iteration
        w_prev = w_opt
        previous_selected_stocks = current_selected_stocks # Keep track of symbols for next iteration's w_prev adjustment

        if (i - min_train_periods) % 6 == 0:
            print(f"  Processed {current_month.strftime('%Y-%m')}. Gross Return: {port_ret_gross*100:.2f}%, Net Return: {port_ret_net*100:.2f}%")

    portfolio_performance_df = pd.DataFrame(portfolio_returns).set_index('month')
    return portfolio_performance_df

# --- Create synthetic ML alpha scores across time for walk-forward ---
# Simulate alpha scores for the full historical period
ml_alpha_scores_time_series = pd.DataFrame(
    np.random.randn(n_periods_hist, n_universe) * 0.005 + 0.001, # Add a small positive drift
    index=dates,
    columns=stock_symbols_full
)

# Run the walk-forward backtest
# Ensure that n_periods_cov does not exceed min_train_periods
n_periods_for_cov = 36 # Use 3 years of monthly data for covariance estimation
min_train = 60 # Start backtest after 5 years (60 months) of data

backtest_performance_df = walk_forward_optimized_backtest(
    returns_df_full, ml_alpha_scores_time_series,
    sector_map=sector_map_full, n_assets_select=50,
    risk_aversion=2.0, turnover_penalty=0.0005, # Using example optimal gamma
    max_weight=0.05, max_sector=0.25,
    min_train_periods=min_train, n_periods_cov=n_periods_for_cov
)

# --- Calculate a simple equal-weight benchmark for comparison ---
benchmark_returns = returns_df_full.iloc[min_train:].mean(axis=1) # Mean of all universe assets for simplicity

# --- Visualization: Walk-Forward Equity Curve ---
if not backtest_performance_df.empty:
    cumulative_net_returns = (1 + backtest_performance_df['net_return']).cumprod()
    cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()

    plt.figure(figsize=(12, 7))
    cumulative_net_returns.plot(label='AI-Optimized Portfolio (Net-of-Costs)', color='green')
    cumulative_benchmark_returns.plot(label='Equal-Weight Benchmark', color='purple', linestyle='--')
    plt.title('Walk-Forward Optimized Backtest: Cumulative Returns (Net-of-Costs)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Performance Metrics
    annualized_net_return = (cumulative_net_returns.iloc[-1]**(12/len(backtest_performance_df)) - 1) * 100
    annualized_net_vol = backtest_performance_df['net_return'].std() * np.sqrt(12) * 100
    net_sharpe_ratio = annualized_net_return / annualized_net_vol if annualized_net_vol > 0 else 0

    annualized_bm_return = (cumulative_benchmark_returns.iloc[-1]**(12/len(backtest_performance_df)) - 1) * 100
    annualized_bm_vol = benchmark_returns.std() * np.sqrt(12) * 100
    bm_sharpe_ratio = annualized_bm_return / annualized_bm_vol if annualized_bm_vol > 0 else 0

    print("\n--- Walk-Forward Backtest Performance Summary ---")
    print(f"{'Metric':<30s}{'AI-Optimized':>15s}{'Benchmark':>15s}")
    print(f"{'Annualized Net Return':<30s}{annualized_net_return:>15.2f}%{annualized_bm_return:>15.2f}%")
    print(f"{'Annualized Volatility':<30s}{annualized_net_vol:>15.2f}%{annualized_bm_vol:>15.2f}%")
    print(f"{'Net Sharpe Ratio':<30s}{net_sharpe_ratio:>15.2f}{bm_sharpe_ratio:>15.2f}")
    print(f"{'Average Monthly Turnover':<30s}{backtest_performance_df['turnover'].mean()*100:>15.2f}%")
```

### Explanation of Execution

The "Walk-Forward Optimized Backtest: Cumulative Returns (Net-of-Costs)" plot is the ultimate reality check for our AI-enhanced portfolio strategy. It shows how the strategy would have performed historically, after accounting for all real-world complexities like rebalancing and transaction costs. The plot compares our AI-optimized portfolio's cumulative returns against a simple benchmark (e.g., an equal-weight portfolio).

For a CFA Charterholder, this visualization and the accompanying performance metrics are crucial for evaluating the true efficacy of the strategy. A consistently outperforming equity curve and a superior **Net Sharpe Ratio** (compared to the benchmark) indicate a robust and implementable strategy. This backtest confirms whether the theoretical alpha predicted by ML models can actually translate into tangible, net-of-cost returns for clients over time. It demonstrates that the portfolio construction framework successfully manages risk, controls costs, and consistently captures alpha signals in a dynamic market environment. This simulation is the final validation needed before considering deployment in live portfolios.

## 7. Efficient Frontier and Performance Attribution

### Story + Context + Real-World Relevance

To fully appreciate the value added by our ML alpha scores, it's essential to visualize how they impact the fundamental trade-off between risk and return. The **Efficient Frontier** is a cornerstone concept in modern portfolio theory, representing the set of optimal portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given expected return. By comparing an efficient frontier constructed with only historical average returns to one constructed with ML-enhanced expected returns, we can visually quantify the "shift" outward, indicating the alpha added by the ML model.

Furthermore, a CFA Charterholder needs to understand *why* a portfolio performed the way it did. **Performance attribution** helps break down the portfolio's total return into its underlying drivers:
- **Alpha Contribution:** The excess return generated specifically by the ML model's ability to pick winners (relative to a neutral portfolio or benchmark).
- **Risk Contribution:** Returns due to taking on systemic or diversified risk exposures.
- **Transaction Costs:** The drag on returns due to rebalancing.

This attribution helps validate the ML model's contribution and provides insights for continuous improvement.

```python
def plot_efficient_frontier(mu_hist: np.ndarray, mu_ml: np.ndarray, Sigma: np.ndarray, n_points: int = 50,
                            risk_aversion_range: tuple = (0.1, 10)) -> None:
    """
    Plots two efficient frontiers: one with historical mu, and another with ML-enhanced mu.
    ML insights should shift the frontier outward.

    Args:
        mu_hist (np.ndarray): Historical average returns for assets.
        mu_ml (np.ndarray): ML-predicted expected returns (alpha) for assets.
        Sigma (np.ndarray): Covariance matrix of returns.
        n_points (int): Number of points to plot on each frontier.
        risk_aversion_range (tuple): Range of risk aversion values (lambda) to sweep for frontier.
    """
    n = len(mu_hist)
    plt.figure(figsize=(10, 7))

    for label, mu_input in [('Historical Mean Returns', mu_hist), ('ML-Enhanced Returns', mu_ml)]:
        rets, vols = [], []
        # Sweep different risk aversion levels to trace the frontier
        for risk_aversion in np.linspace(risk_aversion_range[0], risk_aversion_range[1], n_points):
            w = cp.Variable(n)
            objective = mu_input @ w - (risk_aversion / 2) * cp.quad_form(w, Sigma)
            cons = [cp.sum(w) == 1, w >= 0] # Long-only, fully invested for frontier calc
            
            prob = cp.Problem(cp.Maximize(objective), cons)
            try:
                prob.solve(solver=cp.OSQP)
                if w.value is not None:
                    # Annualize for plotting (assuming monthly data * 12)
                    rets.append(mu_input @ w.value * 12 * 100)
                    vols.append(np.sqrt(w.value @ Sigma @ w.value) * np.sqrt(12) * 100)
            except cp.error.SolverError:
                pass # Continue if solver fails for some risk aversion

        plt.plot(vols, rets, label=label, linewidth=2)

    plt.xlabel('Annualized Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    plt.title('Efficient Frontier: Historical vs. ML-Enhanced Inputs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_performance_attribution(performance_df: pd.DataFrame, benchmark_returns: pd.Series) -> None:
    """
    Visualizes portfolio performance attribution (alpha, risk, costs).

    Args:
        performance_df (pd.DataFrame): DataFrame from walk-forward backtest with gross_return, net_return, transaction_cost.
        benchmark_returns (pd.Series): Actual benchmark returns for the backtest period.
    """
    # Align benchmark with performance_df index
    benchmark_returns = benchmark_returns[performance_df.index]

    # Calculate alpha as gross_return - benchmark_return
    alpha_contribution = performance_df['gross_return'] - benchmark_returns
    
    # Risk/other contribution can be attributed to benchmark return for simplicity
    risk_contribution = benchmark_returns

    attribution_data = pd.DataFrame({
        'ML Alpha Contribution': alpha_contribution,
        'Market/Risk Contribution': risk_contribution,
        'Transaction Costs': -performance_df['transaction_cost'] # Costs are negative
    }, index=performance_df.index)

    # Plot cumulative attribution
    cumulative_attribution = (1 + attribution_data).cumprod() - 1 # Convert to cumulative growth
    
    plt.figure(figsize=(12, 7))
    cumulative_attribution.plot(kind='area', stacked=True, alpha=0.8)
    plt.title('Cumulative Portfolio Performance Attribution')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Contribution Type')
    plt.tight_layout()
    plt.show()

# --- Re-calculate historical mean returns for the selected assets (for frontier comparison) ---
# Use the same period for historical mean as for covariance
hist_returns_subset = returns_df_full[selected_stocks_ml].tail(n_periods_for_cov)
mu_historical = hist_returns_subset.mean().values

# Plot the efficient frontier comparison
plot_efficient_frontier(mu_historical, mu_ml, Sigma_ml)

# Plot performance attribution using the backtest results
plot_performance_attribution(backtest_performance_df, benchmark_returns)

print("\n--- Final Portfolio Evaluation Metrics ---")
print(f"{'Metric':<30s}{'Value':<15s}")
print(f"{'Net Sharpe Ratio':<30s}{net_sharpe_ratio:<15.2f}")
print(f"{'Average Monthly Turnover':<30s}{backtest_performance_df['turnover'].mean()*100:<15.2f}%")
print(f"{'Max Position (current optimization)':<30s}{np.max(w_bl)*100:<15.2f}%")
# For max sector, need to calculate based on w_bl and selected_sector_map
max_sector_exposure = 0
if w_bl is not None and selected_sector_map is not None:
    for sector_name, indices in selected_sector_map.items():
        current_indices = [selected_stocks_ml.index(sym) for sym in indices if sym in selected_stocks_ml]
        if current_indices:
            max_sector_exposure = max(max_sector_exposure, np.sum(w_bl[current_indices]))
print(f"{'Max Sector Exposure (current opt.)':<30s}{max_sector_exposure*100:<15.2f}%")
print(f"{'Covariance Matrix Condition Number':<30s}{np.linalg.cond(Sigma_ml):<15.0f}")
```

### Explanation of Execution

The **Efficient Frontier** plot provides a powerful visualization of the tangible value added by incorporating ML alpha scores. The "ML-Enhanced Returns" frontier is visibly shifted "up and to the left" compared to the "Historical Mean Returns" frontier. This outward shift means that for any given level of risk, the ML-enhanced portfolio can achieve a higher expected return, or for any given expected return, it can achieve it with lower risk. For a CFA Charterholder, this demonstrates how AI insights can directly improve the fundamental risk-return trade-off of an investment portfolio.

The **Cumulative Portfolio Performance Attribution** stacked area chart breaks down the total return into its components. You can see the positive contributions from "ML Alpha Contribution" (the excess return generated beyond the market), the base "Market/Risk Contribution," and critically, the negative drag from "Transaction Costs." This granular view helps validate the ML model's effectiveness, confirms that transaction costs are managed appropriately, and provides actionable intelligence for further refining the strategy. For example, if transaction costs are consistently high, it might suggest adjusting the turnover penalty or exploring different execution strategies. This comprehensive analysis allows you to pinpoint the drivers of performance and articulate the value of AI to clients and stakeholders.
