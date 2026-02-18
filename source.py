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

def generate_synthetic_data(n_universe: int = 200, n_periods_hist: int = 120, num_factors: int = 5, num_sectors: int = 5) -> tuple[pd.DataFrame, pd.Series, dict, pd.DataFrame]:
    """
    Generates synthetic historical returns, ML alpha scores, and a sector map for demonstration.

    Args:
        n_universe (int): Total number of stocks in the universe.
        n_periods_hist (int): Number of historical periods (months).
        num_factors (int): Number of factors for simulating correlated returns.
        num_sectors (int): Number of sectors for the sector map.

    Returns:
        tuple[pd.DataFrame, pd.Series, dict, pd.DataFrame]:
            - returns_df_full (pd.DataFrame): Simulated historical monthly stock returns.
            - ml_alpha_scores_full (pd.Series): Simulated ML-predicted excess returns (alpha).
            - sector_map_full (dict): Simulated sector mapping.
            - ml_alpha_scores_time_series (pd.DataFrame): Simulated ML alpha scores across time for backtest.
    """
    np.random.seed(42) # For reproducibility

    dates = pd.date_range(start='2010-01-01', periods=n_periods_hist, freq='M')
    stock_symbols_full = [f'Stock_{i}' for i in range(n_universe)]

    # Factor model simulation for correlated returns
    factors = np.random.randn(n_periods_hist, num_factors) * 0.01
    factor_betas = np.random.randn(n_universe, num_factors) * 0.5 + 0.5
    idiosyncratic_returns = np.random.randn(n_periods_hist, n_universe) * 0.02

    returns_array = (factors @ factor_betas.T) + idiosyncratic_returns
    returns_df_full = pd.DataFrame(returns_array, index=dates, columns=stock_symbols_full)
    returns_df_full = returns_df_full - returns_df_full.mean() # Center returns

    # Simulate ML alpha scores for a single point in time (initial optimization)
    ml_alpha_scores_full = pd.Series(np.random.randn(n_universe) * 0.005, index=stock_symbols_full)

    # Simulate ML alpha scores across time for walk-forward backtest
    ml_alpha_scores_time_series = pd.DataFrame(
        np.random.randn(n_periods_hist, n_universe) * 0.005 + 0.001, # Add a small positive drift
        index=dates,
        columns=stock_symbols_full
    )

    # Define a simple sector map
    sector_map_full = {f'Sector_{s}': [stock_symbols_full[j] for j in range(s * (n_universe // num_sectors), (s + 1) * (n_universe // num_sectors))]
                       for s in range(num_sectors)}
    # Add any remaining stocks to the last sector if n_universe is not perfectly divisible
    if n_universe % num_sectors != 0:
        last_sector_stocks = stock_symbols_full[(num_sectors-1) * (n_universe // num_sectors):]
        sector_map_full[f'Sector_{num_sectors-1}'] = list(set(sector_map_full[f'Sector_{num_sectors-1}'] + last_sector_stocks))


    return returns_df_full, ml_alpha_scores_full, sector_map_full, ml_alpha_scores_time_series


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
    top_stocks = ml_alpha_scores.abs().nlargest(n_assets).index.tolist()

    # Filter returns and alpha scores to selected top stocks
    returns_subset = returns_df[top_stocks].tail(n_periods_cov).dropna()
    mu = ml_alpha_scores[top_stocks].values

    # Check if returns_subset is empty after filtering
    if returns_subset.empty or len(top_stocks) < n_assets:
        warnings.warn(f"Insufficient data or assets for optimization. {len(top_stocks)} selected, {returns_subset.shape[0]} periods. Returning None.")
        return None, None, [], pd.Series()

    # If the number of selected stocks is less than n_assets due to NAs, adjust
    if len(top_stocks) != returns_subset.shape[1]:
        top_stocks = returns_subset.columns.tolist()
        mu = ml_alpha_scores[top_stocks].values


    # Estimate raw sample covariance matrix for diagnostics
    raw_cov = returns_subset.cov()

    # Covariance: Ledoit-Wolf shrinkage for stability
    lw = LedoitWolf()
    lw.fit(returns_subset)
    Sigma = lw.covariance_
    shrinkage_intensity = lw.shrinkage_

    print(f"Universe selected for optimization: {len(top_stocks)} stocks")
    print(f"Expected returns range (ML alpha): [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"Covariance matrix shape: {Sigma.shape}, shrinkage intensity={shrinkage_intensity:.3f}")
    print(f"Condition number (raw sample covariance): {np.linalg.cond(raw_cov):.0f}")
    print(f"Condition number (Ledoit-Wolf shrunk covariance): {np.linalg.cond(Sigma):.0f}")

    return mu, Sigma, top_stocks, ml_alpha_scores[top_stocks]


def optimize_portfolio(mu: np.ndarray, Sigma: np.ndarray, constraints: str = 'constrained',
                       risk_aversion: float = 2.0, w_prev: np.ndarray = None,
                       turnover_penalty: float = 0.0, max_weight: float = 0.05,
                       sector_map: dict = None, max_sector: float = 0.25,
                       asset_symbols_in_mu: list = None) -> tuple[np.ndarray, float]:
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
        sector_map (dict, optional): Dictionary mapping sector names to lists of asset symbols.
        max_sector (float): Maximum allowed weight for any single sector.
        asset_symbols_in_mu (list, optional): List of asset symbols corresponding to the current `mu` and `Sigma`.
                                             Required for correct sector constraint mapping.

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
        if sector_map is not None and asset_symbols_in_mu is not None:
            for sector_name, symbols_in_sector in sector_map.items():
                # Map stock symbols to their indices in the current `mu` vector
                current_indices = [asset_symbols_in_mu.index(sym) for sym in symbols_in_sector if sym in asset_symbols_in_mu]
                if current_indices: # Only add constraint if sector has assets in current selection
                    cons += [cp.sum(w[current_indices]) <= max_sector]

    elif constraints == 'naive':
        # Naive: fully invested, long-only, but without max_weight or sector_map constraints
        cons += [w >= 0]
        cons += [w <= 1.0] # Effectively no max_weight limit
    else:
        raise ValueError("Invalid constraints type. Choose 'naive' or 'constrained'.")

    # Solve the optimization problem
    prob = cp.Problem(cp.Maximize(objective), cons)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
        if prob.status in ['optimal', 'optimal_near']:
            return w.value, prob.value
        else:
            print(f"Warning: Solver status = {prob.status} for optimization.")
            return None, None
    except cp.error.SolverError:
        print("Error: Solver failed to converge for this problem.")
        return None, None


def turnover_sensitivity(mu: np.ndarray, Sigma: np.ndarray, w_prev: np.ndarray,
                         asset_symbols_in_mu: list, gammas: list = None, risk_aversion: float = 2.0,
                         max_weight: float = 0.05, sector_map: dict = None,
                         max_sector: float = 0.25, transaction_cost_bps: float = 20.0) -> pd.DataFrame:
    """
    Sweeps turnover penalty (gamma) to find the optimal trade-off between alpha capture and transaction costs.

    Args:
        mu (np.ndarray): Expected returns vector (ML alpha scores).
        Sigma (np.ndarray): Covariance matrix of returns.
        w_prev (np.ndarray): Previous period's weights, typically an equal-weight portfolio to start.
        asset_symbols_in_mu (list): List of asset symbols corresponding to the current `mu` and `Sigma`.
        gammas (list): List of turnover penalty (gamma) values to test.
        risk_aversion (float): Lambda parameter for risk aversion.
        max_weight (float): Maximum allowed weight for any single asset.
        sector_map (dict): Dictionary mapping sector names to lists of asset symbols.
        max_sector (float): Maximum allowed weight for any single sector.
        transaction_cost_bps (float): Transaction cost in basis points per unit of turnover (e.g., 20 bps = 0.0020).

    Returns:
        pd.DataFrame: Results of the sensitivity analysis (gamma, turnover, gross/net returns/Sharpe).
    """
    if gammas is None:
        gammas = [0, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.02]

    results = []
    transaction_cost_per_unit_turnover = transaction_cost_bps / 10000.0 # Convert bps to decimal

    print("\nTURNOVER SENSITIVITY ANALYSIS")
    print("=" * 75)
    print(f"{'Gamma':<8s}{'Turnover':>10s}{'Gross Ret':>12s}{'Net Ret':>12s}{'Gross SR':>10s}{'Net SR':>10s}")

    # Adjust sector_map to use only the selected stocks for this optimization step
    current_sector_map = {}
    for sector, symbols in sector_map.items():
        current_sector_map[sector] = [s for s in symbols if s in asset_symbols_in_mu]

    for gamma in gammas:
        w, _ = optimize_portfolio(mu, Sigma, constraints='constrained',
                                 risk_aversion=risk_aversion, w_prev=w_prev,
                                 turnover_penalty=gamma, max_weight=max_weight,
                                 sector_map=current_sector_map, max_sector=max_sector,
                                 asset_symbols_in_mu=asset_symbols_in_mu)

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


def plot_turnover_sensitivity(turnover_results_df: pd.DataFrame) -> None:
    """
    Plots the turnover sensitivity analysis results.
    """
    if turnover_results_df.empty:
        print("No turnover sensitivity data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(turnover_results_df['gamma'], turnover_results_df['sharpe_gross'], marker='o', label='Gross Sharpe Ratio')
    plt.plot(turnover_results_df['gamma'], turnover_results_df['sharpe_net'], marker='x', label='Net Sharpe Ratio')

    # Find optimal gamma for Net Sharpe Ratio
    optimal_gamma_row = turnover_results_df.loc[turnover_results_df['sharpe_net'].idxmax()]
    optimal_gamma = optimal_gamma_row['gamma']
    optimal_net_sharpe = optimal_gamma_row['sharpe_net']

    plt.axvline(x=optimal_gamma, color='gray', linestyle='--', label=f'Optimal Net Sharpe Gamma ({optimal_gamma:.5f})')
    plt.scatter(optimal_gamma, optimal_net_sharpe, color='red', marker='*', s=200, zorder=5, label='Max Net Sharpe')

    plt.title('Turnover Penalty Sensitivity: Gross vs. Net Sharpe Ratio')
    plt.xlabel('Turnover Penalty (γ)')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.xscale('log') # Gamma often varies logarithmically
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plt.show()


def black_litterman(Sigma: np.ndarray, market_cap_weights: np.ndarray, ml_views: np.ndarray,
                    risk_aversion: float = 2.5, tau: float = 0.05, view_confidence: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - posterior_mu (np.ndarray): Black-Litterman posterior mean (blended expected returns).
            - posterior_cov (np.ndarray): Black-Litterman posterior covariance.
            - Pi (np.ndarray): Equilibrium returns.
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
    # Ensure view_confidence is not zero to avoid division by zero.
    view_confidence_safe = max(view_confidence, 1e-6)
    omega_diag = np.diag(tau * Sigma) / view_confidence_safe
    Omega = np.diag(omega_diag)

    # 4. Black-Litterman Posterior
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)

    # Posterior covariance matrix
    posterior_cov_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
    posterior_cov = np.linalg.inv(posterior_cov_inv)

    # Posterior mean vector
    posterior_mu = posterior_cov @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

    print(f"\nBlack-Litterman Posterior (View Confidence: {view_confidence}):")
    print(f"  Equilibrium mean return: {Pi.mean()*100:.3f}%")
    print(f"  ML views mean: {Q.mean()*100:.3f}%")
    print(f"  BL posterior mean: {posterior_mu.mean()*100:.3f}%")

    return posterior_mu, posterior_cov, Pi


def plot_black_litterman_blend(mu_ml: np.ndarray, mu_bl: np.ndarray, Pi: np.ndarray,
                               selected_stocks: list, num_top_stocks: int = 10) -> None:
    """
    Visualizes the Black-Litterman blend for top stocks.
    """
    if mu_bl is None:
        print("No Black-Litterman data to plot.")
        return

    # Select top N stocks by BL posterior return for visualization
    top_stocks_bl_idx = np.argsort(mu_bl)[-num_top_stocks:]
    top_stock_symbols = [selected_stocks[i] for i in top_stocks_bl_idx]

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

    w_prev = None
    previous_selected_stocks = [] # To keep track of stock symbols for w_prev adjustment

    print(f"\nStarting walk-forward backtest over {len(months) - min_train_periods} periods...")
    for i in range(min_train_periods, len(months)):
        current_month = months[i]

        # Training data for this period (up to current_month - 1)
        train_returns_df = returns_df[returns_df.index < current_month]

        # ML alpha scores for the *current* month (to predict next month's returns)
        current_ml_alpha = ml_alpha_scores_df.loc[current_month]

        # 1. Input Preparation (select assets & robust covariance)
        mu_t, Sigma_t, current_selected_stocks, current_selected_alpha = prepare_optimization_inputs(
            train_returns_df, current_ml_alpha, n_assets=n_assets_select, n_periods_cov=n_periods_cov
        )

        if mu_t is None or len(current_selected_stocks) == 0:
            print(f"  Skipping {current_month} due to insufficient data or asset selection failure.")
            # Fallback to zero return or hold previous weights (if possible)
            if w_prev is not None and len(w_prev) > 0:
                # If we have a previous portfolio, try to rebalance into an equal-weighted version of *its* selected assets
                # or just hold cash/zero. For simplicity, we add zero return.
                portfolio_returns.append({
                    'month': current_month,
                    'gross_return': 0.0,
                    'net_return': 0.0,
                    'turnover': 0.0,
                    'transaction_cost': 0.0
                })
            else:
                portfolio_returns.append({
                    'month': current_month,
                    'gross_return': 0.0,
                    'net_return': 0.0,
                    'turnover': 0.0,
                    'transaction_cost': 0.0
                })
            # Ensure w_prev is updated for consistency even if this period was skipped.
            # If no new stocks were selected, previous portfolio cannot be maintained,
            # so next iteration will have `w_prev` reset to equal weight.
            w_prev = None
            previous_selected_stocks = []
            continue


        # Adjust w_prev to match the current selection of stocks, or initialize
        if w_prev is None or len(previous_selected_stocks) == 0: # Cold start or empty previous selection
            w_prev_adjusted = np.ones(len(current_selected_stocks)) / len(current_selected_stocks)
        else:
            w_prev_series = pd.Series(w_prev, index=previous_selected_stocks)
            w_prev_adjusted = w_prev_series.reindex(current_selected_stocks, fill_value=0.0).values
            if w_prev_adjusted.sum() > 0:
                w_prev_adjusted /= w_prev_adjusted.sum()
            else:
                w_prev_adjusted = np.ones(len(current_selected_stocks)) / len(current_selected_stocks)


        # Adjust sector_map to use only the currently selected stocks
        current_sector_map = {}
        for sector, symbols in sector_map.items():
            current_sector_map[sector] = [s for s in symbols if s in current_selected_stocks]

        # 2. Optimize with turnover penalty and all constraints
        w_opt, _ = optimize_portfolio(mu_t, Sigma_t, constraints='constrained',
                                      risk_aversion=risk_aversion, w_prev=w_prev_adjusted,
                                      turnover_penalty=turnover_penalty, max_weight=max_weight,
                                      sector_map=current_sector_map, max_sector=max_sector,
                                      asset_symbols_in_mu=current_selected_stocks)

        if w_opt is None: # If optimization fails, hold previous portfolio or equal weight
            w_opt = w_prev_adjusted # Try to hold previous portfolio if possible
            if w_opt is None or w_opt.sum() == 0:
                 w_opt = np.ones(len(current_selected_stocks)) / len(current_selected_stocks) # Fallback to equal weight
            print(f"  Optimization failed for {current_month}, using fallback weights.")


        # 3. Realize returns for the next period (actual return of the current month)
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
        previous_selected_stocks = current_selected_stocks

        if (i - min_train_periods) % 12 == 0: # Print every year
            print(f"  Processed {current_month.strftime('%Y-%m')}. Gross Return: {port_ret_gross*100:.2f}%, Net Return: {port_ret_net*100:.2f}%")

    portfolio_performance_df = pd.DataFrame(portfolio_returns).set_index('month')
    return portfolio_performance_df


def plot_walk_forward_equity_curve(backtest_performance_df: pd.DataFrame, benchmark_returns: pd.Series) -> None:
    """
    Plots the cumulative returns of the optimized portfolio and a benchmark.
    """
    if backtest_performance_df.empty:
        print("No backtest performance data to plot.")
        return

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


def plot_efficient_frontier(mu_hist: np.ndarray, mu_ml: np.ndarray, Sigma: np.ndarray,
                            asset_symbols: list, n_points: int = 50,
                            risk_aversion_range: tuple = (0.1, 10)) -> None:
    """
    Plots two efficient frontiers: one with historical mu, and another with ML-enhanced mu.
    ML insights should shift the frontier outward.

    Args:
        mu_hist (np.ndarray): Historical average returns for assets.
        mu_ml (np.ndarray): ML-predicted expected returns (alpha) for assets.
        Sigma (np.ndarray): Covariance matrix of returns.
        asset_symbols (list): List of asset symbols corresponding to mu and Sigma.
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
    if performance_df.empty:
        print("No performance attribution data to plot.")
        return

    # Align benchmark with performance_df index
    benchmark_returns_aligned = benchmark_returns[performance_df.index]

    # Calculate alpha as gross_return - benchmark_return
    alpha_contribution = performance_df['gross_return'] - benchmark_returns_aligned

    # Risk/other contribution can be attributed to benchmark return for simplicity
    market_risk_contribution = benchmark_returns_aligned

    attribution_data = pd.DataFrame({
        'ML Alpha Contribution': alpha_contribution,
        'Market/Risk Contribution': market_risk_contribution,
        'Transaction Costs': -performance_df['transaction_cost'] # Costs are negative
    }, index=performance_df.index)

    cumulative_attribution = (1 + attribution_data).cumprod() - 1

    plt.figure(figsize=(12, 7))
    cumulative_attribution.plot(kind='line', alpha=0.8, ax=plt.gca())
    plt.title('Cumulative Portfolio Performance Attribution')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Contribution Type')
    plt.tight_layout()
    plt.show()


def run_portfolio_optimization_pipeline(
    n_universe: int = 200, n_periods_hist: int = 120, n_assets_select: int = 50,
    n_periods_cov: int = 60, min_train_periods: int = 60,
    risk_aversion_base: float = 2.0, max_weight_limit: float = 0.05, max_sector_limit: float = 0.25,
    bl_view_confidence: float = 0.3, bl_tau: float = 0.05
) -> dict:
    """
    Executes the full portfolio optimization pipeline including synthetic data generation,
    MVO, Black-Litterman, turnover sensitivity, and walk-forward backtest.

    Args:
        n_universe (int): Total stocks in the universe.
        n_periods_hist (int): Total historical periods for synthetic data.
        n_assets_select (int): Number of top assets to select for optimization.
        n_periods_cov (int): Number of periods for covariance estimation.
        min_train_periods (int): Minimum periods required for initial training in backtest.
        risk_aversion_base (float): Base risk aversion for MVO.
        max_weight_limit (float): Maximum allowed weight for any single asset.
        max_sector_limit (float): Maximum allowed weight for any single sector.
        bl_view_confidence (float): Confidence level in ML views for Black-Litterman.
        bl_tau (float): Tau parameter for Black-Litterman model.

    Returns:
        dict: A dictionary containing performance metrics and potentially plot handles.
    """
    print("--- Starting Portfolio Optimization Pipeline ---")

    # 1. Generate Synthetic Data
    returns_df_full, ml_alpha_scores_full, sector_map_full, ml_alpha_scores_time_series = generate_synthetic_data(
        n_universe=n_universe, n_periods_hist=n_periods_hist
    )

    # 2. Prepare Inputs for Initial Optimization
    mu_ml, Sigma_ml, selected_stocks_ml, selected_ml_alpha = prepare_optimization_inputs(
        returns_df_full, ml_alpha_scores_full, n_assets=n_assets_select, n_periods_cov=n_periods_cov
    )

    if mu_ml is None:
        print("Initial input preparation failed. Exiting pipeline.")
        return {}

    # Adjust sector_map_full to use only the selected stocks for this optimization step
    initial_selected_sector_map = {}
    for sector, symbols in sector_map_full.items():
        initial_selected_sector_map[sector] = [s for s in symbols if s in selected_stocks_ml]

    # 3. Execute Naive (unconstrained) MVO
    print("\n--- Running Naive MVO ---")
    w_naive, obj_naive = optimize_portfolio(mu_ml, Sigma_ml, constraints='naive',
                                            risk_aversion=0.1, max_weight=1.0, # Low risk aversion, no real max_weight
                                            sector_map=initial_selected_sector_map,
                                            asset_symbols_in_mu=selected_stocks_ml)

    # 4. Execute Constrained MVO
    print("\n--- Running Constrained MVO ---")
    w_constrained, obj_con = optimize_portfolio(mu_ml, Sigma_ml, constraints='constrained',
                                                risk_aversion=risk_aversion_base, max_weight=max_weight_limit,
                                                sector_map=initial_selected_sector_map, max_sector=max_sector_limit,
                                                asset_symbols_in_mu=selected_stocks_ml)

    # 5. Comparison Metrics
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

        # Visualization: Weight Distribution
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

    # 6. Turnover Sensitivity Analysis
    n_selected = len(selected_stocks_ml)
    w_prev_equal = np.ones(n_selected) / n_selected

    turnover_results_df = turnover_sensitivity(mu_ml, Sigma_ml, w_prev=w_prev_equal,
                                             asset_symbols_in_mu=selected_stocks_ml,
                                             sector_map=initial_selected_sector_map,
                                             gammas=[0, 0.00005, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01],
                                             risk_aversion=risk_aversion_base,
                                             max_weight=max_weight_limit,
                                             max_sector=max_sector_limit)

    plot_turnover_sensitivity(turnover_results_df)

    # Determine optimal gamma for Black-Litterman and Backtest
    optimal_gamma_for_bl = 0.0005 # Default if sensitivity analysis fails or is empty
    if not turnover_results_df.empty:
        optimal_gamma_for_bl = turnover_results_df.loc[turnover_results_df['sharpe_net'].idxmax()]['gamma']
        print(f"\nOptimal turnover penalty (gamma) found: {optimal_gamma_for_bl:.5f}")
    else:
        print("\nCould not determine optimal gamma from sensitivity analysis. Using default 0.0005.")

    # 7. Black-Litterman Model Integration
    print("\n--- Running Black-Litterman Model ---")
    market_cap_weights = np.ones(len(selected_stocks_ml)) / len(selected_stocks_ml) # Proxy for market equilibrium

    mu_bl, Sigma_bl_posterior, Pi = black_litterman(Sigma_ml, market_cap_weights, mu_ml,
                                                    risk_aversion=risk_aversion_base, tau=bl_tau,
                                                    view_confidence=bl_view_confidence)

    # Optimize portfolio with BL inputs
    w_bl, obj_bl = optimize_portfolio(mu_bl, Sigma_ml, constraints='constrained',
                                     risk_aversion=risk_aversion_base, w_prev=w_prev_equal,
                                     turnover_penalty=optimal_gamma_for_bl, max_weight=max_weight_limit,
                                     sector_map=initial_selected_sector_map, max_sector=max_sector_limit,
                                     asset_symbols_in_mu=selected_stocks_ml)

    plot_black_litterman_blend(mu_ml, mu_bl, Pi, selected_stocks_ml)

    # 8. Walk-Forward Optimized Backtest
    print("\n--- Running Walk-Forward Backtest ---")
    # Ensure that n_periods_cov does not exceed min_train_periods
    n_periods_for_cov_backtest = min(n_periods_cov, min_train_periods - 1)
    if n_periods_for_cov_backtest < 1:
        print("Warning: n_periods_cov too small for backtest. Adjusting to 1.")
        n_periods_for_cov_backtest = 1

    backtest_performance_df = walk_forward_optimized_backtest(
        returns_df_full, ml_alpha_scores_time_series,
        sector_map=sector_map_full, n_assets_select=n_assets_select,
        risk_aversion=risk_aversion_base, turnover_penalty=optimal_gamma_for_bl,
        max_weight=max_weight_limit, max_sector=max_sector_limit,
        min_train_periods=min_train_periods, n_periods_cov=n_periods_for_cov_backtest
    )

    # Calculate a simple equal-weight benchmark for comparison
    benchmark_returns = returns_df_full.iloc[min_train_periods:].mean(axis=1)

    plot_walk_forward_equity_curve(backtest_performance_df, benchmark_returns)

    # Performance Metrics for Backtest
    results_summary = {}
    if not backtest_performance_df.empty:
        cumulative_net_returns = (1 + backtest_performance_df['net_return']).cumprod()
        cumulative_benchmark_returns = (1 + benchmark_returns.loc[backtest_performance_df.index]).cumprod()

        annualized_net_return = (cumulative_net_returns.iloc[-1]**(12/len(backtest_performance_df)) - 1) * 100
        annualized_net_vol = backtest_performance_df['net_return'].std() * np.sqrt(12) * 100
        net_sharpe_ratio = annualized_net_return / annualized_net_vol if annualized_net_vol > 0 else 0

        annualized_bm_return = (cumulative_benchmark_returns.iloc[-1]**(12/len(backtest_performance_df)) - 1) * 100
        annualized_bm_vol = benchmark_returns.loc[backtest_performance_df.index].std() * np.sqrt(12) * 100
        bm_sharpe_ratio = annualized_bm_return / annualized_bm_vol if annualized_bm_vol > 0 else 0

        print("\n--- Walk-Forward Backtest Performance Summary ---")
        print(f"{'Metric':<30s}{'AI-Optimized':>15s}{'Benchmark':>15s}")
        print(f"{'Annualized Net Return':<30s}{annualized_net_return:>15.2f}%{annualized_bm_return:>15.2f}%")
        print(f"{'Annualized Volatility':<30s}{annualized_net_vol:>15.2f}%{annualized_bm_vol:>15.2f}%")
        print(f"{'Net Sharpe Ratio':<30s}{net_sharpe_ratio:>15.2f}{bm_sharpe_ratio:>15.2f}")
        print(f"{'Average Monthly Turnover':<30s}{backtest_performance_df['turnover'].mean()*100:>15.2f}%")

        results_summary['annualized_net_return'] = annualized_net_return
        results_summary['annualized_net_vol'] = annualized_net_vol
        results_summary['net_sharpe_ratio'] = net_sharpe_ratio
        results_summary['average_monthly_turnover'] = backtest_performance_df['turnover'].mean()*100

    # 9. Visualization: Efficient Frontier
    hist_returns_subset = returns_df_full[selected_stocks_ml].tail(n_periods_cov)
    mu_historical = hist_returns_subset.mean().values
    plot_efficient_frontier(mu_historical, mu_ml, Sigma_ml, selected_stocks_ml)

    # 10. Visualization: Performance Attribution
    if not backtest_performance_df.empty and not benchmark_returns.empty:
        plot_performance_attribution(backtest_performance_df, benchmark_returns)

    # 11. Final Portfolio Evaluation Metrics
    print("\n--- Final Portfolio Evaluation Metrics ---")
    print(f"{'Metric':<30s}{'Value':<15s}")
    print(f"{'Net Sharpe Ratio':<30s}{results_summary.get('net_sharpe_ratio', 'N/A'):<15.2f}")
    print(f"{'Average Monthly Turnover':<30s}{results_summary.get('average_monthly_turnover', 'N/A'):<15.2f}%")

    current_max_weight = 0
    current_max_sector_exposure = 0
    if w_bl is not None:
        current_max_weight = np.max(w_bl) * 100
        for sector_name, symbols in initial_selected_sector_map.items():
            current_indices = [selected_stocks_ml.index(sym) for sym in symbols if sym in selected_stocks_ml]
            if current_indices:
                current_max_sector_exposure = max(current_max_sector_exposure, np.sum(w_bl[current_indices]))
        current_max_sector_exposure *= 100

    print(f"{'Max Position (current opt.)':<30s}{current_max_weight:<15.2f}%")
    print(f"{'Max Sector Exposure (current opt.)':<30s}{current_max_sector_exposure:<15.2f}%")
    print(f"{'Covariance Matrix Condition Number':<30s}{np.linalg.cond(Sigma_ml):<15.0f}")

    print("\n--- Portfolio Optimization Pipeline Completed ---")
    return results_summary


# This block ensures the script runs only when executed directly, not when imported.
if __name__ == "__main__":
    # Example usage of the pipeline function
    pipeline_results = run_portfolio_optimization_pipeline(
        n_universe=200,
        n_periods_hist=120,
        n_assets_select=50,
        n_periods_cov=60, # Use 5 years for covariance (60 months)
        min_train_periods=60, # Start backtest after 5 years (60 months)
        risk_aversion_base=2.0,
        max_weight_limit=0.05,
        max_sector_limit=0.25,
        bl_view_confidence=0.3,
        bl_tau=0.05
    )
    # You can access specific results from the returned dictionary if needed
    # print("\nFinal pipeline results summary:", pipeline_results)
