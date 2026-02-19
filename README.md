Here's a comprehensive `README.md` file for your Streamlit application lab project, designed to be professional and informative.

---

# QuLab: Lab 50 - Portfolio Optimization with AI Predictions

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Overview

This Streamlit application, "QuLab: Lab 50: Portfolio Optimization with AI Predictions," is an interactive tool designed to simulate and analyze the integration of machine learning (ML) insights into the portfolio construction process. Tailored for investment professionals, quantitative analysts, and students, it provides a hands-on experience in applying ML-generated alpha scores within a constrained Mean-Variance Optimization (MVO) framework using **real historical market data from Yahoo Finance**. The application further explores advanced topics such as turnover penalization, the Black-Litterman model for blending market views with ML predictions, and robust strategy validation through walk-forward backtesting.

The project's high-level story flow guides users through a structured workflow, from data loading to comprehensive performance evaluation, allowing for experimentation with various optimization parameters and risk models.

## Features

The application is structured into several interconnected modules, accessible via the sidebar navigation:

1.  **Application Overview**:
    *   Provides a high-level purpose and a step-by-step story flow of the application.

2.  **Input Data Loading**:
    *   **Downloads real historical stock data** from Yahoo Finance using yfinance library.
    *   Retrieves data for approximately 200 popular stocks from the S&P 500.
    *   Includes 120 months of historical returns and real sector classifications.
    *   Generates ML-based alpha scores using momentum and risk-adjusted return signals.
    *   Displays a preview of the loaded dataframes and real market data.

3.  **Mean-Variance Optimization (MVO)**:
    *   Implements classic MVO to find optimal portfolio weights based on ML-predicted returns and historical covariance.
    *   **Objective Function**:
        $$ \max_{w} \left( w^T \mu_{ML} - \frac{\lambda}{2} w^T \Sigma w \right) $$
        $$ \text{s.t. } \sum w_i = 1, \quad w_{min} \le w_i \le w_{max} $$
        where $\mu_{ML}$ are ML Alpha Scores, $\Sigma$ is the covariance matrix, $\lambda$ is risk aversion, and $w$ are portfolio weights.
    *   Configurable risk aversion parameter ($\lambda$).
    *   Supports "Long Only" or "Long/Short" portfolio constraints.
    *   Displays optimal weights, expected return, and volatility, along with a weight distribution chart.

4.  **Turnover Penalization**:
    *   Introduces a penalty term into the MVO objective function to explicitly control portfolio turnover and associated transaction costs.
    *   **Objective Function (with turnover penalty)**:
        $$ \max_{w} \left( w^T \mu - \frac{\lambda}{2} w^T \Sigma w - \gamma ||w - w_{prev}||_1 \right) $$
        where $\gamma$ is the turnover penalty parameter and $w_{prev}$ are previous period's weights.
    *   Allows users to adjust the turnover penalty parameter ($\gamma$).
    *   Visualizes the trade-off between net portfolio return and turnover for different penalty levels.

5.  **Black-Litterman Integration**:
    *   Applies the Black-Litterman model to combine the market equilibrium implied returns (Prior) with the user's ML-based alpha scores (Views).
    *   **Black-Litterman Posterior Expected Returns**:
        $$ E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q] $$
        This blends market implied returns ($\Pi$) with quantitative views ($Q$), weighted by confidence.
    *   Configurable confidence level in ML views (from pure market equilibrium to full ML confidence).
    *   Displays the posterior expected returns distribution, comparing blended returns against market equilibrium.

6.  **Walk-Forward Backtest**:
    *   Simulates the strategy's performance over time using a rolling window approach, re-optimizing the portfolio at set frequencies.
    *   Configurable rolling window size and rebalance frequency (Daily, Weekly, Monthly).
    *   Tests the robustness of the ML signals and optimization constraints in a dynamic market environment.
    *   Plots the cumulative equity curve of the backtested strategy.

7.  **Performance Evaluation**:
    *   Analyzes the risk-adjusted performance of the AI-optimized portfolio.
    *   Visualizes the **Efficient Frontier** comparing the backtested portfolio against various risk-return profiles.
    *   Provides **Performance Attribution** charts to understand sources of returns.
    *   Displays key performance metrics (e.g., Sharpe Ratio, Sortino Ratio, Max Drawdown).

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/quolab-lab50-portfolio-optimization.git
    cd quolab-lab50-portfolio-optimization
    ```
    (Replace `your-username/quolab-lab50-portfolio-optimization.git` with the actual repository URL)

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not provided, you can create it manually based on the `Technology Stack` section below, or install individually:
    ```bash
    pip install streamlit pandas numpy matplotlib
    # Potentially: scipy cvxpy if used in source.py for advanced optimization
    ```

### Running the Application

1.  **Ensure your virtual environment is activated.**
2.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser (usually at `http://localhost:8501`).

## Usage

1.  **Launch the application** using `streamlit run app.py`.
2.  **Navigate** through the different steps using the `Navigate to Step` dropdown menu in the sidebar.
3.  **Start with "2. Input Data Loading"**:
    *   Upload your own ML Alpha Scores and Historical Returns CSV files.
    *   Alternatively, click "Load and Prepare Data" without uploading files to use generated sample data for demonstration.
4.  **Proceed sequentially through the steps (3 to 7)**:
    *   Experiment with sliders, select boxes, and buttons to adjust parameters and trigger calculations/visualizations.
    *   Ensure data is loaded in Step 2 before attempting subsequent steps, as they rely on the session state.
5.  **Observe the results**: The main panel will update with charts, tables, and metrics based on your interactions.

**Note on Data Format**:
*   **ML Alpha Scores CSV**: Expected to have a `Date` column (or similar, used for time-series alignment) and columns for each asset's alpha score.
*   **Historical Returns CSV**: Expected to have a `Date` column and columns for each asset's historical daily/period returns.

## Project Structure

The project is organized as follows:

```
quolab-lab50-portfolio-optimization/
├── app.py                      # Main Streamlit application file
├── source.py                   # Contains core logic, functions for optimization, Black-Litterman, backtesting, etc.
├── requirements.txt            # List of Python dependencies
├── data/                       # (Optional) Directory for sample input CSV files
│   ├── ml_alpha_scores.csv
│   └── historical_returns.csv
└── README.md                   # This README file
```

*   `app.py`: Handles the Streamlit UI, session state management, and orchestrates calls to the functions defined in `source.py`.
*   `source.py`: Encapsulates all the heavy lifting – data preparation, portfolio optimization algorithms, Black-Litterman model, backtesting engine, and plotting functions. This separation keeps the Streamlit UI clean.

## Technology Stack

*   **Python**: The primary programming language.
*   **Streamlit**: For building the interactive web application and user interface.
*   **Pandas**: Essential for data manipulation, cleaning, and time-series operations.
*   **NumPy**: For numerical computations, especially array operations crucial for financial modeling.
*   **Matplotlib / Plotly**: Used for generating static and interactive visualizations (implied by `st.pyplot` and general plotting needs).
*   **SciPy / CVXPY (Potential)**: While not explicitly imported in `app.py`, complex portfolio optimization and mathematical modeling functions within `source.py` might leverage libraries like SciPy (for optimization routines) or CVXPY (for convex optimization problems).

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  **Fork** the repository.
2.  **Create a new branch** (`git checkout -b feature/AmazingFeature`).
3.  **Make your changes** and commit them (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** to the branch (`git push origin feature/AmazingFeature`).
5.  **Open a Pull Request**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, feedback, or inquiries related to this QuLab project, please reach out to:

*   **QuantUniversity Team**
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com)
*   **Email**: info@quantuniversity.com

---