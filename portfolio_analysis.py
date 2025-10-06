import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict

from etf_allocations import (
    ETF_UNIVERSE,
    ETF_PORTFOLIOS,
    fetch_historical_data,
    PortfolioAllocation
)

def generate_etf_report():
    """Generate comprehensive ETF analysis report"""
    # 1. ETF Universe Table
    def print_etf_characteristics():
        characteristics = pd.DataFrame({
            'Net Assets (B)': [ETF_UNIVERSE[t].net_assets for t in ETF_UNIVERSE],
            'Avg Volume (M)': [ETF_UNIVERSE[t].avg_volume for t in ETF_UNIVERSE],
            'Expense Ratio (%)': [ETF_UNIVERSE[t].expense_ratio * 100 for t in ETF_UNIVERSE],
            'Historical Return (%)': [ETF_UNIVERSE[t].historical_return * 100 for t in ETF_UNIVERSE],
            'Volatility (%)': [ETF_UNIVERSE[t].volatility * 100 for t in ETF_UNIVERSE]
        }, index=ETF_UNIVERSE.keys())
        return characteristics

    # 2. Return and Correlation Analysis
    def plot_correlation_matrix(returns: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('ETF Correlation Matrix')
        plt.tight_layout()
        return plt.gcf()

    # 3. Efficient Frontier Visualization
    def plot_efficient_frontier(portfolios: Dict[int, PortfolioAllocation]):
        vols = [p['target_volatility'] for p in portfolios.values()]
        rets = [p['target_return'] for p in portfolios.values()]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(vols, rets, c='blue', marker='o')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        for i, (vol, ret) in enumerate(zip(vols, rets), 1):
            plt.annotate(f'Risk {i}', (vol, ret))
        return plt.gcf()

    # 4. Risk Level Mapping Table
    def print_risk_level_mappings():
        mappings = pd.DataFrame({
            'Risk Level': range(1, 11),
            'Target Vol': [p['target_volatility'] for p in ETF_PORTFOLIOS.values()],
            'Exp Return': [p['target_return'] for p in ETF_PORTFOLIOS.values()],
            'Sharpe': [p['sharpe_ratio'] for p in ETF_PORTFOLIOS.values()]
        })
        return mappings

    # 5. Performance Metrics and Backtesting
    def calculate_portfolio_performance(portfolio: PortfolioAllocation, 
                                     returns: pd.DataFrame) -> dict:
        weights = np.array(list(portfolio['allocation'].values()))
        portfolio_returns = returns.dot(weights)
        
        # Calculate metrics
        sharpe = portfolio['sharpe_ratio']
        vol = portfolio['target_volatility']
        ret = portfolio['target_return']
        var_95 = stats.norm.ppf(0.05, ret, vol)  # 95% VaR
        
        return {
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'var_95': var_95
        }

    # Generate full report
    historical_data = fetch_historical_data()
    returns = historical_data.pct_change().dropna()
    
    report = {
        'etf_characteristics': print_etf_characteristics(),
        'correlation_matrix': plot_correlation_matrix(returns),
        'efficient_frontier': plot_efficient_frontier(ETF_PORTFOLIOS),
        'risk_mappings': print_risk_level_mappings(),
        'performance_metrics': {
            level: calculate_portfolio_performance(portfolio, returns)
            for level, portfolio in ETF_PORTFOLIOS.items()
        }
    }
    
    return report

def backtest_portfolio(portfolio: PortfolioAllocation, 
                      start_date: str = '2019-01-01') -> pd.Series:
    """Backtest portfolio performance"""
    historical_data = fetch_historical_data(start_date)
    returns = historical_data.pct_change().dropna()
    
    weights = np.array(list(portfolio['allocation'].values()))
    portfolio_returns = returns.dot(weights)
    
    return (1 + portfolio_returns).cumprod()  # Cumulative returns