import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Union

from etf_allocations import (
    ETF_DATA,
    get_etf_portfolio,
    get_filtered_etfs,
    get_historical_returns
)

# Type alias for portfolio structure
Portfolio = Dict[str, Any]

def generate_etf_report():
    """Generate comprehensive ETF analysis report"""
    # 1. ETF Universe Table
    def print_etf_characteristics():
        etfs = []
        for category in ETF_DATA.values():
            etfs.extend(category)
            
        # Convert to DataFrame
        df = pd.DataFrame.from_records(etfs)
        characteristics = pd.DataFrame({
            'ETF Name': df['ETF_Name'],
            'Risk Level': df['risk_level'],
            'Expense Ratio (%)': df['expense_ratio'] * 100,
            'Region': df['region'],
            'Asset Type': df['asset_type']
        }, index=df['Ticker'])
        return characteristics

    # 2. Return and Correlation Analysis
    def plot_correlation_matrix(returns: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('ETF Correlation Matrix')
        plt.tight_layout()
        return plt.gcf()

    # 3. Efficient Frontier Visualization
    def plot_efficient_frontier(risk_buckets: List[int] = range(1, 11)):
        portfolios = [get_etf_portfolio(bucket) for bucket in risk_buckets]
        vols = [p['target_volatility'] for p in portfolios]
        rets = [p['target_return'] for p in portfolios]
        
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
        portfolios = [get_etf_portfolio(bucket) for bucket in range(1, 11)]
        mappings = pd.DataFrame({
            'Risk Level': range(1, 11),
            'Target Vol': [p['target_volatility'] for p in portfolios],
            'Exp Return': [p['target_return'] for p in portfolios],
            'Sharpe': [p['sharpe_ratio'] for p in portfolios]
        })
        return mappings

    # 5. Performance Metrics and Backtesting
    def calculate_portfolio_performance(portfolio: Portfolio, 
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
    # Get all tickers from ETF_DATA
    all_etfs = []
    for category in ETF_DATA.values():
        all_etfs.extend(category)
    tickers = [etf['Ticker'] for etf in all_etfs if etf.get('Ticker')]
    
    # Get historical data
    returns = get_historical_returns(tickers)
    
    # Generate portfolios for all risk buckets
    portfolios = {i: get_etf_portfolio(i) for i in range(1, 11)}
    
    report = {
        'etf_characteristics': print_etf_characteristics(),
        'correlation_matrix': plot_correlation_matrix(returns),
        'efficient_frontier': plot_efficient_frontier(),
        'risk_mappings': print_risk_level_mappings(),
        'performance_metrics': {
            level: calculate_portfolio_performance(portfolio, returns)
            for level, portfolio in portfolios.items()
        }
    }
    
    return report

def backtest_portfolio(portfolio: Portfolio, 
                      start_date: str = '2019-01-01') -> pd.Series:
    """Backtest portfolio performance"""
    # Get tickers from portfolio
    tickers = list(portfolio['allocation'].keys())
    
    # Get historical data
    returns = get_historical_returns(tickers, period="max")
    returns = returns[returns.index >= start_date]
    
    weights = np.array(list(portfolio['allocation'].values()))
    portfolio_returns = returns.dot(weights)
    
    return (1 + portfolio_returns).cumprod()  # Cumulative returns