from dataclasses import dataclass
from typing import Dict, List, TypedDict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

@dataclass
class ETFMetrics:
    ticker: str
    name: str
    net_assets: float  # in billions
    avg_volume: float  # in millions
    expense_ratio: float
    historical_return: float
    volatility: float
    min_investment: float

class PortfolioAllocation(TypedDict):
    name: str
    description: str
    target_volatility: float
    target_return: float
    allocation: Dict[str, float]
    sharpe_ratio: float


def get_etf_metrics(ticker: str) -> ETFMetrics:
    """Extract real-time metrics for an ETF using yfinance"""
    try:
        print(f"Fetching data for {ticker}...")
        etf = yf.Ticker(ticker)
        info = etf.info
        if not info:
            raise ValueError(f"No info available for {ticker}")
            
        # Get 3 years of historical data for return calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        hist = etf.history(start=start_date)
        
        if hist.empty:
            raise ValueError(f"No historical data available for {ticker}")
        
        # Calculate returns and volatility
        returns = hist['Close'].pct_change().dropna()
        historical_return = ((1 + returns.mean()) ** 252 - 1)
        volatility = returns.std() * np.sqrt(252)
        
        # Extract other metrics
        net_assets = info.get('totalAssets', 0) / 1e9  # Convert to billions
        avg_volume = info.get('averageVolume10Day', 0) / 1e6  # Convert to millions
        expense_ratio = info.get('expenseRatio', 0)
        current_price = info.get('regularMarketPrice', 100)
        min_investment = current_price * 100  # Assume 100 shares minimum
        
        return ETFMetrics(
            ticker=ticker,
            name=info.get('longName', f"{ticker} ETF"),
            net_assets=net_assets,
            avg_volume=avg_volume,
            expense_ratio=expense_ratio,
            historical_return=historical_return,
            volatility=volatility,
            min_investment=min_investment
        )
    except Exception as e:
        print(f"Error fetching metrics for {ticker}: {str(e)}")
        return None

def initialize_etf_universe() -> Dict[str, ETFMetrics]:
    """Initialize ETF universe with real-time data"""
    etf_tickers = {
        'SPY': 'SPDR S&P 500 ETF',
        'EFA': 'iShares MSCI EAFE ETF',
        'EEM': 'iShares MSCI Emerging Markets ETF',
        'AGG': 'iShares Core U.S. Aggregate Bond ETF',
        'BNDX': 'Vanguard Total International Bond ETF'
    }
    
    universe = {}
    for ticker, name in etf_tickers.items():
        try:
            print(f"Fetching data for {ticker}...")
            metrics = get_etf_metrics(ticker)
            if metrics:
                universe[ticker] = metrics
            else:
                print(f"Warning: Using default values for {ticker}")
                universe[ticker] = ETFMetrics(
                    ticker=ticker,
                    name=name,
                    net_assets=50.0,  # Conservative default
                    avg_volume=10.0,   # Conservative default
                    expense_ratio=0.005,  # Conservative default (0.5%)
                    historical_return=0.07,  # Conservative default (7%)
                    volatility=0.15,    # Conservative default (15%)
                    min_investment=1000  # Default minimum
                )
        except Exception as e:
            print(f"Error initializing {ticker}: {str(e)}")
            # Use default values for this ETF
            universe[ticker] = ETFMetrics(
                ticker=ticker,
                name=name,
                net_assets=50.0,
                avg_volume=10.0,
                expense_ratio=0.005,
                historical_return=0.07,
                volatility=0.15,
                min_investment=1000
            )
            universe[ticker] = ETFMetrics(
                ticker=ticker,
                name=name,
                net_assets=50.0,  # Conservative default
                avg_volume=10.0,   # Conservative default
                expense_ratio=0.005,  # Conservative default (0.5%)
                historical_return=0.07,  # Conservative default (7%)
                volatility=0.15,    # Conservative default (15%)
                min_investment=1000  # Default minimum
            )
    
    return universe

# Initialize global variables
ETF_UNIVERSE: Dict[str, ETFMetrics] = {}
ETF_PORTFOLIOS: Dict[int, PortfolioAllocation] = {}

def initialize_portfolios() -> Dict[int, PortfolioAllocation]:
    """Initialize portfolio allocations with default values if needed"""
    try:
        return generate_portfolio_allocations()
    except Exception as e:
        print(f"Error generating portfolios: {str(e)}")
        # Return default portfolio allocations
        default_portfolios = {}
        risk_levels = np.linspace(0.03, 0.14, 10)
        for i, vol in enumerate(risk_levels, 1):
            default_portfolios[i] = PortfolioAllocation(
                name=f"Risk Level {i}",
                description=f"Default portfolio for risk level {i}",
                target_volatility=vol,
                target_return=vol * 0.5,  # Conservative estimate
                allocation={'SPY': 0.6, 'AGG': 0.4},  # Conservative default allocation
                sharpe_ratio=0.5  # Conservative default
            )
        return default_portfolios

def fetch_historical_data(start_date: str = '2019-01-01') -> pd.DataFrame:
    """Fetch historical data for ETF universe"""
    data = {}
    for ticker in ETF_UNIVERSE.keys():
        etf = yf.Ticker(ticker)
        hist = etf.history(start=start_date)['Close']
        data[ticker] = hist
    return pd.DataFrame(data)

def calculate_portfolio_metrics(returns: pd.DataFrame) -> tuple:
    """Calculate portfolio metrics"""
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    return annual_returns, annual_vol, cov_matrix

def optimize_portfolio(returns: pd.DataFrame, target_vol: float) -> Dict[str, float]:
    """Optimize portfolio for target volatility"""
    n_assets = len(returns.columns)
    annual_returns, _, cov_matrix = calculate_portfolio_metrics(returns)
    
    def objective(weights):
        port_return = np.sum(annual_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol  # Maximize Sharpe ratio
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda x: 
            np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_vol}  # Target volatility
    ]
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    result = minimize(
        objective, 
        np.array([1/n_assets] * n_assets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return dict(zip(returns.columns, result.x))

def generate_portfolio_allocations() -> Dict[int, PortfolioAllocation]:
    """Generate optimized portfolio allocations for each risk level"""
    try:
        if not ETF_UNIVERSE:
            print("Warning: ETF_UNIVERSE is empty when generating portfolios")
            return {}
            
        historical_data = fetch_historical_data()
        if historical_data.empty:
            print("Warning: No historical data available")
            return {}
            
        returns = historical_data.pct_change().dropna()
        
        portfolios = {}
        risk_levels = np.linspace(0.03, 0.14, 10)  # From 3% to 14% volatility
        
        for i, target_vol in enumerate(risk_levels, 1):
            allocation = optimize_portfolio(returns, target_vol)
            annual_returns, annual_vol, _ = calculate_portfolio_metrics(returns)
            exp_return = sum(allocation[k] * annual_returns[k] for k in allocation)
            
            portfolios[i] = PortfolioAllocation(
                name=f"Risk Level {i}",
                description=f"Optimized portfolio for {target_vol:.1%} volatility",
                target_volatility=target_vol,
                target_return=exp_return,
                allocation=allocation,
                sharpe_ratio=exp_return/target_vol
            )
        
        return portfolios
    except Exception as e:
        print(f"Error generating portfolio allocations: {str(e)}")
        return {}

def refresh_etf_universe() -> None:
    """Refresh ETF universe with current data"""
    try:
        global ETF_UNIVERSE, ETF_PORTFOLIOS
        print("Initializing ETF Universe...")
        ETF_UNIVERSE = initialize_etf_universe()
        print("ETF Universe initialized successfully")
        
        print("Generating portfolio allocations...")
        ETF_PORTFOLIOS = initialize_portfolios()
        print(f"Generated {len(ETF_PORTFOLIOS)} portfolios")
    except Exception as e:
        print(f"Error in refresh_etf_universe: {str(e)}")
        if not ETF_PORTFOLIOS:
            print("Initializing default portfolios...")
            ETF_PORTFOLIOS = initialize_portfolios()

# Initialize data
print("Starting initial data refresh...")
refresh_etf_universe()
print("Initial data refresh completed")
def validate_etf_universe() -> Dict[str, bool]:
    """Validate ETFs meet minimum criteria"""
    criteria = {
        'min_assets': 1.0,        # $1B minimum
        'min_volume': 1.0,        # 1M shares minimum
        'max_expense': 0.0075,    # 0.75% maximum
        'min_history': 3.0        # 3 years minimum
    }
    
    validation = {}
    for ticker, metrics in ETF_UNIVERSE.items():
        validation[ticker] = (
            metrics.net_assets >= criteria['min_assets'] and
            metrics.avg_volume >= criteria['min_volume'] and
            metrics.expense_ratio <= criteria['max_expense']
        )
    
    return validation

def fetch_historical_data(start_date: str = '2019-01-01') -> pd.DataFrame:
    """Fetch historical data for ETF universe"""
    data = {}
    for ticker in ETF_UNIVERSE.keys():
        etf = yf.Ticker(ticker)
        hist = etf.history(start=start_date)['Close']
        data[ticker] = hist
    return pd.DataFrame(data)

def calculate_portfolio_metrics(returns: pd.DataFrame) -> tuple:
    """Calculate portfolio metrics"""
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    cov_matrix = returns.cov() * 252
    return annual_returns, annual_vol, cov_matrix

def optimize_portfolio(returns: pd.DataFrame, target_vol: float) -> Dict[str, float]:
    """Optimize portfolio for target volatility"""
    n_assets = len(returns.columns)
    annual_returns, _, cov_matrix = calculate_portfolio_metrics(returns)
    
    def objective(weights):
        port_return = np.sum(annual_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol  # Maximize Sharpe ratio
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda x: 
            np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))) - target_vol}  # Target volatility
    ]
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    result = minimize(
        objective, 
        np.array([1/n_assets] * n_assets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return dict(zip(returns.columns, result.x))

def generate_portfolio_allocations() -> Dict[int, PortfolioAllocation]:
    """Generate optimized portfolio allocations for each risk level"""
    historical_data = fetch_historical_data()
    returns = historical_data.pct_change().dropna()
    
    portfolios = {}
    risk_levels = np.linspace(0.03, 0.14, 10)  # From 3% to 14% volatility
    
    for i, target_vol in enumerate(risk_levels, 1):
        allocation = optimize_portfolio(returns, target_vol)
        annual_returns, annual_vol, _ = calculate_portfolio_metrics(returns)
        exp_return = sum(allocation[k] * annual_returns[k] for k in allocation)
        
        portfolios[i] = PortfolioAllocation(
            name=f"Risk Level {i}",
            description=f"Optimized portfolio for {target_vol:.1%} volatility",
            target_volatility=target_vol,
            target_return=exp_return,
            allocation=allocation,
            sharpe_ratio=exp_return/target_vol
        )
    
    return portfolios

def get_etf_portfolio(risk_bucket: int) -> PortfolioAllocation:
    """Get portfolio allocation for given risk level"""
    try:
        print(f"ETF_PORTFOLIOS keys: {list(ETF_PORTFOLIOS.keys())}")
        print(f"Looking for risk bucket: {risk_bucket}")
        
        if not ETF_PORTFOLIOS:
            print("Warning: ETF_PORTFOLIOS is empty, refreshing data...")
            refresh_etf_universe()
            
        portfolio = ETF_PORTFOLIOS.get(risk_bucket)
        if portfolio is None:
            print(f"No portfolio found for risk bucket {risk_bucket}, using default")
            portfolio = ETF_PORTFOLIOS.get(5)  # Default to moderate risk
            
        if portfolio is None:
            raise ValueError(f"Could not find portfolio for risk bucket {risk_bucket} and no default available")
            
        print(f"Retrieved portfolio: {portfolio}")
        return portfolio
    except Exception as e:
        print(f"Error in get_etf_portfolio: {str(e)}")
        raise