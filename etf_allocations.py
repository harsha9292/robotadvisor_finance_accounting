from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
# Note: yfinance and other imports are commented out as they were not used in the logic provided.
from datetime import datetime, timedelta
import yfinance as yf


# Global ETF storage
ETF_DATA: Dict[str, list] = {}

# -----------------------------
# Helper: clean ticker
# -----------------------------
def clean_ticker(raw_ticker: str) -> Optional[str]:
    """
    Extracts valid ticker from 'ETF / ISIN' format.
    Returns None if ticker is empty or malformed.
    """
    if not raw_ticker:
        return None
    parts = raw_ticker.split("/")
    ticker = parts[0].strip()
    return ticker if ticker else None

# -----------------------------
# Load ETF data from CSV
# -----------------------------
def load_etf_data(path: str = "etf_data.csv"):
    """
    Load ETF data from CSV into global ETF_DATA dictionary.
    Cleans tickers for yfinance and maintains both DataFrame and dictionary representations.
    """
    global ETF_DATA
    try:
        # Load and clean DataFrame
        df = pd.read_csv(path)
        
        # Ensure correct types
        if 'Expense_Ratio' in df.columns:
            df['Expense_Ratio'] = df['Expense_Ratio'].astype(float)
        if 'Risk_Level' in df.columns:
            df['Risk_Level'] = df['Risk_Level'].astype(int)
            
        # Process DataFrame into categorized dictionary
        ETF_DATA.clear()
        for _, row in df.iterrows():
            cat = row['Category']
            ticker = clean_ticker(row.get("Ticker_ISIN", "_"))
            isin = None
            if "/" in row.get("Ticker_ISIN", ""):
                parts = row["Ticker_ISIN"].split("/")
                if len(parts) > 1:
                    isin = parts[1].strip() or None
            
            etf_entry = {
                "risk_level": row["Risk_Level"],
                "ETF_Name": row["ETF_Name"],
                "region": row["Region"],
                "asset_type": row["Asset_Type"],
                "description": row["Description"],
                "Ticker": ticker,
                "ISIN": isin,
                "expense_ratio": row["Expense_Ratio"],
                "typical_use": row["Typical_Use"],
                "sustainability": row.get("Sustainability", "no")
            }
            
            ETF_DATA.setdefault(cat, []).append(etf_entry)
            
        print(f"✅ Loaded {len(df)} ETFs across {len(ETF_DATA)} categories.")
        return df  # Return DataFrame for any code that needs it
        
    except FileNotFoundError:
        print(f"Error: ETF data file not found at {path}. ETF_DATA is empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading ETF data: {e}")
        return pd.DataFrame()

# -----------------------------
# Refresh ETF universe
# -----------------------------
def refresh_etf_universe(path: str = "etf_data.csv"):
    """
    Reload ETF data from CSV (wrapper for load_etf_data)
    """
    load_etf_data(path)

# -----------------------------
# Risk scoring helpers
# -----------------------------
# Implemented the user's 10-bucket profiles, but adjusted scores to be continuous
# across the full 6.00-18.00 range to ensure every score maps correctly.
RISK_MATRIX = [
    {"risk_bucket": 1, "profile": "Capital Preservation", "score_min": 6.00, "score_max": 7.20},
    {"risk_bucket": 2, "profile": "Conservative", "score_min": 7.21, "score_max": 8.40},
    {"risk_bucket": 3, "profile": "Cautious Balanced", "score_min": 8.41, "score_max": 9.60},
    {"risk_bucket": 4, "profile": "Moderate", "score_min": 9.61, "score_max": 10.80},
    {"risk_bucket": 5, "profile": "Balanced Growth", "score_min": 10.81, "score_max": 12.00},
    {"risk_bucket": 6, "profile": "Growth", "score_min": 12.01, "score_max": 13.20},
    {"risk_bucket": 7, "profile": "Aggressive Growth", "score_min": 13.21, "score_max": 14.40},
    {"risk_bucket": 8, "profile": "Global Equity Focus", "score_min": 14.41, "score_max": 15.60},
    {"risk_bucket": 9, "profile": "High Growth", "score_min": 15.61, "score_max": 16.80},
    {"risk_bucket": 10, "profile": "Ultra-Aggressive", "score_min": 16.81, "score_max": 18.00},
]

# Allocation hint for the front-end display, mapped by risk_bucket
# Interpolated 10-bucket allocation (Equity/Bond split)
ALLOCATION_HINTS = {
    1: {"equity_pct": 10, "bond_pct": 90},  
    2: {"equity_pct": 20, "bond_pct": 80},
    3: {"equity_pct": 30, "bond_pct": 70},
    4: {"equity_pct": 40, "bond_pct": 60},
    5: {"equity_pct": 50, "bond_pct": 50}, # Midpoint
    6: {"equity_pct": 60, "bond_pct": 40},
    7: {"equity_pct": 70, "bond_pct": 30},
    8: {"equity_pct": 80, "bond_pct": 20},
    9: {"equity_pct": 90, "bond_pct": 10},
    10: {"equity_pct": 100, "bond_pct": 0}, # Max Aggressive
}


# -----------------------------
# Risk & ETF selection functions
# -----------------------------
def calculate_score(data: dict) -> Dict[str, float]:
    """
    Calculates a composite risk score based on attitude, capacity, and tech scores,
    and returns the final score along with normalized components.
    
    CRITICAL: This function requires 'data' to contain valid scores (1-3) for all 15 keys 
    to prevent a KeyError. No default values are used.
    """
    # The function now uses direct dictionary access (data[key]), relying on the caller 
    # to guarantee all 15 keys are present.
    
    # --- Attitude (6 questions, Min 6, Max 18, Range 12) ---
    attitude_qs = [data[k] for k in [
        "age_range_score","investment_goal_score","market_loss_scenario_score",
        "risk_vs_reward_score","feelings_about_volatility_score","investment_experience_score"
    ]]
    attitude_score = sum(attitude_qs)
    attitude_norm = (attitude_score - 6) / 12
    
    # --- Capacity (7 questions, Min 7, Max 21, Range 14) ---
    capacity_qs = [data[k] for k in [
        "income_stability_score","liabilities_score","emergency_fund_score",
        "monthly_savings_score","initial_investment_score","investment_horizon_score","investment_approach_score"
    ]]
    capacity_score = sum(capacity_qs)
    # Denominator is 14 (21-7) for full Capacity range
    capacity_norm = (capacity_score - 7) / 14 
    
    # --- Tech (2 questions, Min 2, Max 6, Range 4) ---
    tech_qs = [data[k] for k in ["app_experience_score","ai_advisor_score"]] 
    tech_score = sum(tech_qs)
    # Denominator is 4 (6-2) for full Tech range
    tech_norm = (tech_score - 2) / 4
    
    # Weighted composite score (0 to 1)
    composite_norm = attitude_norm*0.5 + capacity_norm*0.3 + tech_norm*0.2
    
    # Scale from [0, 1] to [6, 18]
    score = (composite_norm * 12) + 6
    
    # Return the full score breakdown
    return {
        "score": round(score, 2),
        "attitude_norm": round(attitude_norm, 4),
        "capacity_norm": round(capacity_norm, 4),
        "tech_norm": round(tech_norm, 4),
        "composite_norm": round(composite_norm, 4)
    }


import numpy as np

def compute_portfolio_metrics(returns: np.ndarray, weights: np.ndarray = None):
    """
    Compute annualized expected return and volatility using daily returns.
    Assumes 252 trading days per year.
    """
    n_assets = returns.shape[1]
    if weights is None:
        weights = np.ones(n_assets) / n_assets  # equal weighting

    # Daily portfolio returns
    daily_portfolio_returns = returns @ weights

    # Annualized return: (1 + mean_daily)^252 - 1
    annual_return = (1 + np.mean(daily_portfolio_returns))**252 - 1

    # Annualized volatility: std_daily * sqrt(252)
    annual_volatility = np.std(daily_portfolio_returns) * np.sqrt(252)

    return {
        "expected_return": annual_return,
        "volatility": annual_volatility
    }


def map_risk_bucket(volatility: float, behavioral_score: float = 0.0, vol_bands: list = None) -> int:
    """
    Map portfolio volatility (and optionally behavioral score) to a risk bucket 1–10.
    behavioral_score: normalized 0–1, can adjust bucket slightly
    """
    if vol_bands is None:
        # Default 10 buckets (adjust as needed)
        vol_bands = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 1.0]

    # Base bucket from volatility
    base_bucket = 1
    for i, v_max in enumerate(vol_bands, start=1):
        if volatility <= v_max:
            base_bucket = i
            break

    # Adjust bucket based on behavioral score (scale 0–2)
    score_adjustment = int(np.round(behavioral_score * 2))
    hybrid_bucket = min(max(base_bucket + score_adjustment, 1), 10)  # clamp 1–10

    return hybrid_bucket


def determine_risk_profile(data: dict, historical_returns: np.ndarray, weights: np.ndarray = None):
    """
    Calculates the hybrid risk profile using behavioral scores and mean-variance portfolio metrics.
    """
    # Step 1: Calculate normalized behavioral scores
    score_data = calculate_score(data)
    composite_norm = score_data['composite_norm']  # normalized 0–1

    # Step 2: Compute annualized portfolio metrics
    port_metrics = compute_portfolio_metrics(historical_returns, weights)

    # Step 3: Map hybrid risk bucket using volatility + behavioral score
    risk_bucket = map_risk_bucket(port_metrics['volatility'], composite_norm)

    # Step 4: Map bucket to profile
    risk_profile = RISK_MATRIX[risk_bucket-1]['profile']  # assuming RISK_MATRIX[0] = bucket 1

    # Step 5: Allocation hint
    allocation_hint = ALLOCATION_HINTS.get(risk_bucket, {"equity_pct": 50, "bond_pct": 50})

    return {
        "risk_bucket": risk_bucket,
        "risk_profile": risk_profile,
        "typical_allocation_hint": allocation_hint,
        "final_composite_score": score_data['score'],
        "attitude_norm_score": score_data['attitude_norm'],
        "capacity_norm_score": score_data['capacity_norm'],
        "tech_norm_score": score_data['tech_norm'],
        "weighted_composite_norm": composite_norm,
        "portfolio_expected_return": port_metrics['expected_return'],
        "portfolio_volatility": port_metrics['volatility']
    }


def get_filtered_etfs(preferences: dict) -> List[Dict[str, Any]]:
    """
    Advanced ETF filtering function that combines the functionality of get_etf_candidates
    and get_dynamic_etfs. Filters ETFs based on multiple criteria.
    
    Args:
        preferences: dict containing filter criteria:
            - region_pref: str ("global", "eu", "us", etc.)
            - sustainable_pref: str ("yes", "no")
            - sectors_to_avoid: List[str] (optional)
            - asset_type: str (optional)
            - max_expense_ratio: float (optional)
            - risk_level: int (optional)
    
    Returns:
        List of ETF dictionaries matching the criteria
    """
    # Start with all ETFs
    all_etfs = [etf for cat in ETF_DATA.values() for etf in cat]
    filtered_etfs = []
    
    # Normalize region preference
    region_pref = preferences.get("region_pref", "global").lower().strip()
    region_mapping = {
        "europe (eu)": "europe",
        "u.s. (us)": "us",
        "no preference": "global"
    }
    region_pref = region_mapping.get(region_pref, region_pref)
    
    # Normalize sustainability preference
    sustainable_pref = preferences.get("sustainable_pref", "no").lower().strip()
    
    # Get excluded sectors
    sectors_to_avoid = [s.lower().strip() for s in preferences.get("sectors_to_avoid", [])]
    
    # Filter ETFs
    for etf in all_etfs:
        # Skip if region doesn't match (unless global)
        if region_pref != "global" and not etf["region"].lower().startswith(region_pref):
            continue
            
        # Skip if sustainability preference doesn't match
        if sustainable_pref == "yes" and etf.get("sustainability", "no").lower() != "yes":
            continue
            
        # Skip if sector should be avoided
        if any(sector.lower() in etf["asset_type"].lower() for sector in sectors_to_avoid):
            continue
            
        # Optional filters
        if "asset_type" in preferences and etf["asset_type"].lower() != preferences["asset_type"].lower():
            continue
            
        if "max_expense_ratio" in preferences and etf["expense_ratio"] > preferences["max_expense_ratio"]:
            continue
            
        if "risk_level" in preferences and etf["risk_level"] != preferences["risk_level"]:
            continue
            
        filtered_etfs.append(etf)
    
    return filtered_etfs

# -----------------------------
# Safe yfinance fetch
# -----------------------------
def get_live_performance(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch current price and 1-day % change for tickers.
    Skips invalid/malformed tickers.
    """
    if not tickers:
        return {}
    
    performance_data = {}
    for ticker in tickers:
        if not ticker:
            continue
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            current_price = info.get("currentPrice")
            previous_close = info.get("previousClose")
            daily_change_pct = None
            if current_price and previous_close and previous_close != 0:
                daily_change_pct = (current_price - previous_close) / previous_close * 100
            performance_data[ticker] = {
                "price": current_price,
                "daily_change_pct": round(daily_change_pct, 2) if daily_change_pct is not None else None
            }
        except Exception:
            performance_data[ticker] = {"price": None, "daily_change_pct": None}
    return performance_data


# -----------------------------
# Get portfolio by risk bucket
# -----------------------------
def get_etf_portfolio(risk_bucket: int, preferences: Optional[Dict[str, Any]] = None) -> dict:
    """
    Returns equal-weighted ETF portfolio with live performance.
    
    Args:
        risk_bucket: int (1-10) indicating risk level
        preferences: Optional dict of ETF filter preferences
    """
    # Get filtered ETFs based on preferences
    if preferences:
        preferences["risk_level"] = 1 if risk_bucket <= 3 else (2 if risk_bucket <= 7 else 3)
        source_list = get_filtered_etfs(preferences)
    else:
        source_list = [e for cat in ETF_DATA.values() for e in cat]
    
    # Map 10-bucket risk to 3-level ETF risk
    etfs = []
    for e in source_list:
        level = 1
        if 4 <= risk_bucket <= 7:
            level = 2
        elif risk_bucket >= 8:
            level = 3
        if e.get("risk_level") == level:
            etfs.append(e)
    
    if not etfs:
        return {}

    # Fetch live data
    tickers_list = [e["Ticker"] for e in etfs if e.get("Ticker")]
    live_performance = get_live_performance(tickers_list)
    
    n = len(etfs)
    allocation = {}
    etf_details = []
    
    for e in etfs:
        name = e.get("ETF_Name")
        ticker = e.get("Ticker")
        if name:
            allocation[name] = round(1/n, 4)
            live_data = live_performance.get(ticker, {"price": None, "daily_change_pct": None})
            enriched_etf = e.copy()
            enriched_etf.update({
                "current_price": live_data["price"],
                "daily_change_pct": live_data["daily_change_pct"],
                "allocation_pct": round(1/n * 100, 2)
            })
            etf_details.append(enriched_etf)

    # Interpolate target return & volatility
    target_r_min, target_r_max = 0.02, 0.12
    target_v_min, target_v_max = 0.03, 0.18
    target_r = target_r_min + (risk_bucket - 1) * (target_r_max - target_r_min) / 9
    target_v = target_v_min + (risk_bucket - 1) * (target_v_max - target_v_min) / 9

    return {
        "name": f"Risk Bucket {risk_bucket} Portfolio",
        "description": f"Equal-weighted ETF portfolio for risk bucket {risk_bucket}",
        "allocation": allocation,
        "etf_details": etf_details,
        "target_return": round(target_r, 4),
        "target_volatility": round(target_v, 4),
        "sharpe_ratio": round((target_r - 0.01) / target_v, 4)
    }

# -----------------------------
# Historical data functions
# -----------------------------
def get_historical_returns(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Fetch and calculate historical returns for given tickers.
    
    Args:
        tickers: List of ticker symbols
        period: Time period (e.g., "1y", "max", "5y")
        
    Returns:
        DataFrame of daily returns
    """
    if not tickers:
        return pd.DataFrame()
        
    # Download all tickers at once
    try:
        data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True)
        
        if len(tickers) == 1:
            # single ticker returns a Series
            data = data.to_frame()
            data.columns = [tickers[0]]
            returns = data.pct_change().dropna()
        else:
            # multiple tickers: extract 'Adj Close' or adjusted prices directly
            try:
                adj_close = pd.DataFrame({t: data[t]['Close'] for t in tickers})
                returns = adj_close.pct_change().dropna()
            except Exception as e:
                print(f"Error processing returns: {e}")
                returns = pd.DataFrame()
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        returns = pd.DataFrame()
    
    return returns

# -----------------------------
# Initial load
# -----------------------------
load_etf_data("etf_data.csv")