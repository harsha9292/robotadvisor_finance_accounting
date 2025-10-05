# etf_allocations.py

ETF_PORTFOLIOS = {
    1: {"name": "Capital Preservation", "allocation": {"AGG": 70, "BNDX": 30}},
    2: {"name": "Conservative Income", "allocation": {"AGG": 60, "BNDX": 20, "SPY": 20}},
    3: {"name": "Cautious Balanced", "allocation": {"AGG": 50, "BNDX": 20, "SPY": 20, "EFA": 10}},
    4: {"name": "Moderate Balanced", "allocation": {"AGG": 40, "SPY": 20, "EFA": 20, "BNDX": 20}},
    5: {"name": "Balanced Growth", "allocation": {"SPY": 35, "EFA": 20, "EEM": 15, "AGG": 30}},
    6: {"name": "Growth Tilted", "allocation": {"SPY": 40, "EFA": 20, "EEM": 20, "AGG": 20}},
    7: {"name": "Aggressive Growth", "allocation": {"SPY": 45, "EFA": 25, "EEM": 20, "AGG": 10}},
    8: {"name": "Global Equity Focus", "allocation": {"SPY": 50, "EFA": 25, "EEM": 25}},
    9: {"name": "High Growth", "allocation": {"SPY": 40, "EFA": 20, "EEM": 40}},
    10: {"name": "Ultra-Aggressive", "allocation": {"SPY": 60, "EFA": 20, "EEM": 20}},
}


def get_etf_portfolio(risk_bucket: int):
    """
    Returns the ETF portfolio allocation for a given risk bucket (1-10)
    """
    return ETF_PORTFOLIOS.get(risk_bucket, {"name": "Unknown", "allocation": {}})
