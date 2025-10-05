# risk_scoring.py
"""
Risk scoring and profile determination logic
for Young Professionals RoboAdvisor (v1)
"""

RISK_MATRIX = {
    "bands": [
        {
            "profile": "Conservative",
            "score_min": 8,
            "score_max": 15,
            "target_volatility_pct_range": [0, 5],
            "typical_allocation_hint": {"equity_pct": 20, "bond_pct": 80}
        },
        {
            "profile": "Balanced",
            "score_min": 16,
            "score_max": 23,
            "target_volatility_pct_range": [5, 9],
            "typical_allocation_hint": {"equity_pct": 50, "bond_pct": 50}
        },
        {
            "profile": "Growth",
            "score_min": 24,
            "score_max": 31,
            "target_volatility_pct_range": [9, 13],
            "typical_allocation_hint": {"equity_pct": 70, "bond_pct": 30}
        },
        {
            "profile": "Aggressive",
            "score_min": 32,
            "score_max": 40,
            "target_volatility_pct_range": [13, 100],
            "typical_allocation_hint": {"equity_pct": 90, "bond_pct": 10}
        }
    ]
}


def calculate_score(data: dict) -> int:
    """
    Calculates the risk score from user questionnaire data.
    """
    score = 0

    # Example simple scoring logic — customize as needed
    score += data.get("risk_tolerance", 1) * 3
    score += {"1A": 2, "1B": 3, "1C": 4, "1D": 1, "1E": 2}.get(data.get("primary_goal"), 0)
    score += {"2A": 1, "2B": 2, "2C": 3, "2D": 4}.get(data.get("access_time"), 0)
    score += {"3A": 4, "3B": 3, "3C": 1}.get(data.get("income_stability"), 0)
    score += {"4A": 3, "4B": 2, "4C": 1}.get(data.get("emergency_fund"), 0)
    score += {"5A": 3, "5B": 2, "5C": 3, "5D": 1}.get(data.get("investment_plan"), 0)
    score += {"6A": 1, "6B": 2, "6C": 3}.get(data.get("initial_investment"), 0)
    score += {"7A": 1, "7B": 3, "7C": 5}.get(data.get("reaction_to_loss"), 0)
    score += {"8A": 1, "8B": 3, "8C": 5}.get(data.get("investing_experience"), 0)

    # Normalize to valid band range
    return max(8, min(score, 40))


def determine_risk_profile(score: int) -> dict:
    """
    Determines the risk profile and parameters for a given score.
    """
    for idx, band in enumerate(RISK_MATRIX["bands"], start=1):
        if band["score_min"] <= score <= band["score_max"]:
            return {
                "risk_bucket": idx,
                "risk_profile": band["profile"],
                "target_volatility_pct_range": band["target_volatility_pct_range"],
                "typical_allocation_hint": band["typical_allocation_hint"]
            }

    # Default fallback (Conservative)
    return {
        "risk_bucket": 1,
        "risk_profile": "Conservative",
        "target_volatility_pct_range": [0, 5],
        "typical_allocation_hint": {"equity_pct": 20, "bond_pct": 80}
    }
