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

    # Example simple scoring logic — align mapping keys with frontend form values
    score += data.get("risk_tolerance", 1) * 3

    # primary_goal uses single-letter values 'a'..'e' in the frontend
    score += {"a": 2, "b": 3, "c": 4, "d": 1, "e": 2}.get(str(data.get("primary_goal", "")).lower(), 0)

    # access_time mapping ('a'..'d')
    score += {"a": 1, "b": 2, "c": 3, "d": 4}.get(str(data.get("access_time", "")).lower(), 0)

    # income_stability ('a'..'c')
    score += {"a": 4, "b": 3, "c": 1}.get(str(data.get("income_stability", "")).lower(), 0)

    # emergency_fund ('a'..'c')
    score += {"a": 3, "b": 2, "c": 1}.get(str(data.get("emergency_fund", "")).lower(), 0)

    # investment_plan ('a'..'d')
    score += {"a": 3, "b": 2, "c": 3, "d": 1}.get(str(data.get("investment_plan", "")).lower(), 0)

    # initial_investment ('a'..'c')
    score += {"a": 1, "b": 2, "c": 3}.get(str(data.get("initial_investment", "")).lower(), 0)

    # reaction_to_loss may be a word like 'hold' or similar from frontend; map common values
    rt = str(data.get("reaction_to_loss", "")).lower()
    score += {"hold": 3, "sell": 1, "buy_more": 5}.get(rt, 0)

    # investing_experience may be 'moderate', 'beginner', 'advanced'
    ie = str(data.get("investing_experience", "")).lower()
    score += {"beginner": 1, "moderate": 3, "advanced": 5}.get(ie, 0)

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
