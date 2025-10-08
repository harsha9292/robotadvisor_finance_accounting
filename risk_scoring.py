from typing import Dict, List, Any

"""
Risk scoring and profile determination logic
for Young Professionals RoboAdvisor (v3)
"""

RISK_MATRIX = {
    "bands": [
        {
            "profile": "Capital Preservation",
            "level": 1,
            "score_min": 6.00,
            "score_max": 7.19, # Adjusted for continuous range
            "typical_allocation_hint": {"equity_pct": 10, "bond_pct": 90},
        },
        {
            "profile": "Conservative",
            "level": 2,
            "score_min": 7.20,
            "score_max": 8.39,
            "typical_allocation_hint": {"equity_pct": 20, "bond_pct": 80},
        },
        {
            "profile": "Cautious Balanced",
            "level": 3,
            "score_min": 8.40,
            "score_max": 9.59,
            "typical_allocation_hint": {"equity_pct": 30, "bond_pct": 70},
        },
        {
            "profile": "Moderate",
            "level": 4,
            "score_min": 9.60,
            "score_max": 10.79,
            "typical_allocation_hint": {"equity_pct": 40, "bond_pct": 60},
        },
        {
            "profile": "Balanced Growth",
            "level": 5,
            "score_min": 10.80,
            "score_max": 11.99,
            "typical_allocation_hint": {"equity_pct": 50, "bond_pct": 50},
        },
        {
            "profile": "Growth",
            "level": 6,
            "score_min": 12.00,
            "score_max": 13.19,
            "typical_allocation_hint": {"equity_pct": 60, "bond_pct": 40},
        },
        {
            "profile": "Aggressive Growth",
            "level": 7,
            "score_min": 13.20,
            "score_max": 14.39,
            "typical_allocation_hint": {"equity_pct": 70, "bond_pct": 30},
        },
        {
            "profile": "Global Equity Focus",
            "level": 8,
            "score_min": 14.40,
            "score_max": 15.59,
            "typical_allocation_hint": {"equity_pct": 80, "bond_pct": 20},
        },
        {
            "profile": "High Growth",
            "level": 9,
            "score_min": 15.60,
            "score_max": 16.79,
            "typical_allocation_hint": {"equity_pct": 90, "bond_pct": 10},
        },
        {
            "profile": "Ultra-Aggressive",
            "level": 10,
            "score_min": 16.80,
            "score_max": 18.00, # Max possible score
            "typical_allocation_hint": {"equity_pct": 100, "bond_pct": 0},
        },
    ]
}


def calculate_score(data: dict) -> Dict[str, float]:
    """
    Calculate composite risk score (6–18) using weighted normalized subscores 
    and return the detailed breakdown.
    
    The normalization correctly assumes that all 15 questions can range from 1 to 3 
    (low to high risk).
    """
    # CRITICAL CHANGE: Removed all default values. This function now requires
    # the 'data' dictionary to contain valid scores (1-3) for all 15 keys
    # to avoid a KeyError or TypeError.

    # --- Attitude (6 questions, Min 6, Max 18, Range 12) ---
    attitude_qs = [
        data["age_range_score"], # Removed default
        data["investment_goal_score"], # Removed default
        data["market_loss_scenario_score"], # Removed default
        data["risk_vs_reward_score"], # Removed default
        data["feelings_about_volatility_score"], # Removed default
        data["investment_experience_score"], # Removed default
    ]
    attitude_score = sum(attitude_qs)
    attitude_norm = (attitude_score - 6) / 12  # Normalize 0–1

    # --- Capacity (7 questions, Min 7, Max 21, Range 14) ---
    capacity_qs = [
        data["income_stability_score"], # Removed default
        data["liabilities_score"], # Removed default
        data["emergency_fund_score"], # Removed default
        data["monthly_savings_score"], # Removed default
        data["initial_investment_score"], # Removed default
        data["investment_horizon_score"], # Removed default
        data["investment_approach_score"], # Removed default
    ]
    capacity_score = sum(capacity_qs)
    capacity_norm = (capacity_score - 7) / 14

    # --- Tech (2 questions, Min 2, Max 6, Range 4) ---
    tech_qs = [
        data["app_experience_score"], # Removed default
        data["ai_advisor_score"], # Removed default
    ]
    tech_score = sum(tech_qs)
    tech_norm = (tech_score - 2) / 4

    # --- Weighted composite (0 to 1) ---
    composite_norm = (
        attitude_norm * 0.5 + capacity_norm * 0.3 + tech_norm * 0.2
    )

    # Scale to 6–18 (minimum realistic score is 6)
    composite_scaled = 6 + (composite_norm * 12)

    return {
        "final_composite_score": round(composite_scaled, 2),
        "attitude_norm_score": round(attitude_norm, 4),
        "capacity_norm_score": round(capacity_norm, 4),
        "tech_norm_score": round(tech_norm, 4),
        "weighted_composite_norm": round(composite_norm, 4)
    }


def determine_risk_profile(data: dict) -> dict:
    """
    Calculates the composite score, maps it to a risk bucket, and returns 
    the full profile along with the normalized score breakdown.
    """
    # Calculate all necessary score components and get the breakdown
    score_data = calculate_score(data)
    score = score_data['final_composite_score']
    
    risk_bucket = RISK_MATRIX["bands"][0]["level"] # Default to lowest
    risk_profile = RISK_MATRIX["bands"][0]["profile"] # Default profile
    band_found = RISK_MATRIX["bands"][0]

    # Determine risk profile and bucket
    for band in RISK_MATRIX["bands"]:
        # Check if the score falls within the band's range
        if band["score_min"] <= score <= band["score_max"]:
            risk_bucket = band["level"]
            risk_profile = band["profile"]
            band_found = band
            break
    
    # Fallback for out-of-range scores (shouldn't happen with 6.00-18.00 range)
    if score < RISK_MATRIX["bands"][0]["score_min"]:
        band_found = RISK_MATRIX["bands"][0]
    elif score > RISK_MATRIX["bands"][-1]["score_max"]:
        band_found = RISK_MATRIX["bands"][-1]

    # Calculate target volatility range
    eq = band_found["typical_allocation_hint"]["equity_pct"]
    # Volatility calculation based on Equity %: Low = Eq/10 - 2, High = Eq/10 + 2
    vol_low = round(eq / 10 - 2, 1)
    vol_high = round(eq / 10 + 2, 1)

    return {
        "risk_bucket": band_found["level"],
        "risk_profile": band_found["profile"],
        "target_volatility_pct_range": [vol_low, vol_high],
        "typical_allocation_hint": band_found["typical_allocation_hint"],
        # Add the detailed score breakdown
        "final_composite_score": score,
        "attitude_norm_score": score_data['attitude_norm_score'],
        "capacity_norm_score": score_data['capacity_norm_score'],
        "tech_norm_score": score_data['tech_norm_score'],
        "weighted_composite_norm": score_data['weighted_composite_norm']
    }
