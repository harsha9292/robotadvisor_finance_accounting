from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import pandas as pd

from etf_allocations import calculate_score, determine_risk_profile
from etf_allocations import get_etf_portfolio, ETF_DATA, refresh_etf_universe
from portfolio_analysis import generate_etf_report, backtest_portfolio

router = APIRouter()

@router.get("/refresh-data")
async def refresh_data():
    """Refresh ETF data and regenerate portfolios"""
    try:
        refresh_etf_universe()
        return {"message": "Data refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/etf-universe")
async def get_etf_universe_info():
    """Get information about all ETFs in the universe"""
    try:
        # Convert ETF_DATA to a flat dictionary format
        etf_info = {}
        for category in ETF_DATA.values():
            for etf in category:
                if etf.get('Ticker'):
                    etf_info[etf['Ticker']] = {
                        "name": etf['ETF_Name'],
                        "expense_ratio": f"{etf['expense_ratio']:.3%}",
                        "asset_type": etf['asset_type'],
                        "region": etf['region'],
                        "risk_level": etf['risk_level'],
                        "sustainability": etf['sustainability']
                    }
        return etf_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/etf-analysis")
async def get_etf_analysis():
    """Get comprehensive ETF analysis"""
    try:
        report = generate_etf_report()
        
        # Convert DataFrames to dictionaries for JSON response
        analysis = {
            'etf_characteristics': report['etf_characteristics'].to_dict(),
            'risk_mappings': report['risk_mappings'].to_dict(),
            'performance_metrics': report['performance_metrics']
        }
        
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio-backtest/{risk_level}")
async def backtest_risk_level(risk_level: int, start_date: str = "2019-01-01"):
    """Backtest portfolio performance for given risk level"""
    try:
        portfolio = get_etf_portfolio(risk_level)
        performance = backtest_portfolio(portfolio, start_date)
        
        return {
            "risk_level": risk_level,
            "portfolio_name": portfolio["name"],
            "cumulative_returns": performance.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommend/{bucket}")
async def recommend(bucket: int):
    """Get portfolio recommendation for a given risk bucket"""
    try:
        portfolio = get_etf_portfolio(bucket)
        return {
            "name": portfolio["name"],
            "description": portfolio["description"],
            "target_return": f"{portfolio['target_return']:.2%}",
            "target_volatility": f"{portfolio['target_volatility']:.2%}",
            "allocation": portfolio["allocation"],
            "sharpe_ratio": f"{portfolio['sharpe_ratio']:.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio-metrics/{risk_level}")
async def get_portfolio_metrics(risk_level: int):
    """Get detailed metrics for a specific portfolio"""
    try:
        portfolio = get_etf_portfolio(risk_level)
        report = generate_etf_report()
        metrics = report['performance_metrics'][risk_level]
        
        return {
            "portfolio_name": portfolio["name"],
            "metrics": {
                "expected_return": f"{metrics['return']:.2%}",
                "volatility": f"{metrics['volatility']:.2%}",
                "sharpe_ratio": f"{metrics['sharpe']:.2f}",
                "var_95": f"{metrics['var_95']:.2%}"
            },
            "allocation": portfolio["allocation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))