# ==========================================================
# market_service.py — Market API + BigQuery Robotaxi Milestones (FASTAPI ONLY)
# ==========================================================


import os
from typing import Optional, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import json
import yfinance as yf
from google.cloud import bigquery

# ----------------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------------
app = FastAPI(title="Market Data API (TSLA + Options + Robotaxi)")


# ----------------------------------------------------------
# CORS — REQUIRED for React frontend
# ----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
PROJECT_ID = "nfl-rag-project"
DATASET_ID = "tsla_data"
OPTIONS_TABLE = "tsla_option_chain_all"
ROBOTAXI_TABLE = "robotaxi_milestones"
FORECAST_TABLE = "forecast_growth"

bq = bigquery.Client(project=PROJECT_ID)


# ==========================================================
# 1. REAL-TIME PRICE
# ==========================================================
@app.get("/price")
def get_price(ticker: str):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m")

        if hist.empty:
            return JSONResponse({"error": "No price data"}, status_code=404)

        return {
            "ticker": ticker.upper(),
            "price": float(hist["Close"].iloc[-1]),
            "ts": hist.index[-1].isoformat(),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ==========================================================
# 2. 10-DAY 4H CANDLES
# ==========================================================
@app.get("/history")
def get_history(ticker: str):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="10d", interval="4h")

        if hist.empty:
            return JSONResponse({"error": "No history data"}, status_code=404)

        return {
            "ticker": ticker.upper(),
            "interval": "4h",
            "bars": hist.reset_index().to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ==========================================================
# 3. OPTION EXPIRIES
# ==========================================================
@app.get("/options/expiries")
def get_expiries():
    try:
        q = f"""
            SELECT DISTINCT CAST(expiry AS STRING) AS expiry
            FROM `{PROJECT_ID}.{DATASET_ID}.{OPTIONS_TABLE}`
            ORDER BY expiry
        """
        rows = list(bq.query(q))
        expiries = [row["expiry"] for row in rows]
        return {"expiries": expiries}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/robotaxi-comments")
async def get_robotaxi_comments():
    client = bigquery.Client(project=PROJECT_ID)

    query = """
        SELECT 
            source_type,
            source_name,
            author,
            DATE(date) AS date,
            comment,
            key_forecast,
            price_target,
            previous_price_target,
            upside_pct
        FROM `nfl-rag-project.tsla_data.tsla_robotaxi_comments`
        ORDER BY date DESC
    """

    rows = []
    for r in client.query(query).result():
        item = dict(r)

        # ---- FIX: Convert datetime.date -> str ----
        if "date" in item and item["date"] is not None:
            item["date"] = item["date"].strftime("%Y-%m-%d")

        rows.append(item)

    return JSONResponse(rows)

# ==========================================================
# 4. ROBOTAXI MILESTONES (REQUIRED FOR FLOWCHART)
# ==========================================================
@app.get("/api/robotaxi_milestones")
def robotaxi_milestones():
    try:
        q = f"""
            SELECT *
            FROM `{PROJECT_ID}.{DATASET_ID}.{ROBOTAXI_TABLE}`
            ORDER BY City, milestone_order
        """
        rows = list(bq.query(q))
        data = [dict(r) for r in rows]

        return {"status": "ok", "data": data}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ==========================================================
# 5. OPTION CHAIN (CALLS + PUTS)
# ==========================================================
@app.get("/options/chain")
def get_options_chain(
    ticker: str,
    expiration: Optional[str] = None,
    limit: int = 200
):
    ticker = ticker.upper()
    if ticker != "TSLA":
        return JSONResponse({"error": "Only TSLA supported"}, status_code=400)

    # load expiries
    exp_res = get_expiries()
    expiries = exp_res.get("expiries", [])

    if not expiries:
        return JSONResponse({"error": "No expiries found"}, status_code=404)

    if not expiration or expiration not in expiries:
        expiration = expiries[0]

    q = f"""
        SELECT
            Strike, `Last Price`, Bid, Mid, Ask,
            Volume, `Open Interest`, `Implied Volatility`,
            Delta, Gamma, Theta, Vega,
            type, CAST(expiry AS STRING) AS expiry,
            text_content
        FROM `{PROJECT_ID}.{DATASET_ID}.{OPTIONS_TABLE}`
        WHERE CAST(expiry AS STRING) = @exp
        ORDER BY type, Strike
        LIMIT {limit * 2}
    """

    job = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("exp", "STRING", expiration)]
    )

    try:
        rows = list(bq.query(q, job_config=job))

        calls = []
        puts = []

        for r in rows:
            entry = {
                "strike": float(r["Strike"]),
                "last": r["Last Price"],
                "bid": r["Bid"],
                "mid": r["Mid"],
                "ask": r["Ask"],
                "volume": r["Volume"],
                "open_interest": r["Open Interest"],
                "iv": r["Implied Volatility"],
                "delta": r["Delta"],
                "gamma": r["Gamma"],
                "theta": r["Theta"],
                "vega": r["Vega"],
                "type": r["type"],
                "expiry": r["expiry"],
                "text": r["text_content"],
            }

            if r["type"] == "call":
                calls.append(entry)
            else:
                puts.append(entry)

        return {
            "ticker": "TSLA",
            "expiration": expiration,
            "calls": calls[:limit],
            "puts": puts[:limit],
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ==========================================================
# 6. ATM SUMMARY
# ==========================================================
@app.get("/options/atm_summary")
def get_atm_summary(expiration: Optional[str] = None):
    try:
        t = yf.Ticker("TSLA")
        hist = t.history(period="1d", interval="1m")

        if hist.empty:
            return JSONResponse({"error": "spot unavailable"}, status_code=404)

        spot = float(hist["Close"].iloc[-1])

        chain = get_options_chain("TSLA", expiration=expiration, limit=400)
        calls = chain["calls"]
        puts = chain["puts"]

        def closest(items):
            return min(items, key=lambda x: abs(x["strike"] - spot)) if items else None

        atm_call = closest(calls)
        atm_put = closest(puts)

        atm_strike = atm_call["strike"] if atm_call else (
            atm_put["strike"] if atm_put else None
        )

        return {
            "ticker": "TSLA",
            "spot": spot,
            "atm_strike": atm_strike,
            "atm_iv_call": atm_call["iv"] if atm_call else None,
            "atm_iv_put": atm_put["iv"] if atm_put else None,
            "expiration": chain["expiration"]
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# ==========================================================
# 4B. ROBOTAXI TIMELINE (TABLE FOR FRONTEND TABLE VIEW)
# ==========================================================
@app.get("/robotaxi-timeline")
def robotaxi_timeline_table():
    """
    Returns raw robotaxi city-level milestone rows
    for the React RobotaxiTable.jsx.
    """
    try:
        q = f"""
            SELECT
                State,
                City,
                Population,
                Tesla_Insurance_Available,
                Permit_Applied,
                Permit_Received,
                Vehicle_Operated_Ads,
                Public_Test,
                GeoFence,
                Regulatory_Approval,
                scrape_timestamp
            FROM `{PROJECT_ID}.{DATASET_ID}.robotaxi_timeline`
            ORDER BY State, City
        """

        rows = list(bq.query(q))
        data = []

        for r in rows:
            data.append({
                "State": r.get("State"),
                "City": r.get("City"),
                "Population": r.get("Population"),
                "Tesla_Insurance_Available": _clean_date(r.get("Tesla_Insurance_Available")),
                "Permit_Applied": _clean_date(r.get("Permit_Applied")),
                "Permit_Received": _clean_date(r.get("Permit_Received")),
                "Vehicle_Operated_Ads": _clean_date(r.get("Vehicle_Operated_Ads")),
                "Public_Test": _clean_date(r.get("Public_Test")),
                "GeoFence": r.get("GeoFence"),
                "Regulatory_Approval": _clean_date(r.get("Regulatory_Approval")),
                "scrape_timestamp": str(r.get("scrape_timestamp")),
            })

        return JSONResponse(data)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# helper to convert None/datetime to string or "--"
def _clean_date(v):
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v)


# ==========================================================
# NEW: FORECAST GROWTH
# ==========================================================
@app.get("/api/forecast_growth")
def get_forecast_growth():
    try:
        q = f"""
            SELECT
                year,
                scenario,
                revenue_billions,
                production_units
            FROM `{PROJECT_ID}.{DATASET_ID}.{FORECAST_TABLE}`
            ORDER BY year, scenario
        """
        rows = list(bq.query(q))
        data = [dict(r) for r in rows]
        return {"data": data}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
@app.get("/api/montecarlo_robotaxi")
def get_montecarlo_robotaxi():
    """
    Returns long-form Monte Carlo simulations from mc_robotaxi_long_gemma.

    Table schema:
      sim_id, year, revenue, cumulative, robotaxi_works, launch_year,
      regulatory_risk, demand_multiplier, innovation_speed, competition_factor

    We keep:
      - Only runs where robotaxi_works = 1
      - Launch year <= 2026  (so these are "working" scenarios)
    """
    try:
        q = f"""
            SELECT
              sim_id,
              year,
              revenue,
              cumulative,
              robotaxi_works,
              launch_year,
              regulatory_risk,
              demand_multiplier,
              innovation_speed,
              competition_factor
            FROM `{PROJECT_ID}.{DATASET_ID}.mc_robotaxi_long_gemma`
            WHERE robotaxi_works = 1
              AND launch_year <= 2026
            ORDER BY sim_id, year
            LIMIT 100000
        """

        rows = [dict(r) for r in bq.query(q)]
        return {"data": rows}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# ==========================================================
# X FEED (TESLA TOP POSTS — INFINITE SCROLL)
# ==========================================================
@app.get("/api/x-feed")
def get_x_feed(limit: int = 40, offset: int = 0):
    """
    Streams Tesla posts from BigQuery with infinite-scroll support.
    Table schema:
      datetime, topic, tab, context
    """

    try:
        q = """
            SELECT
                CAST(datetime AS STRING) AS datetime,
                topic,
                tab,
                context
            FROM `nfl-rag-project.tsla_data.x_all_topics_top_latest`
            ORDER BY datetime DESC
            LIMIT @limit OFFSET @offset
        """

        job = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("offset", "INT64", offset)
            ]
        )

        rows = [dict(r) for r in bq.query(q, job_config=job).result()]

        return {
            "data": rows,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ROOT
@app.get("/")
def root():
    return {"status": "Market API active"}