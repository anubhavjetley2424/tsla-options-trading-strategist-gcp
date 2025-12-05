# ==========================================================
# main.py — TSLA Options RAG + Strategy Agent (V19)
# - 2025 Vertex SDK compliant
# - Integrates real TSLA options chain by expiry
# - RAG (Qdrant) feeds catalysts, forecasts & strategy logic
# - Outputs richer JSON: payoff curves + greek time series
# ==========================================================

import os
import json
import traceback
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
import vertexai

from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from qdrant_client import QdrantClient
from openai import OpenAI  # NEW: xAI Grok via OpenAI-compatible SDK



# ==========================================================
# CONFIG
# ==========================================================

PROJECT_ID = os.environ.get("PROJECT_ID", "nfl-rag-project")
VERTEX_REGION = os.environ.get("VERTEX_REGION", "us-central1")

# Market microservice
MARKET_API = os.environ.get(
    "MARKET_API",
    "https://market-data-358918971535.us-central1.run.app",
)


XAI_API_KEY = os.environ.get("XAI_API_KEY")

xai_client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",  # xAI Grok endpoint
)

# Gemini models (2025 names)
LLM_AGENT_MODEL = os.environ.get("LLM_AGENT_MODEL", "gemini-2.5-pro")
LLM_CATALYST_MODEL = os.environ.get("LLM_CATALYST_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-004")

# Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "tsla_rag_v2")

# Options math
CONTRACT_MULTIPLIER = float(os.environ.get("CONTRACT_MULTIPLIER", "100"))
MAX_STRATEGIES_DEFAULT = int(os.environ.get("MAX_STRATEGIES", "3"))


# ==========================================================
# INIT
# ==========================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)
embed_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
bigquery_client = bigquery.Client(project=PROJECT_ID)

try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL and QDRANT_API_KEY else None
except Exception:
    qdrant = None


# ==========================================================
# UNIVERSAL SAFE LLM WRAPPER — 2025 GEMINI SDK
# ==========================================================

def call_llm(
    system_prompt: str,
    user_payload: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Wrap the 2025 Vertex AI Gemini SDK.
    We fuse system + user as a single user role with <system> tag.
    """
    try:
        fused = f"<system>{system_prompt}</system>\n\n{user_payload}"

        contents = [
            {
                "role": "user",
                "parts": [{"text": fused}],
            }
        ]

        model = GenerativeModel(model_name)
        resp = model.generate_content(
            contents=contents,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        return resp.text or ""
    except Exception as e:
        print("LLM ERROR:", e)
        return ""


def call_grok_chat(
    system_prompt: str,
    user_payload: str,
    model: str = "grok-2-latest",
    temperature: float = 0.15,
    max_tokens: int = 2048,
) -> str:
    """
    Thin wrapper around xAI Grok chat API.
    Uses an explicit system + user message structure.
    """
    if not XAI_API_KEY:
        print("⚠️ No XAI_API_KEY set; skipping Grok call.")
        return ""

    try:
        resp = xai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print("GROK ERROR:", e)
        return ""

# ==========================================================
# SYSTEM PROMPTS
# ==========================================================


GROK_CATALYST_SYSTEM = """
You are Grok-Catalyst-Collator (TSLA + MAG7 + Macro).

Input:
- RAG snippets (news, CPI, Truflation, Fed, MAG7 earnings, TSLA catalysts)
- Current TSLA spot
- Options expiry
- Raw catalysts from Agent 2B

Task:
1. Collate SHORT-TERM (≤45 days) catalysts across:
   - Macro: CPI, PPI, Fed, Truflation, NFP, PMI, earnings season
   - TSLA: Deliveries, FSD / Robotaxi, earnings, guidance changes
   - MAG7: Earnings beats/misses, AI capex, big regulatory news
2. For each event, assign:
   - event
   - date (YYYY-MM-DD or "TBD")
   - impact_score (float from -10 to +10)
   - probability (0.0–1.0)
   - confirmed (true/false based on cross-source agreement)
   - sources (list of short source labels)
   - tsla_impact (1–2 line description)
3. Cross-check for hallucinations:
   - If RAG context doesn’t support a detail, lower probability and flag confirmed=false.
4. Output STRICT JSON ONLY, as a list of events.
"""

AGENT_2B_SYSTEM = """
You are a TSLA macro + micro catalyst interpreter.

You are given:
- RAG context snippets (news, filings, transcripts, macro commentary, sentiment)
- Basic TSLA spot price information
- A chosen option expiry date

Your job:
1. Extract a clean list of catalysts that may affect TSLA into or around the given expiry.
2. For each catalyst, assign:
   - event_name
   - expected_date (YYYY-MM-DD if possible, else "TBD")
   - confidence (low / medium / high)
   - recommended_expiry (which listed options expiry is best aligned)
   - trigger_signals (list of short phrases; no quotes inside)
   - source_evidence (short sentence, no internal double quotes)

Be grounded and realistic. If the RAG snippets indicate known events
like earnings, deliveries, FSD/robotaxi announcements, macro prints
(CPI/Fed), or AI/autonomy showcases, surface them explicitly.

Return STRICT JSON ONLY:
[
  {
    "event_name": "Q4 Earnings",
    "expected_date": "2025-01-24",
    "confidence": "high",
    "recommended_expiry": "2025-02-14",
    "trigger_signals": ["volume expansion on green days", "options IV spike into event"],
    "source_evidence": "Earnings date inferred from historical TSLA reporting cadence."
  }
]
"""

AGENT_2A_SYSTEM = """
You are a TSLA price forecaster whose output directly drives the payoff
curve and max gain/loss calculations of an options strategy engine.

You are given:
- Current TSLA spot price
- A specific options expiry date
- A structured catalyst list extracted from RAG context (macro + micro)
- Raw RAG context snippets (news, macro, positioning, sentiment, options color)

Your job:
Forecast TSLA's BEAR, BASE, and BULL underlying price levels
AT THE EXPIRY DATE.

Important:
- These are scenario-bound forecasts, not deterministic predictions.
- The strategy engine will compute PnL using these values.
- The BULL case defines the upper bound of maximum profit.
- The BEAR case defines maximum loss (for long-premium positions).
- The BASE case should be your statistically most likely price.

Constraints:
- BEAR should represent realistic downside based on macro drag, positioning,
  funding conditions, and negative catalyst outcomes.
- BASE should reflect fair-value equilibrium based on typical TSLA behaviour
  into that expiry window, given the catalyst deck and macro context.
- BULL should reflect optimistic but plausible upside driven by catalysts,
  sentiment, or volatility compression (not meme / fantasy levels).
- Your numbers MUST be consistent: bear < base < bull.
- Be numerically sane relative to the current spot price and horizon.
  Avoid 2–3x moves in a very short window unless RAG context strongly supports it.

You are not running a Monte Carlo simulation.
Instead, you are defining three anchor scenarios that will be used
to evaluate an options payoff curve and max gain/loss.

Return STRICT JSON ONLY:
{
  "bear": <float>,
  "base": <float>,
  "bull": <float>
}
"""

AGENT_2C_SYSTEM = """
You are a TSLA event screener and risk analyst.

You are given:
- A raw list of catalysts (events) previously extracted from RAG context.
- The underlying TSLA spot price and a specific options expiry.
- The raw RAG snippets used to derive those events.

Your job:
1. Decide which events are REAL "needle movers" for TSLA around the given expiry.
   Needle movers are events that can plausibly move TSLA by ~3–5% or more in
   a short window (days to weeks), or shift the medium-term valuation narrative.
2. De-emphasize or drop:
   - trivial headlines
   - recycled commentary
   - minor analyst notes without size or credibility
   - generic macro chatter with weak linkage to TSLA
3. For each event you KEEP, add:
   - is_needle_mover: true/false
   - impact_direction: "bullish" / "bearish" / "mixed" / "unclear"
   - impact_magnitude: "small" / "medium" / "large"
   - reasoning: short sentence explaining why it's a needle mover or not.

If the original list has no strong events, you may keep a very small
set (or even an empty list) and explain via the reasoning fields.

Return STRICT JSON ONLY as a list:
[
  {
    "event_name": "...",
    "expected_date": "YYYY-MM-DD or TBD",
    "confidence": "low|medium|high",
    "recommended_expiry": "YYYY-MM-DD or explicit listed expiry",
    "trigger_signals": ["..."],
    "source_evidence": "...",
    "is_needle_mover": true,
    "impact_direction": "bullish",
    "impact_magnitude": "large",
    "reasoning": "Short justification"
  }
]
"""

AGENT_3A_SYSTEM = """
You are a TSLA options strategist.

You do NOT output JSON.
You output only a clear, concise natural-language planning document.

You are given:
- User intent (query, starting_capital, wealth_goal, risk_level, target_date)
- TSLA spot price
- A chosen options expiry (aligned as close as possible to target_date)
- A compact TSLA options chain for that expiry (calls and puts; each with strike, mid price, bid, ask, iv, volume, open interest)
- Catalyst list and weights (already derived from RAG + Grok)
- Forecasted TSLA prices at that expiry (bear/base/bull)
- Raw RAG context snippets summarizing TSLA macro + micro backdrop
- A `chain_intel` object that summarizes:
  - total call/put open interest and volume,
  - put–call ratios,
  - top open-interest and top-volume strikes,
  - rough IV buckets for calls and puts,
  - expected_move_estimate,
  - days_to_expiry.
- A `real_time_risk` object that includes:
  - gamma_squeeze_risk (0–1)
  - iv_crush_risk (0–1)
  - liquidity_shock_risk (0–1)
  - macro_risk (0–1)
  - composite_score (0–1)

Use these to:
- Prefer more liquid strikes and expiries.
- Recognize crowded positioning (high OI walls near spot).
- Detect where options look cheap or expensive (IV vs catalysts).
- Align strike selection with liquidity and gamma/volume clusters.
- Adjust strategy TYPE and POSITION SIZING based on real_time_risk:
  - High gamma_squeeze_risk + bullish backdrop → favour directional call structures, modestly larger size.
  - High iv_crush_risk → prefer spreads or structures that SELL expensive IV rather than long naked premium.
  - High liquidity_shock_risk or macro_risk → reduce size, consider hedges (puts, collars, spreads).

Your job:
1. Decide how many strategies to propose (1–3 is typical, but respect explicit user
   requests like "give me 4 strategies").
2. For each strategy, decide:
   - strategy_name
   - term label ("short-term", "swing", "long-term", etc.)
   - Which actual option contracts to use from the provided chain
     (NO invented strikes or expiries).
   - Direction (bullish / bearish / neutral / IV play).
   - Position sizing respecting starting_capital (avoid absurd leverage), and
     adjusted by real_time_risk.
3. For each strategy in your natural-language plan, describe:
   - The legs (e.g., "Buy 1x 12/12/2025 450C at mid $20")
   - Rationale relative to catalysts, macro context, and forecasts.
   - Rough payoff profile (what happens in bear/base/bull).
   - Rough Greek profile behaviour around key catalysts (delta/gamma/theta/vega themes).
   - Explicit notes where risk signals (gamma_squeeze_risk, iv_crush_risk, liquidity_shock_risk)
     influence structure or contract count.

In this step, you do NOT output JSON.
You output markdown-style bullet points and sections that will LATER be
converted to strict JSON by another agent.
"""


AGENT_3B_SYSTEM = """
You are a strict JSON compiler for TSLA options strategies.

You are given:
- plan_text: a natural-language strategy plan from another agent
- backend_chain: the REAL options chain slice used (calls/puts for a single expiry)
- forecast_map: TSLA price scenarios at expiry (bear/base/bull)
- catalysts: structured catalysts (already derived from RAG and Grok)
- spot_price
- starting_capital
- raw_rag_context: the underlying TSLA macro + micro RAG snippets used
  to justify the strategies (for your reference only)
- chain_intel: options-chain intelligence object (IV buckets, OI/volume clusters, etc.)
- real_time_risk: a dict like:
  {
    "gamma_squeeze_risk": 0.0–1.0,
    "iv_crush_risk": 0.0–1.0,
    "liquidity_shock_risk": 0.0–1.0,
    "macro_risk": 0.0–1.0,
    "composite_score": 0.0–1.0
  }

Your job:
1. Convert plan_text into an ARRAY of strategy objects.
2. You MUST only use real contracts present in backend_chain
   (match strike, type, expiry exactly).
3. You MUST output STRICT JSON ONLY. No trailing commas, no comments, no extra keys
   beyond those described below.

Each strategy object MUST have this shape:

[
  {
    "strategy_name": "Long Call (ATM)",
    "term": "short-term",

    "legs": [
      {
        "action": "buy",
        "type": "call",
        "strike": 450.0,
        "expiry": "2025-12-12",
        "premium": 20.0,
        "contracts": 1,
        "quantity": 1
      }
    ],

    "trade_details": {
      "max_gain": 6000.0,
      "max_loss": 2000.0,
      "total_cost": 2000.0
    },

    "forecasted_prices": {
      "bear": 360.0,
      "base": 430.0,
      "bull": 540.0
    },

    "performance_summary": {
      "expected_returns_pct": {
        "bear": -100.0,
        "base": 0.0,
        "bull": 200.0
      },
      "expected_return_dollars": {
        "bear": -2000.0,
        "base": 0.0,
        "bull": 6000.0
      },
      "greeks": {
        "delta_comment": "High positive delta; behaves like leveraged long TSLA.",
        "gamma_comment": "Positive gamma; convex payoff around catalysts.",
        "theta_comment": "Negative theta; time decay accelerates near expiry.",
        "vega_comment": "Positive vega; benefits from IV expansion into events."
      }
    },

    "greeks_time_series": [
      {
        "label": "now",
        "delta": 0.45,
        "gamma": 0.12,
        "theta": -0.03,
        "vega": 0.25
      },
      {
        "label": "pre-event",
        "delta": 0.52,
        "gamma": 0.14,
        "theta": -0.05,
        "vega": 0.35
      },
      {
        "label": "post-event",
        "delta": 0.40,
        "gamma": 0.10,
        "theta": -0.02,
        "vega": 0.20
      }
    ],

    "payoff_curve": [
      { "underlying": 360.0, "pnl": -2000.0 },
      { "underlying": 430.0, "pnl": 0.0 },
      { "underlying": 540.0, "pnl": 6000.0 }
    ],

    "trade_plan": {
      "entry_logic": "Enter on strength holding above short-term support with healthy tape and macro.",
      "exit_logic": "Take profits into strong moves or cut if TSLA loses key support or 50–70% of premium."
    },

    "catalysts": [
      {
        "event_name": "Q4 Earnings",
        "expected_date": "2025-01-24",
        "confidence": "high",
        "recommended_expiry": "2025-02-14",
        "trigger_signals": ["volume expansion on green days", "IV spike into event"],
        "source_evidence": "Fallback from provided catalyst list."
      }
    ],

    "risk_signals": {
      "gamma_squeeze_risk": 0.8,
      "iv_crush_risk": 0.6,
      "liquidity_shock_risk": 0.5,
      "macro_risk": 0.4,
      "composite_score": 0.6
    }
  }
]

Notes:
- greeks_time_series is meant to be plotted as a line chart over discrete labels
  (often aligned to catalysts).
- payoff_curve is meant to be plotted as a PnL vs underlying price line chart.
- The numeric values do NOT need to be mathematically perfect but must be
  internally consistent: e.g. max_loss should match worst-case PnL in
  payoff_curve, and expected_return_dollars should be consistent with
  expected_returns_pct and total_cost.
- Never include NaN or Infinity; if you do not know a number, approximate
  with a finite float.
- The same real_time_risk object can be reused for all strategies, or
  you may specialize per-strategy where the plan_text clearly implies it.
- DO NOT include any commentary or text outside the JSON array.
"""

# ==========================================================
# MARKET HELPERS
# ==========================================================

def fetch_json(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print("fetch_json ERROR:", e)
    return {}


def fetch_spot_price() -> float:
    data = fetch_json(f"{MARKET_API}/price?ticker=TSLA")
    try:
        return float(
            data.get("price")
            or data.get("Price")
            or data.get("close")
            or data.get("Close")
            or 0.0
        )
    except Exception:
        return 0.0

def filter_catalysts_needle_movers(
    raw_catalysts: list[dict],
    rag_context: str,
    spot: float,
    expiry: str,
) -> list[dict]:
    """
    Second-stage filter on catalysts to keep only genuine needle-mover events.
    """
    if not raw_catalysts:
        return []

    payload = json.dumps(
        {
            "spot": spot,
            "expiry": expiry,
            "raw_catalysts": raw_catalysts,
            "rag_context": rag_context,
        },
        indent=2,
    )

    txt = call_llm(AGENT_2C_SYSTEM, payload, LLM_CATALYST_MODEL, temperature=0.15)
    try:
        out = json.loads(txt)
        if isinstance(out, list):
            return out
    except Exception as e:
        print("CATALYST FILTER JSON ERROR:", e)
    return raw_catalysts

def fetch_expiries():
    data = fetch_json(f"{MARKET_API}/options/expiries")
    if isinstance(data, dict):
        return data.get("expiries", [])
    return data if isinstance(data, list) else []


def fetch_chain_for_expiry(expiration: str):
    """
    Hit the market-data chain endpoint for a specific expiry.
    Returns (calls, puts).
    """
    url = f"{MARKET_API}/options/chain?ticker=TSLA&expiration={expiration}"
    data = fetch_json(url)
    if not isinstance(data, dict):
        return [], []
    return data.get("calls", []), data.get("puts", [])


def _parse_expiry_to_date(exp_str: str) -> datetime:
    """
    Normalize expiries that may include suffixes like ':w' or ':m'.
    """
    if not exp_str:
        raise ValueError("empty expiry string")
    base = exp_str.split(":")[0]
    return datetime.fromisoformat(base)


def choose_best_expiry(target_date_str: str | None, expiries: list[str]) -> str | None:
    """
    Choose the expiry that best aligns with the user target_date.
    If target_date_str is None, pick the nearest upcoming expiry.
    """
    if not expiries:
        return None

    parsed = []
    for e in expiries:
        try:
            parsed.append((e, _parse_expiry_to_date(e)))
        except Exception:
            continue

    if not parsed:
        return None

    today = datetime.utcnow().date()

    if target_date_str:
        try:
            target_dt = datetime.fromisoformat(target_date_str).date()
        except Exception:
            target_dt = today
    else:
        target_dt = today

    best = min(parsed, key=lambda tup: abs(tup[1].date() - target_dt))
    return best[0]


def build_chain_compact(
    calls: list[dict],
    puts: list[dict],
    expiry: str,
    max_per_side: int = 40,
) -> dict:
    """
    Build a compact chain payload for the LLM to reason over.
    """

    def clean_option(opt):
        return {
            "strike": float(opt.get("strike")),
            "last": opt.get("last"),
            "bid": opt.get("bid"),
            "ask": opt.get("ask"),
            "mid": opt.get("mid"),
            "volume": opt.get("volume"),
            "open_interest": opt.get("open_interest"),
            "iv": opt.get("iv"),
            "delta": opt.get("delta"),
            "gamma": opt.get("gamma"),
            "theta": opt.get("theta"),
            "vega": opt.get("vega"),
            "type": opt.get("type", "call"),
            "expiry": opt.get("expiry") or opt.get("expiration") or expiry,
        }

    calls_clean = [clean_option(o) for o in calls if o.get("strike") is not None]
    puts_clean = [clean_option(o) for o in puts if o.get("strike") is not None]

    calls_sorted = sorted(calls_clean, key=lambda o: o["strike"])
    puts_sorted = sorted(puts_clean, key=lambda o: o["strike"])

    calls_compact = calls_sorted[:max_per_side]
    puts_compact = puts_sorted[:max_per_side]

    return {
        "expiry": expiry,
        "calls": calls_compact,
        "puts": puts_compact,
    }


# ==========================================================
# RAG SEARCH
# ==========================================================

def search_rag(query: str, top_k: int = 8) -> list[str]:
    collection = get_latest_snapshot_collection()
    
    if not qdrant:
        return []

    try:
        vec = embed_model.get_embeddings([query])[0].values
        results = qdrant.search(
            collection_name=collection,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
        return [r.payload.get("text", "") for r in results]
    except Exception as e:
        print("RAG ERROR:", e)
        return []


# ==========================================================
# LLM PIPELINE HELPERS
# ==========================================================

def generate_catalysts(rag_context: str, spot: float, expiry: str) -> list[dict]:
    payload = json.dumps(
        {
            "spot": spot,
            "expiry": expiry,
            "rag_context": rag_context,
        },
        indent=2,
    )

    txt = call_llm(AGENT_2B_SYSTEM, payload, LLM_CATALYST_MODEL, temperature=0.2)
    try:
        return json.loads(txt)
    except Exception as e:
        print("CATALYST JSON ERROR:", e)
        return []

def grok_collate_catalysts_with_rl(
    rag_context: str,
    spot: float,
    expiry: str,
    raw_catalysts: list[dict],
    max_rounds: int = 3,
) -> list[dict]:
    """
    Use Grok as a catalyst collator + self-critic.
    - Round 0: initial structured catalyst list
    - Subsequent rounds: critique & fix hallucinations / contradictions
    """
    if not raw_catalysts:
        return []

    current = raw_catalysts
    history = []

    for round_idx in range(max_rounds):
        payload = json.dumps(
            {
                "round": round_idx + 1,
                "spot": spot,
                "expiry": expiry,
                "rag_context": rag_context,
                "current_catalysts": current,
                "history_last_rounds": history[-2:],
            },
            indent=2,
        )

        user_prompt = f"""
You are running CRITIQUE ROUND {round_idx + 1}.

Goals:
- Remove or downweight hallucinated events.
- Fix obviously wrong dates, directions, or magnitudes.
- Cross-check each event against rag_context.
- Prefer events that are clearly supported by multiple sources.

Output:
- A CLEANED JSON list of catalysts, following the schema in the system prompt.
- Do NOT wrap in any surrounding text.
"""

        txt = call_grok_chat(
            system_prompt=GROK_CATALYST_SYSTEM,
            user_payload=user_prompt + "\n\n" + payload,
            model="grok-2-latest",
            temperature=0.1,
            max_tokens=2048,
        )

        try:
            revised = json.loads(txt)
            if not isinstance(revised, list):
                break

            changed = (revised != current)
            history.append({"round": round_idx + 1, "changed": changed})

            current = revised
            if not changed:
                break  # converged
        except Exception as e:
            print("GROK RL JSON ERROR:", e)
            break

    return current


def generate_forecast(expiry: str, spot: float, catalysts: list[dict], rag_context: str) -> dict:
    payload = json.dumps(
        {
            "spot": spot,
            "expiry": expiry,
            "catalysts": catalysts,
            "rag_context": rag_context,
        },
        indent=2,
    )
    txt = call_llm(AGENT_2A_SYSTEM, payload, LLM_AGENT_MODEL, temperature=0.2)
    try:
        obj = json.loads(txt)
        bear_val = float(obj.get("bear", spot * 0.85))
        base_val = float(obj.get("base", spot))
        bull_val = float(obj.get("bull", spot * 1.25))

        # Enforce ordering bear < base < bull just in case
        ordered = sorted([bear_val, base_val, bull_val])
        return {
            "bear": ordered[0],
            "base": ordered[1],
            "bull": ordered[2],
        }
    except Exception as e:
        print("FORECAST JSON ERROR:", e)
        return {
            "bear": spot * 0.85,
            "base": spot,
            "bull": spot * 1.25,
        }


def plan_strategies_agent(
    user_query: str,
    starting_capital: float,
    wealth_goal,
    target_date: str | None,
    risk_level: str,
    spot: float,
    expiry: str,
    chain_compact: dict,
    catalysts: list[dict],
    forecasts: dict,
    rag_context: str,
    chain_intel: dict,
    real_time_risk: dict,   # 👈 NEW
) -> str:


    """
    Call AGENT_3A to produce a natural-language planning document.
    """
    payload = json.dumps(
        {
            "user_query": user_query,
            "starting_capital": starting_capital,
            "wealth_goal": wealth_goal,
            "risk_level": risk_level,
            "target_date": target_date,
            "spot_price": spot,
            "chosen_expiry": expiry,
            "option_chain": chain_compact,
            "real_time_risk": real_time_risk,
            "chain_intel": chain_intel, 
            "catalysts": catalysts,
            "forecast_map": forecasts,
            "rag_context_snippets": rag_context,
        },
        indent=2,
    )
    txt = call_llm(
        AGENT_3A_SYSTEM,
        payload,
        LLM_AGENT_MODEL,
        temperature=0.4,
        max_tokens=2048,
    )
    print("PLAN_TEXT RAW:", (txt or "")[:2000])
    return txt or ""


def _safe_load_json_array(txt: str):
    """
    Try to robustly extract a JSON array from LLM output.
    """
    txt = (txt or "").strip()
    if not txt:
        return None

    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("strategies"), list):
            return obj["strategies"]
    except Exception:
        pass

    try:
        start = txt.find("[")
        end = txt.rfind("]")
        if start != -1 and end != -1 and end > start:
            sub = txt[start : end + 1]
            obj = json.loads(sub)
            if isinstance(obj, list):
                return obj
    except Exception:
        pass

    return None


def finalize_strategy_json(
    plan_text: str,
    chain_compact: dict,
    forecasts: dict,
    catalysts: list[dict],
    spot: float,
    starting_capital: float,
    rag_context: str,
    real_time_risk: dict,
    chain_intel: dict, 
    attempts: int = 3,
) -> list[dict]:
    """
    Call AGENT_3B to convert plan_text into strict JSON strategies.
    RAG context is included so the finalizer can stay consistent
    with macro/micro backdrop.
    """
    backend_context = {
        "plan_text": plan_text,
        "backend_chain": chain_compact,
        "forecast_map": forecasts,
        "catalysts": catalysts,
        "spot_price": spot,
        "starting_capital": starting_capital,
        "raw_rag_context": rag_context,
        "chain_intel": chain_intel,
        "real_time_risk": real_time_risk   
    }


    for attempt in range(1, attempts + 1):
        payload = json.dumps(backend_context, indent=2)
        txt = call_llm(
            AGENT_3B_SYSTEM,
            payload,
            LLM_AGENT_MODEL,
            temperature=0.1,
            max_tokens=3072,
        )
        print(f"FINALIZER RAW (attempt {attempt}):", (txt or "")[:2000])

        try:
            strategies = _safe_load_json_array(txt)
            if strategies is not None:
                return strategies
        except Exception as e:
            print(f"FINALIZER JSON ERROR (attempt {attempt}):", e)

    print("⚠️ FINALIZER FAILED AFTER ATTEMPTS — returning empty list.")
    return []


# ==========================================================
# FALLBACK STRATEGY (CHAIN-DRIVEN)
# ==========================================================

def build_fallback_strategy(
    spot: float,
    expiry: str,
    chain_compact: dict,
    forecasts_primary: dict,
    starting_capital: float,
    catalysts: list[dict],
) -> list[dict]:
    """
    Simple ATM long call using the real chain as a hard fallback.
    """

    calls = chain_compact.get("calls", [])
    if not calls:
        return []

    calls_sorted = sorted(calls, key=lambda c: abs(c["strike"] - spot))
    atm = calls_sorted[0]
    premium = atm.get("mid") or atm.get("last") or atm.get("bid") or atm.get("ask") or 0.0
    premium = float(premium)

    if premium <= 0:
        return []

    max_contracts = int(starting_capital // (premium * CONTRACT_MULTIPLIER))
    contracts = max(1, max_contracts) if starting_capital > 0 else 1
    total_cost = contracts * premium * CONTRACT_MULTIPLIER

    bear = forecasts_primary.get("bear", spot * 0.85)
    base = forecasts_primary.get("base", spot)
    bull = forecasts_primary.get("bull", spot * 1.25)

    def call_payoff(underlying):
        return max(underlying - atm["strike"], 0) * CONTRACT_MULTIPLIER * contracts - total_cost

    payoff_curve = [
        {"underlying": float(bear), "pnl": float(call_payoff(bear))},
        {"underlying": float(base), "pnl": float(call_payoff(base))},
        {"underlying": float(bull), "pnl": float(call_payoff(bull))},
    ]

    max_loss = total_cost
    max_gain = max(p["pnl"] for p in payoff_curve)

    expected_dollars = {
        "bear": payoff_curve[0]["pnl"],
        "base": payoff_curve[1]["pnl"],
        "bull": payoff_curve[2]["pnl"],
    }
    expected_pct = {
        "bear": (expected_dollars["bear"] / total_cost) * 100.0 if total_cost else 0.0,
        "base": (expected_dollars["base"] / total_cost) * 100.0 if total_cost else 0.0,
        "bull": (expected_dollars["bull"] / total_cost) * 100.0 if total_cost else 0.0,
    }

    greeks_time_series = [
        {"label": "now", "delta": 0.45, "gamma": 0.12, "theta": -0.03, "vega": 0.25},
        {"label": "pre-event", "delta": 0.52, "gamma": 0.14, "theta": -0.05, "vega": 0.35},
        {"label": "post-event", "delta": 0.40, "gamma": 0.10, "theta": -0.02, "vega": 0.20},
    ]

    return [
        {
            "strategy_name": "Fallback Long Call",
            "term": "short-term",
            "legs": [
                {
                    "action": "buy",
                    "type": "call",
                    "strike": float(atm["strike"]),
                    "expiry": atm.get("expiry") or expiry,
                    "premium": premium,
                    "contracts": int(contracts),
                    "quantity": int(contracts),
                }
            ],
            "trade_details": {
                "max_gain": float(max_gain),
                "max_loss": float(max_loss),
                "total_cost": float(total_cost),
            },
            "forecasted_prices": {
                "bear": float(bear),
                "base": float(base),
                "bull": float(bull),
            },
            "performance_summary": {
                "expected_returns_pct": {
                    "bear": float(expected_pct["bear"]),
                    "base": float(expected_pct["base"]),
                    "bull": float(expected_pct["bull"]),
                },
                "expected_return_dollars": {
                    "bear": float(expected_dollars["bear"]),
                    "base": float(expected_dollars["base"]),
                    "bull": float(expected_dollars["bull"]),
                },
                "greeks": {
                    "delta_comment": "Positive delta; behaves like leveraged long TSLA.",
                    "gamma_comment": "Positive gamma; convex payoff around catalysts.",
                    "theta_comment": "Negative theta; time decay accelerates near expiry.",
                    "vega_comment": "Positive vega; benefits from IV expansion into events.",
                },
            },
            "greeks_time_series": greeks_time_series,
            "payoff_curve": payoff_curve,
            "trade_plan": {
                "entry_logic": "Enter when TSLA holds above short-term support with constructive tape and macro.",
                "exit_logic": "Take profits into strong moves or cut if TSLA loses key support or 50–70% of premium.",
            },
            "catalysts": catalysts[:3],
        }
    ]


# ==========================================================
# TERM BUCKET HELPER
# ==========================================================

def bucket_term(horizon_days: int) -> str:
    if horizon_days <= 30:
        return "short-term"
    if horizon_days <= 120:
        return "swing"
    if horizon_days <= 365:
        return "medium-term"
    return "long-term"


# ==========================================================
# FULL PIPELINE
# ==========================================================

def run_full_strategy_pipeline(
    user_query: str,
    starting_capital: float,
    wealth_goal,
    target_date_str: str | None,
    risk_level: str,
) -> dict:
    spot = fetch_spot_price()
    expiries = fetch_expiries()
    chosen_expiry = choose_best_expiry(target_date_str, expiries) if expiries else None

    if not chosen_expiry:
        raise RuntimeError("No option expiries available from market-data service.")

    try:
        today = datetime.utcnow().date()
        target_for_term = (
            datetime.fromisoformat(target_date_str).date()
            if target_date_str
            else _parse_expiry_to_date(chosen_expiry).date()
        )
        horizon_days = max(1, (target_for_term - today).days)
    except Exception:
        horizon_days = 90
    term_label = bucket_term(horizon_days)

    calls, puts = fetch_chain_for_expiry(chosen_expiry)
    chain_compact = build_chain_compact(calls, puts, chosen_expiry, max_per_side=40)
       
    chain_intel = compute_chain_intel(chain_compact, spot=spot, expiry=chosen_expiry)


    rag_query = f"{user_query} TSLA macro Fed CPI FSD Robotaxi earnings deliveries options vol positioning sentiment"
    rag_snippets = search_rag(rag_query, top_k=8)
    rag_context = "\n\n---\n\n".join(snippets for snippets in rag_snippets if snippets)

    raw_catalysts = generate_catalysts(rag_context, spot, chosen_expiry)

    real_time_risk = compute_real_time_risk(
        chain_intel=chain_intel,
        catalysts=catalysts,
    )

# === NEW: Grok collation + RL cleanup ===
    grok_catalysts = grok_collate_catalysts_with_rl(
        rag_context=rag_context,
        spot=spot,
        expiry=chosen_expiry,
        raw_catalysts=raw_catalysts,
    )

    catalysts = filter_catalysts_needle_movers(
        raw_catalysts=grok_catalysts or raw_catalysts,
        rag_context=rag_context,
        spot=spot,
        expiry=chosen_expiry,
    )


    forecasts_primary = generate_forecast(chosen_expiry, spot, catalysts, rag_context)

    forecasts = {
        "primary_term": forecasts_primary
    }

    plan_text = plan_strategies_agent(
        user_query=user_query,
        starting_capital=starting_capital,
        wealth_goal=wealth_goal,
        target_date=target_date_str,
        risk_level=risk_level,
        spot=spot,
        expiry=chosen_expiry,
        chain_compact=chain_compact,
        catalysts=catalysts,
        forecasts=forecasts,
        rag_context=rag_context,
        chain_intel=chain_intel,
        real_time_risk=real_time_risk  # 👈 NEW
    )

    strategies = finalize_strategy_json(
        plan_text=plan_text,
        chain_compact=chain_compact,
        forecasts=forecasts,
        catalysts=catalysts,
        spot=spot,
        starting_capital=starting_capital,
        rag_context=rag_context,
        chain_intel=chain_intel,
        real_time_risk=real_time_risk   # 👈 NEW
    )


    if not strategies:
        print("⚠️ Using FALLBACK strategy.")
        strategies = build_fallback_strategy(
            spot=spot,
            expiry=chosen_expiry,
            chain_compact=chain_compact,
            forecasts_primary=forecasts_primary,
            starting_capital=starting_capital,
            catalysts=catalysts,
        )

    for s in strategies:
        s.setdefault("term", term_label)

    return {"strategies": strategies}


# ==========================================================
# ROUTES
# ==========================================================

@app.route("/plan", methods=["POST"])
def plan():
    payload = request.get_json(force=True) or {}

    try:
        user_query = payload.get("query") or "Design TSLA option strategies aligned with my inputs."
        starting_capital = float(payload.get("starting_capital") or 15000)
        wealth_goal = payload.get("wealth_goal")
        target_date_str = payload.get("target_date")
        risk_level = (payload.get("risk_level") or "high").lower()

        out = run_full_strategy_pipeline(
            user_query=user_query,
            starting_capital=starting_capital,
            wealth_goal=wealth_goal,
            target_date_str=target_date_str,
            risk_level=risk_level,
        )
        return jsonify(out), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def recreate_qdrant_collection(collection_name: str, vector_dim: int = 768):
    try:
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "size": vector_dim,
                "distance": "Cosine"
            }
        )
        print(f"✔ Recreated Qdrant collection {collection_name}")
    except Exception as e:
        print("Qdrant recreate error:", e)


# ==========================================================
# OPTION CHAIN INTELLIGENCE
# ==========================================================
def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _safe_ratio(num: float, den: float) -> float | None:
    try:
        if den and den != 0:
            return num / den
    except Exception:
        pass
    return None


def _compute_macro_risk_proxy(catalysts: list[dict]) -> float:
    """
    Very simple proxy using filtered catalysts (after 2C).
    Bearish / mixed events push risk up, bullish events reduce it.
    """
    if not catalysts:
        return 0.5

    score = 0.5
    for c in catalysts:
        direction = (c.get("impact_direction") or "").lower()
        mag = (c.get("impact_magnitude") or "").lower()
        w = {"small": 0.05, "medium": 0.1, "large": 0.2}.get(mag, 0.05)

        if direction == "bearish":
            score += w
        elif direction == "bullish":
            score -= w
        elif direction == "mixed":
            score += w * 0.3

    return _clamp01(score)


def _compute_gamma_squeeze_risk(chain_intel: dict) -> float:
    summary = chain_intel.get("summary", {})
    calls = chain_intel.get("calls", {})
    spot = float(chain_intel.get("spot") or 0.0)
    dte = int(chain_intel.get("days_to_expiry") or 0)

    total_call_oi = float(summary.get("total_call_open_interest") or 0.0)
    total_put_oi = float(summary.get("total_put_open_interest") or 0.0)
    put_call_oi_ratio = summary.get("put_call_oi_ratio")

    # Call/put dominance: high call-tilt => squeeze risk
    if put_call_oi_ratio not in (None, 0):
        call_put_ratio = 1.0 / put_call_oi_ratio
    else:
        call_put_ratio = _safe_ratio(total_call_oi, total_put_oi or 1.0) or 1.0

    # Normalize call_put_ratio above 1.0 into 0–1 band
    call_tilt = _clamp01((call_put_ratio - 1.0) / 2.0)  # ratio 3 → tilt=1

    # OI wall near spot
    top_oi = calls.get("top_open_interest_strikes") or []
    if not top_oi or spot <= 0:
        proximity_score = 0.0
    else:
        dists = []
        for s in top_oi:
            k = float(s.get("strike") or 0.0)
            if k > 0:
                d_rel = abs(k - spot) / spot
                dists.append(d_rel)
        min_dist = min(dists) if dists else 1.0
        # within 5% of spot → strong
        proximity_score = _clamp01((0.05 - min_dist) / 0.05) if min_dist < 0.05 else 0.0

    # Short DTE → more sensitive to gamma squeezes
    if dte <= 7:
        dte_factor = 1.0
    elif dte <= 21:
        dte_factor = 0.7
    elif dte <= 45:
        dte_factor = 0.4
    else:
        dte_factor = 0.2

    raw = 0.4 * call_tilt + 0.4 * proximity_score + 0.2 * dte_factor
    return _clamp01(raw)


def _compute_iv_crush_risk(chain_intel: dict, catalysts: list[dict]) -> float:
    summary = chain_intel.get("summary", {})
    calls_bucket = summary.get("calls_iv_bucket") or "unknown"
    puts_bucket = summary.get("puts_iv_bucket") or "unknown"
    dte = int(chain_intel.get("days_to_expiry") or 0)

    # Base on IV bucket
    bucket_score = {
        "unknown": 0.2,
        "low": 0.2,
        "normal": 0.4,
        "high": 0.7,
        "extreme": 0.9,
    }.get(str(calls_bucket).lower(), 0.4)

    # Event density near expiry (earnings, CPI, NFP, etc.)
    event_density = 0.0
    if catalysts and dte > 0:
        soon_events = 0
        for c in catalysts:
            date_str = c.get("expected_date") or c.get("date")
            if not date_str or date_str in ("TBD", "tbd"):
                continue
            try:
                event_dt = datetime.fromisoformat(date_str).date()
                expiry_dt = _parse_expiry_to_date(chain_intel.get("expiry")).date()
                diff = abs((event_dt - expiry_dt).days)
                if diff <= 3:
                    soon_events += 1
            except Exception:
                continue

        if soon_events > 0:
            event_density = _clamp01(0.3 + 0.15 * soon_events)

    # Shorter-dated options more likely to see IV crush post-event
    if dte <= 7:
        dte_factor = 1.0
    elif dte <= 21:
        dte_factor = 0.7
    else:
        dte_factor = 0.4

    raw = 0.5 * bucket_score + 0.3 * event_density + 0.2 * dte_factor
    return _clamp01(raw)


def _compute_liquidity_shock_risk(chain_intel: dict) -> float:
    summary = chain_intel.get("summary", {})
    calls = chain_intel.get("calls", {})
    puts = chain_intel.get("puts", {})

    total_call_vol = float(summary.get("total_call_volume") or 0.0)
    total_put_vol = float(summary.get("total_put_volume") or 0.0)
    total_vol = total_call_vol + total_put_vol

    total_call_oi = float(summary.get("total_call_open_interest") or 0.0)
    total_put_oi = float(summary.get("total_put_open_interest") or 0.0)
    total_oi = total_call_oi + total_put_oi

    # OI concentration: if top strikes have most OI, shocks can be violent
    def _concentration(side_metrics: dict, total_side_oi: float) -> float:
        top = side_metrics.get("top_open_interest_strikes") or []
        if not top or not total_side_oi:
            return 0.0
        top_oi_sum = sum(float(x.get("open_interest") or 0.0) for x in top)
        return _clamp01(top_oi_sum / total_side_oi)

    calls_conc = _concentration(calls, total_call_oi)
    puts_conc = _concentration(puts, total_put_oi)
    oi_conc = max(calls_conc, puts_conc)

    # Low volume + high OI concentration → liquidity shock risk
    vol_per_oi = _safe_ratio(total_vol, total_oi) or 0.0
    if vol_per_oi < 0.05:
        vol_factor = 1.0
    elif vol_per_oi < 0.2:
        vol_factor = 0.7
    else:
        vol_factor = 0.3

    # Extreme IV buckets can also signal stressed conditions
    calls_bucket = str(summary.get("calls_iv_bucket") or "unknown").lower()
    iv_factor = {
        "unknown": 0.3,
        "low": 0.3,
        "normal": 0.4,
        "high": 0.6,
        "extreme": 0.8,
    }.get(calls_bucket, 0.4)

    raw = 0.5 * oi_conc + 0.3 * vol_factor + 0.2 * iv_factor
    return _clamp01(raw)


def compute_real_time_risk(chain_intel: dict, catalysts: list[dict]) -> dict:
    """
    Fusion engine: combine options-chain microstructure with catalyst deck
    into a real-time risk object consumed by Agent 3A/3B.
    """
    gamma_risk = _compute_gamma_squeeze_risk(chain_intel)
    iv_crush_risk = _compute_iv_crush_risk(chain_intel, catalysts)
    liquidity_risk = _compute_liquidity_shock_risk(chain_intel)
    macro_risk = _compute_macro_risk_proxy(catalysts)

    composite = _clamp01(
        0.25 * gamma_risk
        + 0.25 * iv_crush_risk
        + 0.25 * liquidity_risk
        + 0.25 * macro_risk
    )

    return {
        "gamma_squeeze_risk": gamma_risk,
        "iv_crush_risk": iv_crush_risk,
        "liquidity_shock_risk": liquidity_risk,
        "macro_risk": macro_risk,
        "composite_score": composite,
    }

def compute_chain_intel(chain_compact: dict, spot: float, expiry: str, contract_multiplier: float = CONTRACT_MULTIPLIER) -> dict:
    """
    Compute useful summary metrics from the compact chain that the LLM can reason over.
    This does NOT need to be perfectly 'quant correct' – it just needs to be
    internally consistent and interpretable.
    """

    calls = chain_compact.get("calls", []) or []
    puts = chain_compact.get("puts", []) or []

    def _side_metrics(options: list[dict], side: str):
        total_oi = 0
        total_volume = 0
        ivs = []
        strikes = []
        moneyness = []  # (strike / spot - 1)
        enriched = []

        for opt in options:
            oi = float(opt.get("open_interest") or 0.0)
            vol = float(opt.get("volume") or 0.0)
            iv = float(opt.get("iv") or 0.0)
            k = float(opt.get("strike") or 0.0)

            total_oi += oi
            total_volume += vol

            if iv > 0:
                ivs.append(iv)
            if k > 0 and spot > 0:
                strikes.append(k)
                moneyness.append((k / spot) - 1.0)

            enriched.append({
                "strike": k,
                "open_interest": oi,
                "volume": vol,
                "iv": iv,
            })

        # top clusters by OI / volume
        top_oi = sorted(enriched, key=lambda x: x["open_interest"], reverse=True)[:5]
        top_vol = sorted(enriched, key=lambda x: x["volume"], reverse=True)[:5]

        iv_avg = float(np.mean(ivs)) if ivs else 0.0
        iv_min = float(np.min(ivs)) if ivs else 0.0
        iv_max = float(np.max(ivs)) if ivs else 0.0

        # crude IV bucket
        if iv_avg == 0:
            iv_bucket = "unknown"
        elif iv_avg < 0.4:
            iv_bucket = "low"
        elif iv_avg < 0.7:
            iv_bucket = "normal"
        elif iv_avg < 1.0:
            iv_bucket = "high"
        else:
            iv_bucket = "extreme"

        return {
            "side": side,
            "total_open_interest": total_oi,
            "total_volume": total_volume,
            "iv_avg": iv_avg,
            "iv_min": iv_min,
            "iv_max": iv_max,
            "iv_bucket": iv_bucket,
            "top_open_interest_strikes": top_oi,
            "top_volume_strikes": top_vol,
        }

    calls_metrics = _side_metrics(calls, "call")
    puts_metrics = _side_metrics(puts, "put")
    expected_move = float(abs(np.mean([
    (opt["iv"] or 0) * spot * np.sqrt(max(((_parse_expiry_to_date(expiry).date() - date.today()).days) / 365, 1e-6))
    for opt in chain_compact.get("calls", [])[:5]
    ])))

  
    dte = ( _parse_expiry_to_date(expiry).date() - date.today() ).days
    total_call_oi = calls_metrics["total_open_interest"]
    total_put_oi = puts_metrics["total_open_interest"]
    total_call_vol = calls_metrics["total_volume"]
    total_put_vol = puts_metrics["total_volume"]

    put_call_oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else None
    put_call_vol_ratio = (total_put_vol / total_call_vol) if total_call_vol > 0 else None
    

    return {
        "spot": float(spot),
        "expiry": expiry,
        "summary": {
            "total_call_open_interest": total_call_oi,
            "total_put_open_interest": total_put_oi,
            "put_call_oi_ratio": put_call_oi_ratio,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "put_call_volume_ratio": put_call_vol_ratio,
            "calls_iv_bucket": calls_metrics["iv_bucket"],
            "puts_iv_bucket": puts_metrics["iv_bucket"],
        },
        "calls": calls_metrics,
        "puts": puts_metrics,
        "expected_move_estimate": expected_move,
        "days_to_expiry": dte
    }


from qdrant_client.http.models import PointStruct

def load_table_rows(table_name: str):
    query = f"SELECT * FROM `{PROJECT_ID}.tsla_data.{table_name}`"
    try:
        df = bigquery_client.query(query).result().to_dataframe()
        return df
    except Exception as e:
        print(f"BigQuery ERROR loading {table_name}:", e)
        return pd.DataFrame()

def embed_text_batch(text_list: list[str]):
    if not text_list:
        return []

    embeddings = embed_model.get_embeddings(text_list)
    return [e.values for e in embeddings]   # 768-d floats

def upsert_to_qdrant(collection: str, df: pd.DataFrame, text_col: str = "text"):
    points = []
    for idx, row in df.iterrows():
        text = str(row[text_col]) if text_col in row else str(row)
        vector = embed_text_batch([text])[0]
        points.append(
            PointStruct(
                id=f"{collection}-{idx}",
                vector=vector,
                payload=row.to_dict()
            )
        )
    try:
        qdrant.upsert(collection_name=collection, points=points)
        print(f"✔ Upserted {len(points)} points into {collection}")
    except Exception as e:
        print("Qdrant upsert error:", e)

def get_latest_snapshot_collection():
    try:
        cols = qdrant.get_collections().collections
        names = [c.name for c in cols if c.name.startswith("tsla_rag_snapshot_")]

        if not names:
            return QDRANT_COLLECTION   # fallback
        return sorted(names)[-1]       # newest YYYYMMDD
    except:
        return QDRANT_COLLECTION

def list_all_tsla_tables():
    """Return all table names inside dataset tsla_data."""
    try:
        dataset_ref = bigquery.DatasetReference(PROJECT_ID, "tsla_data")
        tables = list(bigquery_client.list_tables(dataset_ref))
        return [t.table_id for t in tables]
    except Exception as e:
        print("BQ LIST TABLES ERROR:", e)
        return []

def serialize_row_to_text(row):
    """
    Converts a BigQuery row of ANY schema into a clean text string.
    """
    pieces = []
    for col, val in row.items():
        if pd.isna(val):
            continue

        # Lists/objects → flatten safely
        if isinstance(val, (list, dict)):
            val = json.dumps(val)

        pieces.append(f"{col}: {val}")

    return " | ".join(pieces)

@app.route("/embed-snapshot", methods=["POST"])
def embed_snapshot():
    """
    Rebuild the ENTIRE RAG snapshot from all tables in BigQuery dataset tsla_data.
    ANY table will be embedded, regardless of schema.
    If 'text' column does not exist, we automatically construct one.
    Collection name = tsla_rag_snapshot_<YYYYMMDD>
    """
    try:
        today = datetime.utcnow().strftime("%Y%m%d")
        collection_name = f"tsla_rag_snapshot_{today}"

        recreate_qdrant_collection(collection_name)

        tables = list_all_tsla_tables()
        print("📄 Tables found in tsla_data:", tables)

        for table in tables:
            df = load_table_rows(table)
            if df.empty:
                print(f"⚠ No rows in {table}")
                continue

            # ============================================================
            # 🔥 AUTO-CREATE A TEXT COLUMN IF MISSING
            # ============================================================
            if "text" not in df.columns:
                print(f"ℹ️ Table {table} missing 'text' column — generating automatically.")
                df["text"] = df.apply(lambda row: serialize_row_to_text(row), axis=1)

            # ============================================================
            # 🔥 Embed rows (now guaranteed to have `text`)
            # ============================================================
            upsert_to_qdrant(collection_name, df, text_col="text")

        return jsonify({
            "status": "ok",
            "collection": collection_name
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/gcs-csv-event", methods=["POST"])
def gcs_csv_event():
    """
    Triggered by EventArc when a CSV file is finalized in GCS.

    Supports ALL possible structures:
    - CloudEvent wrapper
    - Pub/Sub wrapped event
    - Direct GCS event
    """

    try:
        # ==========================================================================
        # 1) Extract raw envelope
        # ==========================================================================
        envelope = request.get_json(force=True)

        if envelope is None:
            return "Bad Request: No JSON received", 400

        # There are MANY formats EventArc may use depending on transport.
        # We normalize ALL of them here safely.

        event = envelope

        # Pub/Sub wrapped events put the data in base64-encoded field.
        if "message" in envelope:
            msg = envelope["message"]

            # If data is base64 → decode it
            if "data" in msg:
                import base64
                try:
                    decoded = base64.b64decode(msg["data"]).decode("utf-8")
                    event = json.loads(decoded)
                except Exception:
                    event = msg  # fallback

        # ==========================================================================
        # 2) Extract bucket & object name (works for all formats)
        # ==========================================================================
        bucket = (
            event.get("bucket")
            or event.get("bucketName")
            or event.get("source", {}).get("bucket")
        )

        name = (
            event.get("object")
            or event.get("name")
            or event.get("objectId")
            or event.get("protoPayload", {}).get("resourceName", "").split("/")[-1]
        )

        if not bucket or not name:
            print("EVENT DEBUG:", envelope)
            return jsonify({"error": "Could not extract bucket or object name"}), 400

        # ==========================================================================
        # 3) Only process .csv files
        # ==========================================================================
        if not name.lower().endswith(".csv"):
            return jsonify({
                "status": "ignored",
                "file": name,
                "reason": "Not a CSV"
            }), 200

        # ==========================================================================
        # 4) Convert file name → BigQuery table name
        # ==========================================================================
        table_name = (
            name.replace("/", "_")
            .replace(".csv", "")
            .replace("-", "_")
            .replace(" ", "_")
            .lower()
        )

        dataset_id = "tsla_data"
        table_id = f"{PROJECT_ID}.{dataset_id}.{table_name}"
        uri = f"gs://{bucket}/{name}"

        print(f"🚀 Loading file into BigQuery: {uri} → {table_id}")

        # ==========================================================================
        # 5) Load CSV → BigQuery
        # ==========================================================================
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=0,
            field_delimiter=",",
            write_disposition="WRITE_TRUNCATE",
            allow_jagged_rows=True,
            allow_quoted_newlines=True,
            quote_character='"',
        )

        load_job = bigquery_client.load_table_from_uri(uri, table_id, job_config=job_config)
        load_job.result()  # Wait until finished

        table_obj = bigquery_client.get_table(table_id)

        return jsonify({
            "status": "success",
            "bucket": bucket,
            "file": name,
            "table_created": table_id,
            "rows_loaded": table_obj.num_rows
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
