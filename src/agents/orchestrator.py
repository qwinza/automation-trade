"""Hermes Multi-Agent Orchestrator.

Orchestrates sequential reasoning between:
1. The Bull (SMC Long specialist)
2. The Bear (SMC Short specialist)
3. The Judge (Chief Risk Manager enforcing RR >= 1:2)
4. The Economist (Macro / Fundamental specialist for red news)

All LLM calls are centralized through a single local API Gateway (9Router)
using the OpenAI-compatible AsyncOpenAI client.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agents.prompts import (
    BEAR_AGENT_PROMPT,
    BULL_AGENT_PROMPT,
    FUNDAMENTAL_AGENT_PROMPT,
    JUDGE_AGENT_PROMPT,
)

# Load environment variables (.env has priority)
load_dotenv(override=True)

logger = logging.getLogger("hermes.agents.orchestrator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Global 9Router client (OpenAI-compatible)
client = AsyncOpenAI(
    api_key=os.getenv("ROUTER_API_KEY") or "dummy-key",
    base_url=os.getenv("ROUTER_BASE_URL", "http://139.190.97.252:20128/v1"),
)


def _clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """Extract and parse pure JSON from LLM output.

    Handles edge cases like markdown codeblocks (```json ... ```)
    or conversational text wrapping the JSON object.
    """
    text = raw_text.strip()

    # Attempt direct json parse first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Strip code block wrappers if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned.strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Find the first balanced JSON object {...} using regex
    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Fallback structure if parsing completely fails
    logger.error("Failed to parse Judge response into JSON. Raw: %s", raw_text)
    return {
        "action": "HOLD",
        "confidence": 0,
        "reasoning": f"Failed to parse valid JSON from Judge response. Raw text: {raw_text[:200]}...",
    }


def _build_judge_prompt(
    asset: str,
    market_data: str,
    bull_thesis: str,
    bear_thesis: str,
    fundamental_context: str = "CLEAR: No high-impact red news. Rely entirely on technicals.",
) -> str:
    """Build user message for The Judge containing asset, market data, fundamental context, and both theses."""
    return f"""Asset: {asset}

Market Data:
{market_data}

==================================================
[FUNDAMENTAL CONTEXT (MACRO / NEWS)]
==================================================
{fundamental_context}

==================================================
[THE BULL THESIS (LONG SETUP)]
==================================================
{bull_thesis}

==================================================
[THE BEAR THESIS (SHORT SETUP)]
==================================================
{bear_thesis}

==================================================
[EVALUATION INSTRUCTIONS]
==================================================
Analyze the market data and independently verify the claims of The Bull and The Bear.
Incorporate the Fundamental Context:
- If Fundamental Context is "NEUTRAL" or "CLEAR", ignore it and focus 100% on the technical Bull/Bear analysis.
- If Fundamental Context is Bullish/Dovish or Bearish/Hawkish from major red news (FOMC, NFP, CPI, PPI), use it as a Veto against counter-trend setups or as a trend amplifier.
Apply strict Risk Management:
- Ensure Risk-to-Reward (RR) is at least 1:2 (RR >= 1:2). If not, select "HOLD".
- Require verified SMC confluences (liquidity sweep, structure shift, order block / FVG alignment).
- If signals are conflicting or unclear, choose "HOLD".

Respond ONLY in raw, valid JSON with no markdown wrapping:
{{"action": "BUY" | "SELL" | "HOLD", "confidence": 0-100, "reasoning": "..."}}
"""


def _extract_news_context(market_data: str, payload: Any = None) -> tuple[str, bool]:
    """Helper to extract news_context and check for high-impact macroeconomic events."""
    news_context = "None"
    if payload is not None:
        if hasattr(payload, "news_context"):
            news_context = str(payload.news_context)
        elif isinstance(payload, dict) and "news_context" in payload:
            news_context = str(payload["news_context"])

    keywords = ["fomc", "cpi", "nfp", "ppi", "red"]
    news_target = f"{market_data} {news_context}".lower()
    has_high_impact = any(kw in news_target for kw in keywords)
    return news_context, has_high_impact


async def run_multi_agent_analysis_async(
    asset: str,
    market_data: str,
    payload: Any = None,
) -> Dict[str, Any]:
    """Execute asynchronous multi-agent SMC & Fundamental analysis for a given asset via 9Router.

    Workflow:
    1. Fundamental Agent ("The Economist") evaluates news via 9Router (gemini-1.5-flash) if high-impact news detected.
    2. The Bull analyzes market_data for Long setups via 9Router (claude-sonnet-4-6).
    3. The Bear analyzes market_data for Short setups via 9Router (claude-sonnet-4-6).
    4. The Judge arbitrates Bull, Bear, and Fundamental Context against market_data with strict RR >= 1:2.

    Args:
        asset: Symbol/instrument name (e.g., "BTCUSDT", "EURUSD").
        market_data: Comprehensive market data string (price, structure, liquidity, OB, FVG).
        payload: Optional TradingViewPayload object or dict containing news_context.

    Returns:
        dict: Contains Bull analysis, Bear analysis, Fundamental analysis, and parsed Judge JSON decision.
    """
    fundamental_analysis = "CLEAR: No high-impact red news. Rely entirely on technicals."
    news_context, has_high_impact = _extract_news_context(market_data, payload)

    # 1. Fundamental Agent (The Economist)
    if has_high_impact:
        logger.info("[Async] High-impact red news detected. Invoking Fundamental Agent via 9Router...")
        context_to_send = news_context if news_context and news_context != "None" else market_data
        try:
            econ_resp = await client.chat.completions.create(
                model="gemini-1.5-flash",
                messages=[
                    {"role": "system", "content": FUNDAMENTAL_AGENT_PROMPT},
                    {"role": "user", "content": f"News Event / Context:\n{context_to_send}"},
                ],
            )
            if econ_resp and econ_resp.choices and econ_resp.choices[0].message.content:
                fundamental_analysis = econ_resp.choices[0].message.content.strip()
            logger.info("[Async] Fundamental analysis complete: %s", fundamental_analysis)
        except Exception as exc:
            logger.error("[Async] Failed to run fundamental analysis via 9Router: %s", exc, exc_info=True)
            fundamental_analysis = "NEUTRAL: Router Error"

    # 2. Run The Bull
    logger.info("[Async] Running The Bull agent for %s...", asset)
    bull_user_msg = (
        f"Asset: {asset}\n\nMarket Data:\n{market_data}\n\n"
        "Evaluate this market data from an SMC Long perspective. Provide your full analysis."
    )
    bull_resp = await client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": BULL_AGENT_PROMPT},
            {"role": "user", "content": bull_user_msg},
        ],
    )
    bull_analysis = (bull_resp.choices[0].message.content or "").strip() if bull_resp and bull_resp.choices else ""

    # 3. Run The Bear
    logger.info("[Async] Running The Bear agent for %s...", asset)
    bear_user_msg = (
        f"Asset: {asset}\n\nMarket Data:\n{market_data}\n\n"
        "Evaluate this market data from an SMC Short perspective. Provide your full analysis."
    )
    bear_resp = await client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": BEAR_AGENT_PROMPT},
            {"role": "user", "content": bear_user_msg},
        ],
    )
    bear_analysis = (bear_resp.choices[0].message.content or "").strip() if bear_resp and bear_resp.choices else ""

    # 4. Run The Judge
    logger.info("[Async] Running The Judge agent for %s...", asset)
    judge_user_msg = _build_judge_prompt(
        asset=asset,
        market_data=market_data,
        bull_thesis=bull_analysis,
        bear_thesis=bear_analysis,
        fundamental_context=fundamental_analysis,
    )
    judge_resp = await client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": JUDGE_AGENT_PROMPT},
            {"role": "user", "content": judge_user_msg},
        ],
    )
    judge_raw = (judge_resp.choices[0].message.content or "").strip() if judge_resp and judge_resp.choices else ""
    judge_json = _clean_and_parse_json(judge_raw)

    return {
        "asset": asset,
        "fundamental": fundamental_analysis,
        "fundamental_analysis": fundamental_analysis,
        "bull": bull_analysis,
        "bear": bear_analysis,
        "judge": judge_json,
        "bull_analysis": bull_analysis,
        "bear_analysis": bear_analysis,
        "judge_decision": judge_json,
        "raw_judge_response": judge_raw,
    }


def run_multi_agent_analysis(
    asset: str,
    market_data: str,
    payload: Any = None,
) -> Dict[str, Any]:
    """Execute synchronous multi-agent SMC & Fundamental analysis for a given asset."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, run_multi_agent_analysis_async(asset, market_data, payload)).result()
    else:
        return asyncio.run(run_multi_agent_analysis_async(asset, market_data, payload))
