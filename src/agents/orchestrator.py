"""Hermes Multi-Agent Orchestrator.

Orchestrates sequential reasoning between:
1. The Bull (SMC Long specialist)
2. The Bear (SMC Short specialist)
3. The Judge (Chief Risk Manager enforcing RR >= 1:2)
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
import google.generativeai as genai

from src.agents.prompts import (
    BEAR_AGENT_PROMPT,
    BULL_AGENT_PROMPT,
    FUNDAMENTAL_AGENT_PROMPT,
    JUDGE_AGENT_PROMPT,
)

# Load environment variables (.env has priority)
load_dotenv(override=True)

# Initialize Gemini API globally
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logger = logging.getLogger("hermes.agents.orchestrator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _get_env_config() -> tuple[str, str, str]:
    """Retrieve Anthropic client configuration from environment.

    Returns:
        tuple[api_key, base_url, model_name]
    """
    load_dotenv(override=True)
    raw_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or ""
    # Strip URL or whitespace if accidentally included in api_key (e.g. https://.../sk-...)
    api_key = raw_api_key.strip()
    if "sk-" in api_key:
        api_key = api_key[api_key.find("sk-"):]

    raw_base_url = os.getenv("ANTHROPIC_BASE_URL") or "https://api.mwapi.dev"
    base_url = raw_base_url.strip().rstrip("/")
    # Anthropic SDK automatically appends /v1/messages.
    # Strip trailing /v1 so we don't end up with /v1/v1/messages.
    base_url = re.sub(r"/v1/?$", "", base_url)

    model_name = os.getenv("MODEL_NAME", "claude-sonnet-5")

    return api_key, base_url, model_name


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


def run_multi_agent_analysis(
    asset: str,
    market_data: str,
    payload: Any = None,
) -> Dict[str, Any]:
    """Execute synchronous multi-agent SMC & Fundamental analysis for a given asset.

    Workflow:
    1. The Economist (Gemini) evaluates news event if high-impact news detected.
    2. The Bull analyzes market_data for Long setups.
    3. The Bear analyzes market_data for Short setups.
    4. The Judge arbitrates Bull, Bear, and Fundamental Context against market_data with strict RR >= 1:2.

    Args:
        asset: Symbol/instrument name (e.g., "BTCUSDT", "EURUSD").
        market_data: Comprehensive market data string (price, structure, liquidity, OB, FVG).
        payload: Optional TradingViewPayload object or dict containing news_context.

    Returns:
        dict: Contains Bull analysis, Bear analysis, Fundamental analysis, and parsed Judge JSON decision.
    """
    api_key, base_url, model_name = _get_env_config()

    # Fundamental Agent evaluation
    fundamental_analysis = "CLEAR: No high-impact red news. Rely entirely on technicals."
    news_context, has_high_impact = _extract_news_context(market_data, payload)

    if has_high_impact:
        logger.info("High-impact red news detected. Invoking Fundamental Agent (Gemini)...")
        context_to_send = news_context if news_context and news_context != "None" else market_data
        fundamental_prompt = f"{FUNDAMENTAL_AGENT_PROMPT}\n\nNews Event / Context:\n{context_to_send}"
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            gemini_resp = model.generate_content(fundamental_prompt)
            if gemini_resp and hasattr(gemini_resp, "text") and gemini_resp.text:
                fundamental_analysis = gemini_resp.text.strip()
            logger.info("Fundamental analysis result: %s", fundamental_analysis)
        except (NotFound, Exception) as exc:
            is_not_found = isinstance(exc, NotFound) or "not found" in str(exc).lower() or "404" in str(exc)
            if is_not_found:
                logger.warning("gemini-1.5-flash-latest returned NotFound: %s. Attempting fallback models...", exc)
                fallback_models = ["gemini-pro", "gemini-1.5-pro-latest", "gemini-flash-latest"]
                fallback_success = False
                for fb_name in fallback_models:
                    try:
                        fb_model = genai.GenerativeModel(fb_name)
                        fb_resp = fb_model.generate_content(fundamental_prompt)
                        if fb_resp and hasattr(fb_resp, "text") and fb_resp.text:
                            fundamental_analysis = fb_resp.text.strip()
                            logger.info("Fallback model %s succeeded: %s", fb_name, fundamental_analysis)
                            fallback_success = True
                            break
                    except Exception as fb_exc:
                        logger.warning("Fallback model %s failed: %s", fb_name, fb_exc)
                if not fallback_success:
                    logger.error("All fallback models failed after NotFound: %s", exc)
                    fundamental_analysis = f"NEUTRAL (Error: Gemini models not found - {exc})"
            else:
                logger.error("Failed to run Gemini fundamental analysis: %s", exc, exc_info=True)
                fundamental_analysis = f"NEUTRAL (Error: {exc})"

    logger.info("Initializing Anthropic client with base_url: %s, model: %s", base_url, model_name)
    client = Anthropic(api_key=api_key, base_url=base_url)

    # 1. Run The Bull
    logger.info("Running The Bull agent for %s...", asset)
    bull_user_msg = (
        f"Asset: {asset}\n\nMarket Data:\n{market_data}\n\n"
        "Evaluate this market data from an SMC Long perspective. Provide your full analysis."
    )
    bull_resp = client.messages.create(
        model=model_name,
        system=BULL_AGENT_PROMPT,
        max_tokens=1500,
        messages=[{"role": "user", "content": bull_user_msg}],
    )
    bull_analysis = bull_resp.content[0].text if bull_resp.content else ""

    # 2. Run The Bear
    logger.info("Running The Bear agent for %s...", asset)
    bear_user_msg = (
        f"Asset: {asset}\n\nMarket Data:\n{market_data}\n\n"
        "Evaluate this market data from an SMC Short perspective. Provide your full analysis."
    )
    bear_resp = client.messages.create(
        model=model_name,
        system=BEAR_AGENT_PROMPT,
        max_tokens=1500,
        messages=[{"role": "user", "content": bear_user_msg}],
    )
    bear_analysis = bear_resp.content[0].text if bear_resp.content else ""

    # 3. Run The Judge
    logger.info("Running The Judge agent for %s...", asset)
    judge_user_msg = _build_judge_prompt(
        asset=asset,
        market_data=market_data,
        bull_thesis=bull_analysis,
        bear_thesis=bear_analysis,
        fundamental_context=fundamental_analysis,
    )
    judge_resp = client.messages.create(
        model=model_name,
        system=JUDGE_AGENT_PROMPT,
        max_tokens=1000,
        messages=[{"role": "user", "content": judge_user_msg}],
    )
    judge_raw = judge_resp.content[0].text if judge_resp.content else ""

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


async def run_multi_agent_analysis_async(
    asset: str,
    market_data: str,
    payload: Any = None,
) -> Dict[str, Any]:
    """Execute asynchronous multi-agent SMC & Fundamental analysis for a given asset.

    Workflow:
    1. Fundamental Agent ("The Economist") evaluates news via Gemini API if high-impact news detected.
    2. The Bull analyzes market_data for Long setups.
    3. The Bear analyzes market_data for Short setups.
    4. The Judge arbitrates Bull, Bear, and Fundamental Context against market_data with strict RR >= 1:2.
    """
    api_key, base_url, model_name = _get_env_config()

    fundamental_analysis = "CLEAR: No high-impact red news. Rely entirely on technicals."
    news_context, has_high_impact = _extract_news_context(market_data, payload)

    if has_high_impact:
        logger.info("[Async] High-impact red news detected. Invoking Fundamental Agent (Gemini)...")
        context_to_send = news_context if news_context and news_context != "None" else market_data
        fundamental_prompt = f"{FUNDAMENTAL_AGENT_PROMPT}\n\nNews Event / Context:\n{context_to_send}"
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            gemini_resp = await model.generate_content_async(fundamental_prompt)
            if gemini_resp and hasattr(gemini_resp, "text") and gemini_resp.text:
                fundamental_analysis = gemini_resp.text.strip()
            logger.info("[Async] Fundamental analysis complete: %s", fundamental_analysis)
        except (NotFound, Exception) as exc:
            is_not_found = isinstance(exc, NotFound) or "not found" in str(exc).lower() or "404" in str(exc)
            if is_not_found:
                logger.warning("[Async] gemini-1.5-flash-latest returned NotFound: %s. Attempting fallback models...", exc)
                fallback_models = ["gemini-pro", "gemini-1.5-pro-latest", "gemini-flash-latest"]
                fallback_success = False
                for fb_name in fallback_models:
                    try:
                        fb_model = genai.GenerativeModel(fb_name)
                        fb_resp = await fb_model.generate_content_async(fundamental_prompt)
                        if fb_resp and hasattr(fb_resp, "text") and fb_resp.text:
                            fundamental_analysis = fb_resp.text.strip()
                            logger.info("[Async] Fallback model %s succeeded: %s", fb_name, fundamental_analysis)
                            fallback_success = True
                            break
                    except Exception as fb_exc:
                        logger.warning("[Async] Fallback model %s failed: %s", fb_name, fb_exc)
                if not fallback_success:
                    logger.error("[Async] All fallback models failed after NotFound: %s", exc)
                    fundamental_analysis = f"NEUTRAL (Error: Gemini models not found - {exc})"
            else:
                logger.error("[Async] Failed to run Gemini fundamental analysis: %s", exc, exc_info=True)
                fundamental_analysis = f"NEUTRAL (Error: {exc})"

    client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    # 1. Run The Bull
    logger.info("[Async] Running The Bull agent for %s...", asset)
    bull_user_msg = (
        f"Asset: {asset}\n\nMarket Data:\n{market_data}\n\n"
        "Evaluate this market data from an SMC Long perspective. Provide your full analysis."
    )
    bull_resp = await client.messages.create(
        model=model_name,
        system=BULL_AGENT_PROMPT,
        max_tokens=1500,
        messages=[{"role": "user", "content": bull_user_msg}],
    )
    bull_analysis = bull_resp.content[0].text if bull_resp.content else ""

    # 2. Run The Bear
    logger.info("[Async] Running The Bear agent for %s...", asset)
    bear_user_msg = (
        f"Asset: {asset}\n\nMarket Data:\n{market_data}\n\n"
        "Evaluate this market data from an SMC Short perspective. Provide your full analysis."
    )
    bear_resp = await client.messages.create(
        model=model_name,
        system=BEAR_AGENT_PROMPT,
        max_tokens=1500,
        messages=[{"role": "user", "content": bear_user_msg}],
    )
    bear_analysis = bear_resp.content[0].text if bear_resp.content else ""

    # 3. Run The Judge
    logger.info("[Async] Running The Judge agent for %s...", asset)
    judge_user_msg = _build_judge_prompt(
        asset=asset,
        market_data=market_data,
        bull_thesis=bull_analysis,
        bear_thesis=bear_analysis,
        fundamental_context=fundamental_analysis,
    )
    judge_resp = await client.messages.create(
        model=model_name,
        system=JUDGE_AGENT_PROMPT,
        max_tokens=1000,
        messages=[{"role": "user", "content": judge_user_msg}],
    )
    judge_raw = judge_resp.content[0].text if judge_resp.content else ""

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
