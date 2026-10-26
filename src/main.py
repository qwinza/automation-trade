"""Hermes Multi-Agent Trading Orchestrator - Main Application Entrypoint.

Handles webhook ingestion from TradingView, orchestrates SMC multi-agent AI analysis
(The Bull, The Bear, The Judge), and stores trading decisions into MySQL via Prisma.
"""

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from prisma import Json
from prisma.models import TradeDecision
from pydantic import BaseModel, Field

from src.agents.orchestrator import run_multi_agent_analysis_async
from src.db.prisma import db
from src.db.redis import get_signal_lock_key, redis_client

# Constants
LOCK_EXPIRATION_SECONDS = 300  # 5 minutes lock cooldown for duplicate signals

# Configure logging
logger = logging.getLogger("hermes.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Initialize FastAPI Application
app = FastAPI(
    title="Hermes Multi-Agent Trading Orchestrator",
    description="Automated directional trading orchestrator using TradingView webhooks, SMC Multi-Agent AI, and Prisma ORM.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Lifecycle Events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    """Connect to MySQL via Prisma client on application startup."""
    logger.info("Connecting to MySQL via Prisma client...")
    if not db.is_connected():
        await db.connect()
    logger.info("Prisma client successfully connected to MySQL.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Disconnect Prisma client and safely close Redis connection on application shutdown."""
    logger.info("Disconnecting Prisma client...")
    if db.is_connected():
        await db.disconnect()
    logger.info("Prisma client disconnected successfully.")

    logger.info("Closing Redis connection...")
    if redis_client is not None:
        try:
            await redis_client.close()
            logger.info("Redis connection closed successfully.")
        except Exception as exc:
            logger.warning("Error while closing Redis connection: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
from src.models.payload import TradingViewPayload


def format_market_data(payload: TradingViewPayload) -> str:
    """Format TradingView payload into a clean, structured string for agent consumption."""
    text = (
        f"Asset: {payload.asset}\n"
        f"Timeframe: {payload.timeframe}\n"
        f"Current Price: {payload.price}\n"
        f"Market Structure: {payload.structure}\n"
        f"Liquidity: {payload.liquidity}\n"
        f"Order Block: {payload.order_block}\n"
        f"Fair Value Gap (FVG): {payload.fvg}"
    )
    if payload.news_context and payload.news_context != "None":
        text += f"\nNews Context: {payload.news_context}"
    return text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> Dict[str, Any]:
    """Health check endpoint returning service status and database connection status."""
    is_connected = db.is_connected()
    return {
        "status": "online",
        "service": "Hermes Multi-Agent Trading Orchestrator",
        "database": "connected" if is_connected else "disconnected",
        "database_connected": is_connected,
    }


@app.post("/tv-webhook", status_code=status.HTTP_200_OK)
async def tv_webhook(payload: TradingViewPayload) -> Dict[str, Any]:
    """Handle incoming TradingView webhook, execute SMC analysis, and log decision to DB."""
    logger.info(
        "Received TradingView webhook for %s on %s (Price: %s)",
        payload.asset,
        payload.timeframe,
        payload.price,
    )

    # 1. Concurrency & Cooldown Lock via Redis
    lock_key = get_signal_lock_key(payload.asset, payload.timeframe)
    try:
        is_locked = await redis_client.get(lock_key)
        if is_locked:
            logger.info(
                "Cooldown active for %s (%s). Webhook ignored.",
                lock_key,
                payload.asset,
            )
            return {
                "status": "ignored",
                "message": "Cooldown active for this asset and timeframe.",
            }

        # Set key in Redis with 300s TTL (5 minutes cooldown)
        await redis_client.set(lock_key, "locked", ex=LOCK_EXPIRATION_SECONDS)
        logger.info(
            "Acquired Redis lock %s for %d seconds.",
            lock_key,
            LOCK_EXPIRATION_SECONDS,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Redis lock operation failed for %s: %s", lock_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Redis lock operation failed: {exc}",
        ) from exc

    # 2. Format payload into clean market data string
    market_data = format_market_data(payload)

    # 3. Run multi-agent AI analysis (The Bull, The Bear, The Judge, and The Economist)
    try:
        analysis_result = await run_multi_agent_analysis_async(
            asset=payload.asset,
            market_data=market_data,
            payload=payload,
        )
    except Exception as exc:
        logger.error("Error executing multi-agent analysis: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-agent analysis failed: {exc}",
        ) from exc

    judge_decision = analysis_result.get("judge_decision") or {}
    action = str(judge_decision.get("action", "HOLD")).upper()
    confidence = int(judge_decision.get("confidence", 0))
    reasoning = str(judge_decision.get("reasoning", ""))

    bull_thinking = analysis_result.get("bull", "") or analysis_result.get("bull_analysis", "")
    bear_thinking = analysis_result.get("bear", "") or analysis_result.get("bear_analysis", "")
    fundamental_thinking = analysis_result.get("fundamental", "") or analysis_result.get("fundamental_analysis", "")

    # 3. Insert decision into MySQL using Prisma
    try:
        market_context_data = {
            "bull": bull_thinking,
            "bear": bear_thinking,
            "fundamental": fundamental_thinking,
            "bull_analysis": bull_thinking,
            "bear_analysis": bear_thinking,
            "fundamental_analysis": fundamental_thinking,
            "judge_reasoning": reasoning,
            "market_data": market_data,
            "news_context": payload.news_context,
            "price": payload.price,
            "structure": payload.structure,
            "liquidity": payload.liquidity,
            "order_block": payload.order_block,
            "fvg": payload.fvg,
        }

        created_record = await TradeDecision.prisma().create(
            data={
                "asset": payload.asset,
                "timeframe": payload.timeframe,
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "market_context": Json(market_context_data),
            }
        )
        logger.info(
            "Trade decision saved to MySQL with ID: %s (Action: %s, Confidence: %d)",
            created_record.id,
            action,
            confidence,
        )
    except Exception as exc:
        logger.error("Failed to persist trade decision to database: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database insertion failed: {exc}",
        ) from exc

    # 4. Return response with HTTP 200
    return {
        "status": "success",
        "decision": judge_decision,
    }
