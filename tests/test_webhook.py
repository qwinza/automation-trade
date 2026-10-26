"""Unit and integration tests for FastAPI webhooks, Redis locking, and endpoints."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from prisma.models import TradeDecision
from src.main import TradingViewPayload, app, format_market_data


class TestHermesWebhook(unittest.TestCase):
    """Test suite for Hermes FastAPI webhook, Redis concurrency locks, and lifecycle endpoints."""

    def setUp(self) -> None:
        self.sample_payload = {
            "asset": "BTCUSDT",
            "timeframe": "15m",
            "price": 64500.5,
            "structure": "Bullish BoS",
            "liquidity": "SSL swept at 64,000",
            "order_block": "Bullish OB at 64,100",
            "fvg": "Bullish FVG at 64,200",
        }
        self.mock_analysis = {
            "asset": "BTCUSDT",
            "bull": "Bull setup verified: SSL sweep followed by strong bullish displacement.",
            "bear": "Bear setup invalidated: No overhead premium imbalance.",
            "judge": {
                "action": "BUY",
                "confidence": 85,
                "reasoning": "Clear SMC confluence with 1:3 RR targeting 65,800 BSL.",
            },
            "judge_decision": {
                "action": "BUY",
                "confidence": 85,
                "reasoning": "Clear SMC confluence with 1:3 RR targeting 65,800 BSL.",
            },
        }

        # Mock Redis client in src.main so unit tests do not depend on external Redis service
        self.redis_patcher = patch("src.main.redis_client")
        self.mock_redis = self.redis_patcher.start()
        self.mock_redis.get = AsyncMock(return_value=None)
        self.mock_redis.set = AsyncMock(return_value=True)
        self.mock_redis.close = AsyncMock(return_value=None)

    def tearDown(self) -> None:
        """Clean up patches and test records from database."""
        self.redis_patcher.stop()

        import asyncio
        from src.db.prisma import db

        async def _cleanup():
            if not db.is_connected():
                await db.connect()
            await TradeDecision.prisma().delete_many(where={"asset": "BTCUSDT"})
            await TradeDecision.prisma().delete_many(where={"asset": "XAUUSD"})
            if db.is_connected():
                await db.disconnect()

        asyncio.run(_cleanup())

    def test_pydantic_payload_model(self) -> None:
        """Verify TradingViewPayload schema validation."""
        payload = TradingViewPayload(**self.sample_payload)
        self.assertEqual(payload.asset, "BTCUSDT")
        self.assertEqual(payload.timeframe, "15m")
        self.assertEqual(payload.price, 64500.5)

    def test_format_market_data(self) -> None:
        """Verify format_market_data outputs structured string."""
        payload = TradingViewPayload(**self.sample_payload)
        formatted = format_market_data(payload)
        self.assertIn("Asset: BTCUSDT", formatted)
        self.assertIn("Timeframe: 15m", formatted)
        self.assertIn("Current Price: 64500.5", formatted)
        self.assertIn("Market Structure: Bullish BoS", formatted)
        self.assertIn("Liquidity: SSL swept at 64,000", formatted)
        self.assertIn("Order Block: Bullish OB at 64,100", formatted)
        self.assertIn("Fair Value Gap (FVG): Bullish FVG at 64,200", formatted)

    def test_root_endpoint(self) -> None:
        """Verify GET / returns online status."""
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "online")
            self.assertIn("database", data)

    def test_webhook_invalid_payload(self) -> None:
        """Verify POST /tv-webhook returns 422 for malformed payload."""
        with TestClient(app) as client:
            response = client.post("/tv-webhook", json={"asset": "BTCUSDT"})
            self.assertEqual(response.status_code, 422)

    @patch("src.main.run_multi_agent_analysis_async", new_callable=AsyncMock)
    def test_webhook_success_and_db_persistence(self, mock_agent: AsyncMock) -> None:
        """Verify POST /tv-webhook checks lock, acquires lock, executes analysis, and stores in DB."""
        mock_agent.return_value = self.mock_analysis

        with TestClient(app) as client:
            response = client.post("/tv-webhook", json=self.sample_payload)
            self.assertEqual(response.status_code, 200)

            body = response.json()
            self.assertEqual(body["status"], "success")
            self.assertEqual(body["decision"], self.mock_analysis["judge_decision"])

            # Verify Redis lock check and set
            self.mock_redis.get.assert_called_once_with("signal_lock:BTCUSDT:15m")
            self.mock_redis.set.assert_called_once_with("signal_lock:BTCUSDT:15m", "locked", ex=300)

            # Verify orchestrator was invoked
            mock_agent.assert_called_once()
            call_kwargs = mock_agent.call_args.kwargs
            self.assertEqual(call_kwargs["asset"], "BTCUSDT")

    @patch("src.main.run_multi_agent_analysis_async", new_callable=AsyncMock)
    def test_webhook_cooldown_lock_active(self, mock_agent: AsyncMock) -> None:
        """Verify duplicate webhook during active cooldown returns HTTP 200 ignored response."""
        # Simulate active lock in Redis
        self.mock_redis.get.return_value = "locked"

        with TestClient(app) as client:
            response = client.post("/tv-webhook", json=self.sample_payload)
            self.assertEqual(response.status_code, 200)

            body = response.json()
            self.assertEqual(body["status"], "ignored")
            self.assertEqual(body["message"], "Cooldown active for this asset and timeframe.")

            # Verify Redis get was called
            self.mock_redis.get.assert_called_once_with("signal_lock:BTCUSDT:15m")

            # Verify Redis set, LLM orchestrator, and DB are NEVER called when locked
            self.mock_redis.set.assert_not_called()
            mock_agent.assert_not_called()

    @patch("src.main.run_multi_agent_analysis_async", new_callable=AsyncMock)
    def test_webhook_custom_asset_and_timeframe_lock_key(self, mock_agent: AsyncMock) -> None:
        """Verify unique lock key generation for different assets and timeframes (e.g. XAUUSD:M15)."""
        mock_agent.return_value = self.mock_analysis
        payload = dict(self.sample_payload, asset="XAUUSD", timeframe="M15")

        with TestClient(app) as client:
            response = client.post("/tv-webhook", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "success")

            # Verify key matches signal_lock:XAUUSD:M15
            self.mock_redis.get.assert_called_once_with("signal_lock:XAUUSD:M15")
            self.mock_redis.set.assert_called_once_with("signal_lock:XAUUSD:M15", "locked", ex=300)

    @patch("src.main.run_multi_agent_analysis_async", new_callable=AsyncMock)
    def test_webhook_orchestrator_failure(self, mock_agent: AsyncMock) -> None:
        """Verify POST /tv-webhook handles agent failure with HTTP 500."""
        mock_agent.side_effect = RuntimeError("Claude API unreachable")

        with TestClient(app) as client:
            response = client.post("/tv-webhook", json=self.sample_payload)
            self.assertEqual(response.status_code, 500)
            self.assertIn("Claude API unreachable", response.json()["detail"])

    def test_webhook_redis_failure(self) -> None:
        """Verify POST /tv-webhook handles Redis connection/command failure with HTTP 500."""
        self.mock_redis.get.side_effect = RuntimeError("Redis connection refused")

        with TestClient(app) as client:
            response = client.post("/tv-webhook", json=self.sample_payload)
            self.assertEqual(response.status_code, 500)
            self.assertIn("Redis lock operation failed", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
