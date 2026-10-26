"""Unit tests for Redis client configuration and utility functions."""

import os
import unittest
from unittest.mock import AsyncMock, patch

from src.db.redis import close_redis, get_signal_lock_key, redis_client


class TestRedisConfigAndUtils(unittest.TestCase):
    """Test suite for Redis client configuration and helpers."""

    def test_get_signal_lock_key(self) -> None:
        """Verify lock key generation format for assets and timeframes."""
        self.assertEqual(get_signal_lock_key("BTCUSDT", "15m"), "signal_lock:BTCUSDT:15m")
        self.assertEqual(get_signal_lock_key("XAUUSD", "M15"), "signal_lock:XAUUSD:M15")
        self.assertEqual(get_signal_lock_key("EURUSD", "1h"), "signal_lock:EURUSD:1h")

    @patch("src.db.redis.redis_client.close", new_callable=AsyncMock)
    def test_close_redis_success(self, mock_close: AsyncMock) -> None:
        """Verify close_redis invokes client close method."""
        import asyncio

        asyncio.run(close_redis())
        mock_close.assert_called_once()

    @patch("src.db.redis.redis_client.close", new_callable=AsyncMock)
    def test_close_redis_handles_exception(self, mock_close: AsyncMock) -> None:
        """Verify close_redis suppresses exceptions gracefully."""
        import asyncio

        mock_close.side_effect = RuntimeError("Socket already closed")
        # Should not raise exception
        asyncio.run(close_redis())
        mock_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
