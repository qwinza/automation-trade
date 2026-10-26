"""Unit tests for Fundamental Agent ('The Economist') integration with 9Router OpenAI Gateway and The Judge."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import (
    _build_judge_prompt,
    run_multi_agent_analysis,
    run_multi_agent_analysis_async,
)
from src.agents.prompts import FUNDAMENTAL_AGENT_PROMPT, JUDGE_AGENT_PROMPT
from src.main import format_market_data
from src.models.payload import TradingViewPayload


def _make_mock_completion(content: str):
    """Helper to mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestFundamentalAgent(unittest.IsolatedAsyncioTestCase):
    """Test suite for Fundamental Agent (9Router OpenAI Gateway) and prompt arbitration."""

    def setUp(self) -> None:
        self.sample_payload_clean = {
            "asset": "BTCUSDT",
            "timeframe": "15m",
            "price": 64500.5,
            "structure": "Bullish BoS",
            "liquidity": "SSL swept at 64,000",
            "order_block": "Bullish OB at 64,100",
            "fvg": "Bullish FVG at 64,200",
        }
        self.market_data_clean = (
            "Asset: BTCUSDT\nTimeframe: 15m\nCurrent Price: 64500.5\n"
            "Market Structure: Bullish BoS\nLiquidity: SSL swept at 64,000\n"
            "Order Block: Bullish OB at 64,100\nFair Value Gap (FVG): Bullish FVG at 64,200"
        )

    def test_pydantic_payload_optional_news_context(self) -> None:
        """Verify news_context defaults to 'None' and accepts custom values."""
        payload_default = TradingViewPayload(**self.sample_payload_clean)
        self.assertEqual(payload_default.news_context, "None")

        data_with_news = dict(self.sample_payload_clean, news_context="FOMC Rate Decision: 50bps rate cut announced")
        payload_news = TradingViewPayload(**data_with_news)
        self.assertEqual(payload_news.news_context, "FOMC Rate Decision: 50bps rate cut announced")

    def test_format_market_data_with_and_without_news(self) -> None:
        """Verify format_market_data formats news_context correctly."""
        payload_default = TradingViewPayload(**self.sample_payload_clean)
        formatted_default = format_market_data(payload_default)
        self.assertNotIn("News Context", formatted_default)

        data_with_news = dict(self.sample_payload_clean, news_context="Red News: CPI year-over-year came at 2.4%")
        payload_news = TradingViewPayload(**data_with_news)
        formatted_news = format_market_data(payload_news)
        self.assertIn("News Context: Red News: CPI year-over-year came at 2.4%", formatted_news)

    def test_prompt_contents(self) -> None:
        """Verify FUNDAMENTAL_AGENT_PROMPT and JUDGE_AGENT_PROMPT contents."""
        self.assertIn("Macroeconomic Analyst", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("FOMC", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("NFP", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("CPI", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("PPI", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("Red News", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("DOVISH", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("HAWKISH", FUNDAMENTAL_AGENT_PROMPT)
        self.assertIn("NEUTRAL", FUNDAMENTAL_AGENT_PROMPT)

        self.assertIn("Fundamental Context", JUDGE_AGENT_PROMPT)
        self.assertIn("NEUTRAL", JUDGE_AGENT_PROMPT)
        self.assertIn("CLEAR", JUDGE_AGENT_PROMPT)
        self.assertIn("Veto", JUDGE_AGENT_PROMPT)

    def test_build_judge_prompt_injects_fundamental(self) -> None:
        """Verify _build_judge_prompt injects fundamental_context section into judge prompt."""
        judge_prompt = _build_judge_prompt(
            asset="BTCUSDT",
            market_data="Price: 65000",
            bull_thesis="Long setup",
            bear_thesis="Short setup",
            fundamental_context="DOVISH: FOMC announced surprise rate cut.",
        )
        self.assertIn("[FUNDAMENTAL CONTEXT (MACRO / NEWS)]", judge_prompt)
        self.assertIn("DOVISH: FOMC announced surprise rate cut.", judge_prompt)
        self.assertIn("[THE BULL THESIS (LONG SETUP)]", judge_prompt)
        self.assertIn("[THE BEAR THESIS (SHORT SETUP)]", judge_prompt)

    @patch("src.agents.orchestrator.client.chat.completions.create", new_callable=AsyncMock)
    async def test_async_orchestrator_clear_when_no_red_news(
        self,
        mock_chat_create: AsyncMock,
    ) -> None:
        """When no red news keywords are present, Economist is NOT invoked and context is CLEAR."""
        mock_resp_bull = _make_mock_completion("Bull thesis")
        mock_resp_bear = _make_mock_completion("Bear thesis")
        mock_resp_judge = _make_mock_completion('{"action": "BUY", "confidence": 85, "reasoning": "Valid Long"}')

        mock_chat_create.side_effect = [mock_resp_bull, mock_resp_bear, mock_resp_judge]

        payload = TradingViewPayload(**self.sample_payload_clean)
        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=self.market_data_clean,
            payload=payload,
        )

        # Chat completions called exactly 3 times (Bull, Bear, Judge)
        self.assertEqual(mock_chat_create.call_count, 3)
        self.assertEqual(mock_chat_create.call_args_list[0].kwargs["model"], "claude-sonnet-4-6")
        self.assertEqual(mock_chat_create.call_args_list[1].kwargs["model"], "claude-sonnet-4-6")
        self.assertEqual(mock_chat_create.call_args_list[2].kwargs["model"], "claude-sonnet-4-6")

        self.assertEqual(result["fundamental"], "CLEAR: No high-impact red news. Rely entirely on technicals.")
        self.assertEqual(result["judge"]["action"], "BUY")

    @patch("src.agents.orchestrator.client.chat.completions.create", new_callable=AsyncMock)
    async def test_async_orchestrator_invokes_gemini_on_red_news(
        self,
        mock_chat_create: AsyncMock,
    ) -> None:
        """When red news keywords (e.g. FOMC, CPI) are present, Economist is invoked via gemini-1.5-flash and passed to Judge."""
        mock_resp_econ = _make_mock_completion("DOVISH (Bullish for Gold/Crypto): Rate pause with dovish forward guidance.")
        mock_resp_bull = _make_mock_completion("Bull SMC analysis")
        mock_resp_bear = _make_mock_completion("Bear SMC analysis")
        mock_resp_judge = _make_mock_completion('{"action": "BUY", "confidence": 90, "reasoning": "Bullish tech + Dovish FOMC"}')

        mock_chat_create.side_effect = [mock_resp_econ, mock_resp_bull, mock_resp_bear, mock_resp_judge]

        data_with_fomc = dict(self.sample_payload_clean, news_context="FOMC statement release at 2:00 PM EST")
        payload = TradingViewPayload(**data_with_fomc)

        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=self.market_data_clean,
            payload=payload,
        )

        # 4 total calls: Economist, Bull, Bear, Judge
        self.assertEqual(mock_chat_create.call_count, 4)
        first_call = mock_chat_create.call_args_list[0]
        self.assertEqual(first_call.kwargs["model"], "gemini-1.5-flash")
        called_user_content = first_call.kwargs["messages"][1]["content"]
        self.assertIn("FOMC statement release", called_user_content)

        # Verify fundamental analysis was updated
        self.assertEqual(
            result["fundamental"],
            "DOVISH (Bullish for Gold/Crypto): Rate pause with dovish forward guidance.",
        )
        self.assertEqual(result["judge"]["action"], "BUY")

        # Verify Judge prompt received fundamental analysis
        judge_call = mock_chat_create.call_args_list[3]
        judge_user_content = judge_call.kwargs["messages"][1]["content"]
        self.assertIn("[FUNDAMENTAL CONTEXT (MACRO / NEWS)]", judge_user_content)
        self.assertIn("DOVISH (Bullish for Gold/Crypto)", judge_user_content)

    @patch("src.agents.orchestrator.client.chat.completions.create", new_callable=AsyncMock)
    async def test_async_orchestrator_detects_keywords_in_market_data(
        self,
        mock_chat_create: AsyncMock,
    ) -> None:
        """When keyword like 'CPI' is directly in market_data string, Economist is triggered even if payload is None."""
        mock_resp_econ = _make_mock_completion("HAWKISH (Bearish for Gold/Crypto): Core CPI hot.")
        mock_resp_bull = _make_mock_completion("Bull thesis")
        mock_resp_bear = _make_mock_completion("Bear thesis")
        mock_resp_judge = _make_mock_completion('{"action": "SELL", "confidence": 88, "reasoning": "Hawkish CPI + Bear OB"}')

        mock_chat_create.side_effect = [mock_resp_econ, mock_resp_bull, mock_resp_bear, mock_resp_judge]

        market_data_with_cpi = self.market_data_clean + "\nEvent: High impact Red News CPI print coming up"
        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=market_data_with_cpi,
            payload=None,
        )

        self.assertEqual(mock_chat_create.call_count, 4)
        self.assertEqual(mock_chat_create.call_args_list[0].kwargs["model"], "gemini-1.5-flash")
        self.assertEqual(result["fundamental"], "HAWKISH (Bearish for Gold/Crypto): Core CPI hot.")
        self.assertEqual(result["judge"]["action"], "SELL")

    @patch("src.agents.orchestrator.client.chat.completions.create", new_callable=AsyncMock)
    async def test_async_orchestrator_router_error_fallback(
        self,
        mock_chat_create: AsyncMock,
    ) -> None:
        """When Economist call fails on 9Router, fallback to 'NEUTRAL: Router Error' safely injected to Judge."""
        mock_resp_bull = _make_mock_completion("Bull thesis")
        mock_resp_bear = _make_mock_completion("Bear thesis")
        mock_resp_judge = _make_mock_completion('{"action": "HOLD", "confidence": 50, "reasoning": "Relying on technicals"}')

        # First call (Economist) fails with an API error
        mock_chat_create.side_effect = [
            RuntimeError("502 Bad Gateway from 9Router"),
            mock_resp_bull,
            mock_resp_bear,
            mock_resp_judge,
        ]

        data_with_news = dict(self.sample_payload_clean, news_context="Red News: PPI inflation spikes")
        payload = TradingViewPayload(**data_with_news)

        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=self.market_data_clean,
            payload=payload,
        )

        self.assertEqual(result["fundamental"], "NEUTRAL: Router Error")
        judge_call = mock_chat_create.call_args_list[3]
        judge_user_content = judge_call.kwargs["messages"][1]["content"]
        self.assertIn("[FUNDAMENTAL CONTEXT (MACRO / NEWS)]", judge_user_content)
        self.assertIn("NEUTRAL: Router Error", judge_user_content)


if __name__ == "__main__":
    unittest.main()
