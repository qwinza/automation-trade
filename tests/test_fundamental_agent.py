"""Unit tests for Fundamental Agent ('The Economist') integration with Gemini API and The Judge."""

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


class TestFundamentalAgent(unittest.IsolatedAsyncioTestCase):
    """Test suite for Fundamental Agent (Gemini API) and prompt arbitration."""

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

    @patch("src.agents.orchestrator.genai.GenerativeModel")
    @patch("src.agents.orchestrator.AsyncAnthropic")
    async def test_async_orchestrator_clear_when_no_red_news(
        self,
        mock_anthropic_cls: MagicMock,
        mock_genai_cls: MagicMock,
    ) -> None:
        """When no red news keywords are present, Gemini is NOT invoked and context is CLEAR."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_resp_bull = MagicMock(content=[MagicMock(text="Bull thesis")])
        mock_resp_bear = MagicMock(content=[MagicMock(text="Bear thesis")])
        mock_resp_judge = MagicMock(
            content=[MagicMock(text='{"action": "BUY", "confidence": 85, "reasoning": "Valid Long"}')]
        )
        mock_client.messages.create = AsyncMock(side_effect=[mock_resp_bull, mock_resp_bear, mock_resp_judge])

        payload = TradingViewPayload(**self.sample_payload_clean)
        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=self.market_data_clean,
            payload=payload,
        )

        # Gemini should NOT be invoked
        mock_genai_cls.assert_not_called()

        self.assertEqual(result["fundamental"], "CLEAR: No high-impact red news. Rely entirely on technicals.")
        self.assertEqual(result["judge"]["action"], "BUY")

    @patch("src.agents.orchestrator.genai.GenerativeModel")
    @patch("src.agents.orchestrator.AsyncAnthropic")
    async def test_async_orchestrator_invokes_gemini_on_red_news(
        self,
        mock_anthropic_cls: MagicMock,
        mock_genai_cls: MagicMock,
    ) -> None:
        """When red news keywords (e.g. FOMC, CPI) are present, Gemini is invoked and passed to Judge."""
        # Setup Gemini mock
        mock_gemini_instance = MagicMock()
        mock_genai_cls.return_value = mock_gemini_instance
        mock_gemini_resp = MagicMock()
        mock_gemini_resp.text = "DOVISH (Bullish for Gold/Crypto): Rate pause with dovish forward guidance."
        mock_gemini_instance.generate_content_async = AsyncMock(return_value=mock_gemini_resp)

        # Setup Anthropic mock
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_resp_bull = MagicMock(content=[MagicMock(text="Bull SMC analysis")])
        mock_resp_bear = MagicMock(content=[MagicMock(text="Bear SMC analysis")])
        mock_resp_judge = MagicMock(
            content=[MagicMock(text='{"action": "BUY", "confidence": 90, "reasoning": "Bullish tech + Dovish FOMC"}')]
        )
        mock_client.messages.create = AsyncMock(side_effect=[mock_resp_bull, mock_resp_bear, mock_resp_judge])

        data_with_fomc = dict(self.sample_payload_clean, news_context="FOMC statement release at 2:00 PM EST")
        payload = TradingViewPayload(**data_with_fomc)

        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=self.market_data_clean,
            payload=payload,
        )

        # Verify Gemini was invoked initially with gemini-1.5-flash-latest
        mock_genai_cls.assert_called_with("gemini-1.5-flash-latest")
        mock_gemini_instance.generate_content_async.assert_called_once()
        called_prompt = mock_gemini_instance.generate_content_async.call_args[0][0]
        self.assertIn("FOMC statement release", called_prompt)

        # Verify fundamental analysis was updated
        self.assertEqual(
            result["fundamental"],
            "DOVISH (Bullish for Gold/Crypto): Rate pause with dovish forward guidance.",
        )
        self.assertEqual(result["judge"]["action"], "BUY")

        # Verify Judge prompt received fundamental analysis
        judge_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        judge_user_content = judge_call_kwargs["messages"][0]["content"]
        self.assertIn("[FUNDAMENTAL CONTEXT (MACRO / NEWS)]", judge_user_content)
        self.assertIn("DOVISH (Bullish for Gold/Crypto)", judge_user_content)

    @patch("src.agents.orchestrator.genai.GenerativeModel")
    @patch("src.agents.orchestrator.AsyncAnthropic")
    async def test_async_orchestrator_detects_keywords_in_market_data(
        self,
        mock_anthropic_cls: MagicMock,
        mock_genai_cls: MagicMock,
    ) -> None:
        """When keyword like 'CPI' is directly in market_data string, Gemini is triggered even if payload is None."""
        mock_gemini_instance = MagicMock()
        mock_genai_cls.return_value = mock_gemini_instance
        mock_gemini_resp = MagicMock()
        mock_gemini_resp.text = "HAWKISH (Bearish for Gold/Crypto): Core CPI hot."
        mock_gemini_instance.generate_content_async = AsyncMock(return_value=mock_gemini_resp)

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_resp_bull = MagicMock(content=[MagicMock(text="Bull thesis")])
        mock_resp_bear = MagicMock(content=[MagicMock(text="Bear thesis")])
        mock_resp_judge = MagicMock(
            content=[MagicMock(text='{"action": "SELL", "confidence": 88, "reasoning": "Hawkish CPI + Bear OB"}')]
        )
        mock_client.messages.create = AsyncMock(side_effect=[mock_resp_bull, mock_resp_bear, mock_resp_judge])

        market_data_with_cpi = self.market_data_clean + "\nEvent: High impact Red News CPI print coming up"
        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=market_data_with_cpi,
            payload=None,
        )

        mock_genai_cls.assert_called_with("gemini-1.5-flash-latest")
        self.assertEqual(result["fundamental"], "HAWKISH (Bearish for Gold/Crypto): Core CPI hot.")
        self.assertEqual(result["judge"]["action"], "SELL")

    @patch("src.agents.orchestrator.genai.GenerativeModel")
    @patch("src.agents.orchestrator.AsyncAnthropic")
    async def test_async_orchestrator_notfound_fallback(
        self,
        mock_anthropic_cls: MagicMock,
        mock_genai_cls: MagicMock,
    ) -> None:
        """When gemini-1.5-flash-latest raises NotFound (404), orchestrator falls back to gemini-pro / gemini-1.5-pro-latest."""
        from google.api_core.exceptions import NotFound

        # Setup mock behavior:
        # 1st call for gemini-1.5-flash-latest raises NotFound
        # 2nd call for gemini-pro succeeds
        mock_primary = MagicMock()
        mock_primary.generate_content_async = AsyncMock(
            side_effect=NotFound("404 models/gemini-1.5-flash-latest is not found")
        )

        mock_fallback = MagicMock()
        mock_fb_resp = MagicMock()
        mock_fb_resp.text = "HAWKISH (Bearish for Gold/Crypto): Rate hike expectation."
        mock_fallback.generate_content_async = AsyncMock(return_value=mock_fb_resp)

        def _model_side_effect(name):
            if name == "gemini-1.5-flash-latest":
                return mock_primary
            elif name == "gemini-pro":
                return mock_fallback
            return MagicMock()

        mock_genai_cls.side_effect = _model_side_effect

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_resp_bull = MagicMock(content=[MagicMock(text="Bull thesis")])
        mock_resp_bear = MagicMock(content=[MagicMock(text="Bear thesis")])
        mock_resp_judge = MagicMock(
            content=[MagicMock(text='{"action": "SELL", "confidence": 80, "reasoning": "Fallback Hawkish"}')]
        )
        mock_client.messages.create = AsyncMock(side_effect=[mock_resp_bull, mock_resp_bear, mock_resp_judge])

        data_with_news = dict(self.sample_payload_clean, news_context="Red News: NFP jobs report surging")
        payload = TradingViewPayload(**data_with_news)

        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=self.market_data_clean,
            payload=payload,
        )

        # Verify fallback succeeded and updated fundamental analysis
        self.assertEqual(result["fundamental"], "HAWKISH (Bearish for Gold/Crypto): Rate hike expectation.")

        # Verify Judge received the fallback analysis
        judge_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        judge_user_content = judge_call_kwargs["messages"][0]["content"]
        self.assertIn("HAWKISH (Bearish for Gold/Crypto): Rate hike expectation.", judge_user_content)

    @patch("src.agents.orchestrator.genai.GenerativeModel")
    @patch("src.agents.orchestrator.AsyncAnthropic")
    async def test_async_orchestrator_all_models_fail_safe_injection(
        self,
        mock_anthropic_cls: MagicMock,
        mock_genai_cls: MagicMock,
    ) -> None:
        """When all models fail, NEUTRAL error message is safely injected into Judge prompt without crashing."""
        from google.api_core.exceptions import NotFound

        mock_failing = MagicMock()
        mock_failing.generate_content_async = AsyncMock(side_effect=NotFound("Model not found"))
        mock_genai_cls.return_value = mock_failing

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_resp_bull = MagicMock(content=[MagicMock(text="Bull thesis")])
        mock_resp_bear = MagicMock(content=[MagicMock(text="Bear thesis")])
        mock_resp_judge = MagicMock(
            content=[MagicMock(text='{"action": "HOLD", "confidence": 50, "reasoning": "Relying on technicals"}')]
        )
        mock_client.messages.create = AsyncMock(side_effect=[mock_resp_bull, mock_resp_bear, mock_resp_judge])

        data_with_news = dict(self.sample_payload_clean, news_context="Red News: PPI inflation spikes")
        payload = TradingViewPayload(**data_with_news)

        result = await run_multi_agent_analysis_async(
            asset="BTCUSDT",
            market_data=self.market_data_clean,
            payload=payload,
        )

        self.assertIn("NEUTRAL", result["fundamental"])
        judge_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        judge_user_content = judge_call_kwargs["messages"][0]["content"]
        self.assertIn("[FUNDAMENTAL CONTEXT (MACRO / NEWS)]", judge_user_content)
        self.assertIn("NEUTRAL", judge_user_content)


if __name__ == "__main__":
    unittest.main()
