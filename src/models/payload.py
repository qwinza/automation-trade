"""Pydantic model for TradingView webhook alerts."""

from pydantic import BaseModel, Field


class TradingViewPayload(BaseModel):
    """Payload schema received from TradingView alerts/webhooks."""

    asset: str = Field(..., description="Trading asset identifier (e.g. 'BTCUSDT', 'EURUSD')")
    timeframe: str = Field(..., description="Chart timeframe (e.g. '15m', '1h', '4h')")
    price: float = Field(..., description="Current market price")
    structure: str = Field(..., description="Market structure condition (e.g. 'Bullish BoS', 'Bearish CHoCH')")
    liquidity: str = Field(..., description="Liquidity status (e.g. 'SSL swept at 64,000', 'BSL resting above')")
    order_block: str = Field(..., description="Order block context (e.g. 'Bullish OB in discount at 64,100')")
    fvg: str = Field(..., description="Fair Value Gap context (e.g. 'Bullish FVG at 64,200')")
    news_context: str = Field(default="None", description="Macroeconomic news or fundamental context (e.g. 'FOMC', 'CPI release')")

    model_config = {
        "json_schema_extra": {
            "example": {
                "asset": "BTCUSDT",
                "timeframe": "15m",
                "price": 64500.5,
                "structure": "Bullish BoS",
                "liquidity": "SSL swept at 64,000",
                "order_block": "Bullish OB in discount at 64,100",
                "fvg": "Bullish FVG at 64,200",
                "news_context": "None",
            }
        }
    }
