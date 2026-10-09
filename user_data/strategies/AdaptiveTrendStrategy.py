# pragma pylint: disable=missing-docstring, invalid-name
"""
AdaptiveTrendStrategy
=====================

A regime-aware, risk-managed strategy for Freqtrade.

What it does differently from SampleStrategy:

* Market regime filter
    - Higher-timeframe (4h) trend: EMA50/EMA200 + ADX decides whether we are in
      an uptrend, downtrend or range.
    - BTC filter: no new longs while BTC itself trades below its 1h EMA (most
      altcoins dump together with BTC).
* Three entry setups, each enabled only in the regime it works in
    - ``trend_pullback`` : buy the dip to EMA20 inside an established trend.
    - ``breakout``       : Donchian breakout confirmed by volume and rising ADX.
    - ``mean_rev``       : Bollinger re-entry + oversold RSI, only in a range.
* Volatility-based risk management
    - Initial stop = entry -/+ ATR * multiplier (not a fixed percentage).
    - Stop moves to break-even after +1R, then trails as a chandelier stop.
    - Takes partial profit (default 50%) at +1.5R and lets the rest run.
* Volatility-based position sizing
    - Every trade risks a fixed fraction of the wallet (default 1%), so a
      volatile coin automatically gets a smaller stake than a calm one.
* Built-in protections (cooldown, stoploss guard, max drawdown, low-profit pairs).

Run in spot (long only) as ``AdaptiveTrendStrategy`` or in futures (long + short)
as ``AdaptiveTrendFuturesStrategy``.

No strategy is guaranteed to be profitable. Always backtest, hyperopt and
dry-run before trading real money:

    freqtrade download-data -c user_data/config.json --timeframes 15m 1h 4h --days 400
    freqtrade backtesting  -c user_data/config.json -s AdaptiveTrendStrategy --timerange 20250101-
    freqtrade hyperopt     -c user_data/config.json -s AdaptiveTrendStrategy \
        --hyperopt-loss SortinoHyperOptLossDaily --spaces buy sell -e 300
"""

from datetime import datetime, timedelta

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Order, Trade
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_absolute,
)


class AdaptiveTrendStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    inf_timeframe = "4h"
    btc_timeframe = "1h"

    can_short = False

    # Exits are managed by the ATR stop, partial take-profit and exit signals.
    # ROI only acts as an emergency "too good to be true" exit.
    minimal_roi = {"0": 0.50}

    # Hard stop - the dynamic ATR stop below is always tighter than this.
    stoploss = -0.15
    use_custom_stoploss = True
    trailing_stop = False

    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    process_only_new_candles = True
    # Warm-up candles (also applied to the 4h / 1h informative timeframes in backtesting).
    startup_candle_count: int = 400

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # ---------------------------------------------------------------- params
    # Regime
    use_btc_filter = BooleanParameter(default=True, space="buy", optimize=True)
    adx_trend = IntParameter(15, 30, default=20, space="buy", optimize=True)

    # Setup toggles
    enable_pullback = BooleanParameter(default=True, space="buy", optimize=False)
    enable_breakout = BooleanParameter(default=True, space="buy", optimize=False)
    enable_mean_rev = BooleanParameter(default=True, space="buy", optimize=False)

    # Pullback
    pullback_rsi = IntParameter(38, 55, default=45, space="buy", optimize=True)
    # Breakout
    breakout_vol_mult = DecimalParameter(1.2, 3.0, default=1.6, decimals=1, space="buy")
    # Mean reversion
    mr_rsi = IntParameter(20, 40, default=32, space="buy", optimize=True)

    # Risk management
    atr_stop_mult = DecimalParameter(1.5, 4.0, default=2.5, decimals=1, space="sell")
    chandelier_mult = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell")
    breakeven_r = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="sell")
    tp1_r = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell")
    tp1_fraction = DecimalParameter(0.25, 0.75, default=0.5, decimals=2, space="sell")
    stale_hours = IntParameter(12, 72, default=36, space="sell", optimize=True)

    # Fraction of the total wallet risked per trade (distance entry -> stop).
    risk_per_trade = 0.01

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 96,
                "trade_limit": 3,
                "stop_duration_candles": 48,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 10,
                "stop_duration_candles": 96,
                "max_allowed_drawdown": 0.12,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 672,
                "trade_limit": 3,
                "stop_duration_candles": 192,
                "required_profit": -0.02,
            },
        ]

    plot_config = {
        "main_plot": {
            "ema20": {"color": "orange"},
            "ema50": {"color": "blue"},
            "bb_lower": {"color": "grey"},
            "bb_upper": {"color": "grey"},
            "ema200_4h": {"color": "red"},
        },
        "subplots": {
            "ADX": {"adx": {"color": "purple"}},
            "RSI": {"rsi": {"color": "green"}},
        },
    }

    # ------------------------------------------------------------- helpers
    def _btc_pair(self) -> str:
        stake = self.config["stake_currency"]
        if self.config.get("trading_mode", "spot") == "futures":
            return f"BTC/{stake}:{stake}"
        return f"BTC/{stake}"

    def informative_pairs(self):
        pairs = self.dp.current_whitelist() if self.dp else []
        inf = [(pair, self.inf_timeframe) for pair in pairs]
        inf.append((self._btc_pair(), self.btc_timeframe))
        return inf

    # ---------------------------------------------------------- indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Higher timeframe regime
        inf = self.dp.get_pair_dataframe(metadata["pair"], self.inf_timeframe)
        if not inf.empty:
            inf["ema50"] = ta.EMA(inf, timeperiod=50)
            inf["ema200"] = ta.EMA(inf, timeperiod=200)
            inf["adx"] = ta.ADX(inf, timeperiod=14)
            dataframe = merge_informative_pair(
                dataframe, inf, self.timeframe, self.inf_timeframe, ffill=True
            )
        else:
            for col in ("close", "ema50", "ema200", "adx"):
                dataframe[f"{col}_{self.inf_timeframe}"] = np.nan

        # --- BTC market filter
        btc = self.dp.get_pair_dataframe(self._btc_pair(), self.btc_timeframe)
        if not btc.empty:
            btc["ema100"] = ta.EMA(btc, timeperiod=100)
            btc["rsi"] = ta.RSI(btc, timeperiod=14)
            btc = btc[["date", "close", "ema100", "rsi"]].rename(
                columns={"close": "btc_close", "ema100": "btc_ema100", "rsi": "btc_rsi"}
            )
            dataframe = merge_informative_pair(
                dataframe, btc, self.timeframe, self.btc_timeframe, ffill=True
            )
            sfx = f"_{self.btc_timeframe}"
            dataframe["btc_bull"] = dataframe[f"btc_close{sfx}"] > dataframe[f"btc_ema100{sfx}"]
            dataframe["btc_bear"] = dataframe[f"btc_close{sfx}"] < dataframe[f"btc_ema100{sfx}"]
        else:
            # No BTC data (e.g. BTC not tradable on this exchange): filter is neutral.
            dataframe["btc_bull"] = True
            dataframe["btc_bear"] = True

        # --- Base timeframe
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]

        dataframe["dc_high"] = dataframe["high"].rolling(20).max().shift(1)
        dataframe["dc_low"] = dataframe["low"].rolling(20).min().shift(1)
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        # --- Regime classification on the 4h timeframe
        s = f"_{self.inf_timeframe}"
        dataframe["regime_up"] = (dataframe[f"close{s}"] > dataframe[f"ema200{s}"]) & (
            dataframe[f"ema50{s}"] > dataframe[f"ema200{s}"]
        )
        dataframe["regime_down"] = (dataframe[f"close{s}"] < dataframe[f"ema200{s}"]) & (
            dataframe[f"ema50{s}"] < dataframe[f"ema200{s}"]
        )
        dataframe["regime_range"] = dataframe[f"adx{s}"] < 20

        return dataframe

    # --------------------------------------------------------------- entries
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        df["enter_long"] = 0
        if self.can_short:
            df["enter_short"] = 0
        has_volume = df["volume"] > 0
        btc_ok_long = df["btc_bull"] if self.use_btc_filter.value else True
        btc_ok_short = df["btc_bear"] if self.use_btc_filter.value else True
        adx_min = self.adx_trend.value

        # 1) Trend pullback: dip to EMA20 inside an uptrend, RSI turning back up
        if self.enable_pullback.value:
            touched_ema = (df["low"] <= df["ema20"]).rolling(3).max() > 0
            long_pb = (
                df["regime_up"]
                & btc_ok_long
                & (df["ema20"] > df["ema50"])
                & (df["close"] > df["ema50"])
                & (df["adx"] > adx_min)
                & touched_ema
                & (df["rsi"].shift(1) < self.pullback_rsi.value)
                & (df["rsi"] >= self.pullback_rsi.value)
                & (df["close"] > df["open"])
                & has_volume
            )
            df.loc[long_pb, ["enter_long", "enter_tag"]] = (1, "trend_pullback")

            if self.can_short:
                touched_ema_s = (df["high"] >= df["ema20"]).rolling(3).max() > 0
                short_rsi = 100 - self.pullback_rsi.value
                short_pb = (
                    df["regime_down"]
                    & btc_ok_short
                    & (df["ema20"] < df["ema50"])
                    & (df["close"] < df["ema50"])
                    & (df["adx"] > adx_min)
                    & touched_ema_s
                    & (df["rsi"].shift(1) > short_rsi)
                    & (df["rsi"] <= short_rsi)
                    & (df["close"] < df["open"])
                    & has_volume
                )
                df.loc[short_pb, ["enter_short", "enter_tag"]] = (1, "trend_pullback")

        # 2) Volume-confirmed Donchian breakout
        if self.enable_breakout.value:
            vol_ok = df["volume"] > df["vol_sma"] * self.breakout_vol_mult.value
            adx_rising = (df["adx"] > adx_min) & (df["adx"] > df["adx"].shift(2))
            long_bo = (
                df["regime_up"]
                & btc_ok_long
                & (df["close"] > df["dc_high"])
                & vol_ok
                & adx_rising
                & (df["rsi"] < 75)
                & (df["enter_long"] != 1)
            )
            df.loc[long_bo, ["enter_long", "enter_tag"]] = (1, "breakout")

            if self.can_short:
                short_bo = (
                    df["regime_down"]
                    & btc_ok_short
                    & (df["close"] < df["dc_low"])
                    & vol_ok
                    & adx_rising
                    & (df["rsi"] > 25)
                    & (df["enter_short"] != 1)
                )
                df.loc[short_bo, ["enter_short", "enter_tag"]] = (1, "breakout")

        # 3) Mean reversion: only in a ranging higher timeframe, long side only
        if self.enable_mean_rev.value:
            long_mr = (
                df["regime_range"]
                & btc_ok_long
                & (df["close"].shift(1) < df["bb_lower"].shift(1))
                & (df["close"] > df["bb_lower"])
                & (df["rsi"].shift(1) < self.mr_rsi.value)
                & (df["mfi"] < 35)
                & has_volume
                & (df["enter_long"] != 1)
            )
            df.loc[long_mr, ["enter_long", "enter_tag"]] = (1, "mean_rev")

        return df

    # ----------------------------------------------------------------- exits
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        # Trend broken: EMA20 crosses below EMA50 and price is below both
        df.loc[
            (df["ema20"] < df["ema50"])
            & (df["ema20"].shift(1) >= df["ema50"].shift(1))
            & (df["close"] < df["ema50"])
            & (df["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_break")

        if self.can_short:
            df.loc[
                (df["ema20"] > df["ema50"])
                & (df["ema20"].shift(1) <= df["ema50"].shift(1))
                & (df["close"] > df["ema50"])
                & (df["volume"] > 0),
                ["exit_short", "exit_tag"],
            ] = (1, "trend_break")
        return df

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None
        last = df.iloc[-1]

        # Mean-reversion trades target the Bollinger mid-band, not a trend.
        if trade.enter_tag == "mean_rev" and not trade.is_short:
            if current_rate >= last["bb_mid"] and current_profit > 0:
                return "mr_target"
            if last["rsi"] > 70:
                return "mr_overbought"

        # Dead trades: tie up capital without moving - free the slot.
        r_pct = self._risk_pct(trade)
        is_old = current_time - trade.open_date_utc > timedelta(hours=self.stale_hours.value)
        if is_old and current_profit < 0.3 * r_pct:
            return "stale"
        return None

    # ------------------------------------------------------- risk management
    def _risk_pct(self, trade: Trade) -> float:
        """Initial risk (entry -> initial stop) as a ratio of the entry price."""
        init_stop = trade.get_custom_data("init_stop")
        if not init_stop:
            return abs(self.stoploss)
        return abs(trade.open_rate - init_stop) / trade.open_rate

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> None:
        if order.ft_order_side != trade.entry_side or trade.get_custom_data("init_stop"):
            return
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty or np.isnan(df.iloc[-1]["atr"]):
            return
        dist = df.iloc[-1]["atr"] * self.atr_stop_mult.value
        # Never risk more than the hard stoploss allows.
        dist = min(dist, trade.open_rate * abs(self.stoploss) * 0.95)
        init_stop = trade.open_rate + dist if trade.is_short else trade.open_rate - dist
        trade.set_custom_data("init_stop", float(init_stop))

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        init_stop = trade.get_custom_data("init_stop")
        if not init_stop:
            return None

        r_pct = self._risk_pct(trade)
        stop = init_stop
        sign = -1 if trade.is_short else 1

        # Lock in break-even (+ fees) after +1R.
        if current_profit >= self.breakeven_r.value * r_pct:
            fee_buffer = (
                trade.open_rate * ((trade.fee_open or 0.001) + (trade.fee_close or 0.001)) * 1.5
            )
            be = trade.open_rate + sign * fee_buffer
            stop = max(stop, be) if not trade.is_short else min(stop, be)

        # Chandelier trailing stop once the trade has proven itself.
        if current_profit >= self.tp1_r.value * r_pct:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not df.empty and not np.isnan(df.iloc[-1]["atr"]):
                atr = df.iloc[-1]["atr"]
                if trade.is_short:
                    chandelier = trade.min_rate + atr * self.chandelier_mult.value
                    stop = min(stop, chandelier)
                else:
                    chandelier = trade.max_rate - atr * self.chandelier_mult.value
                    stop = max(stop, chandelier)

        return stoploss_from_absolute(
            stop, current_rate, is_short=trade.is_short, leverage=trade.leverage
        )

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | tuple[float | None, str | None] | None:
        # Partial take profit at tp1_r * R, once per trade. Never add to positions.
        if trade.has_open_orders or trade.nr_of_successful_exits > 0:
            return None
        if trade.enter_tag == "mean_rev":
            return None
        if current_profit >= self.tp1_r.value * self._risk_pct(trade):
            amount = -(trade.stake_amount * self.tp1_fraction.value)
            if min_stake and abs(amount) < min_stake:
                return None
            # Do not leave a remainder too small to be sold later.
            if min_stake and trade.stake_amount + amount < min_stake:
                return None
            return amount, "tp1"
        return None

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Size the position so that hitting the ATR stop costs ``risk_per_trade`` of equity."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty or np.isnan(df.iloc[-1]["atr"]) or current_rate <= 0:
            return proposed_stake
        stop_pct = df.iloc[-1]["atr"] * self.atr_stop_mult.value / current_rate
        stop_pct = min(max(stop_pct, 0.005), abs(self.stoploss))

        equity = self.wallets.get_total_stake_amount() if self.wallets else 0
        if equity <= 0:
            return proposed_stake
        # Price risk is multiplied by leverage, so the margin needed shrinks accordingly.
        stake = equity * self.risk_per_trade / (stop_pct * leverage)
        stake = min(stake, proposed_stake, max_stake)
        if min_stake:
            stake = max(stake, min_stake)
        return stake

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        # Skip entries when the price already ran away from the signal candle (> 1%).
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return True
        close = df.iloc[-1]["close"]
        if side == "long" and rate > close * 1.01:
            return False
        return not (side == "short" and rate < close * 0.99)


class AdaptiveTrendFuturesStrategy(AdaptiveTrendStrategy):
    """Futures variant: trades both directions with conservative leverage."""

    can_short = True
    max_leverage = 3.0

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(self.max_leverage, max_leverage)
