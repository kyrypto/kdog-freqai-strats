import logging
from functools import reduce
from datetime import datetime
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)


class RL_futures_custom(IStrategy):
    """
Here be stonks
1. freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy RL_kdog_futures --freqaimodel ReinforcementLearner --spaces sell roi stoploss --timerange "$(date --date='-1 week' '+%Y%m%d')"-"$(date '+%Y%m%d')" -e 1000
2. freqtrade trade --logfile ./logs --freqaimodel ReinforcementLearner_multiproc --strategy RL_kdog_futures
    """

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "prediction": {"prediction": {"color": "blue"}},
            "target_roi": {
                "target_roi": {"color": "brown"},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
            "&-action": {
                "&-action": {"color": "green"},
            },
        },
    }
    position_adjustment_enable = True
    stoploss = -1
    max_entry_position_adjustment = 5
    max_dca_multiplier = 10
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True
    # Specific variables
    leverage_optimize = True
    leverage_num = DecimalParameter(1, 20, default=5, decimals=2, space='sell', optimize=leverage_optimize)
    linear_roi_offset = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=False, load=True
    )
    max_roi_time_long = IntParameter(0, 800, default=400, space="sell", optimize=False, load=True)
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        trade_date = timeframe_to_prev_date(
            self.timeframe, (trade.open_date_utc - timedelta(minutes=int(self.timeframe[:-1])))
        )
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]
        trade_df = dataframe.loc[(dataframe["date"] > trade_date)]
        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()
        entry_tag = trade.enter_tag

        if trade_df['&-action'].sum() > 0 and entry_tag == 'enter_long':
            return '&-action_missed'

        if trade_df['&-action'].sum() > 0 and entry_tag == 'enter_short':
            return '&-action_missed'

        if last_candle["doPredict"] <= -1:
            return "Outlier detected  (do_predict = -1)"

        if current_profit < -0.07:
            return "stop_loss"

        if last_candle['&-action'] == 1 and entry_tag == 'enter_short':
            return "&-action_detected_short"

        if last_candle['&-action'] == 1 and entry_tag == 'enter_long':
            return "&-action_detected_long"


    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str,
        side: str,
        **kwargs
    ) -> bool:

        open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

        num_shorts, num_longs = 0, 0
        for trade in open_trades:
            if trade.enter_tag == 'short':
                num_shorts += 1
            elif trade.enter_tag == 'long':
                num_longs += 1

        if side == "long" and num_longs >= 4:
            return False

        if side == "short" and num_shorts >= 4:
            return False

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """

        if current_profit > -0.03:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        entry_tag = trade.enter_tag

        filled_buys = trade.select_filled_orders()
        count_of_buys = trade.nr_of_successful_entries

        try:
            stake_amount = filled_buys[0].cost
            stake_amount = stake_amount * (1 + (count_of_buys * 0.25))

            if (last_candle["&-action"] or previous_candle["&-action"]) and entry_tag == 'enter_long':
                return stake_amount
            if (last_candle["&-action"] or previous_candle["&-action"]) and entry_tag == 'enter_short':
                return stake_amount

        except Exception as exception:
            logger.warning(f"{exception}")

        return None
        # This is called when placing the initial order (opening trade)
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return self.leverage_num.value
    def informative_pairs(self):
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue  # avoid duplication
                informative_pairs.append((pair, tf))
        return informative_pairs

    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):

        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:
            t = int(t)
            # MIN - Lowest value over a specified period
            # informative[f"%-{coin}min-period_{t}"] = ta.MIN(informative, timeperiod=t)
            # MAX - Highest value over a specified period
            # informative[f"%-{coin}max-period_{t}"] = ta.MAX(informative, timeperiod=t)
            # DEMA - Double Exponential Moving Average
            informative[f"%-{coin}dema-period_{t}"] = ta.DEMA(informative, timeperiod=t)
            # Linear Regression
            informative[f"%-{coin}linearreg-period_{t}"] = ta.LINEARREG(informative, timeperiod=t)
            informative[f"%-{coin}linearreg_angle-period_{t}"] = ta.LINEARREG_ANGLE(informative, timeperiod=t)
            informative[f"%-{coin}linearreg_intercept-period_{t}"] = ta.LINEARREG_INTERCEPT(informative, timeperiod=t)
            informative[f"%-{coin}linearreg_slope-period_{t}"] = ta.LINEARREG_SLOPE(informative, timeperiod=t)
            # CORREL - Pearson's Correlation Coefficient (r)
            # informative[f"%-{coin}correl-period_{t}"] = ta.CORREL(informative, timeperiod=t)
            # STDDEV - Standard Deviation
            # informative[f"%-{coin}stddev-period_{t}"] = ta.STDDEV(informative, timeperiod=t)
            # TSF - Time Series Forecast
            informative[f"%-{coin}tsf-period_{t}"] = ta.TSF(informative, timeperiod=t)
            # VAR - Variance
            informative[f"%-{coin}var-period_{t}"] = ta.VAR(informative, timeperiod=t)
            # Momentum
            informative[f"%-{coin}mom-period_{t}"] = ta.MOM(informative, timeperiod=t)
            # RSI
            informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            # Commodity Channel Index
            informative[f"%-{coin}cci-period_{t}"] = ta.CCI(informative, timeperiod=t)
            # Minus Directional Indicator / Movement
            informative[f"%-{coin}minus_di-period_{t}"] = ta.MINUS_DI(informative, timeperiod=t)
            informative[f"%-{coin}minus_dm-period_{t}"] = ta.MINUS_DM(informative, timeperiod=t)
            # Plus Directional Indicator / Movement
            informative[f"%-{coin}plus_di-period_{t}"] = ta.PLUS_DI(informative, timeperiod=t)
            informative[f"%-{coin}plus_dm-period_{t}"] = ta.PLUS_DM(informative, timeperiod=t)
            # MFI
            informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            # ADX
            informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, window=t)
            # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
            # informative[f"%-{coin}trix-period_{t}"] = ta.TRIX(informative, timeperiod=t)
            # WILLR - Williams' %R
            informative[f"%-{coin}willr-period_{t}"] = ta.WILLR(informative, timeperiod=t)
            # Kaufman's Adaptive Moving Average (KAMA)
            informative[f"%-{coin}kama-period_{t}"] = ta.KAMA(informative, window=t)
            # SMA
            informative[f"%-{coin}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            # EMA
            informative[f"%-{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)
            # TEMA
            # informative[f"%-{coin}tema-period_{t}"] = ta.TEMA(informative, timeperiod=t)
            # ATR - Average True Range
            informative[f"%-{coin}atr-period_{t}"] = ta.ATR(informative, timeperiod=t)
            # Stoch
            stoch = ta.STOCH(informative)
            informative[f"%-{coin}slowd"] = stoch["slowd"]
            informative[f"%-{coin}slowk"] = stoch["slowk"]
            # Stoch Fast
            stochf = ta.STOCHF(informative)
            informative[f"%-{coin}fastd"] = stochf["fastd"]
            informative[f"%-{coin}fastk"] = stochf["fastk"]
            # Stoch RSI
            stoch_rsi = ta.STOCHRSI(informative)
            informative[f"%-{coin}fastd"] = stoch_rsi["fastd"]
            informative[f"%-{coin}fastk"] = stoch_rsi["fastk"]
            # Hilbert
            hilbert = ta.HT_SINE(informative)
            informative[f"%-{coin}htsine"] = hilbert["sine"]
            informative[f"%-{coin}htleadsine"] = hilbert["leadsine"]
            # Bollinger bands
            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{coin}bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{coin}bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{coin}bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{coin}bb_width-period_{t}"] = (
                informative[f"{coin}bb_upperband-period_{t}"]
                - informative[f"{coin}bb_lowerband-period_{t}"]
            ) / informative[f"{coin}bb_middleband-period_{t}"]
            informative[f"%-{coin}close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{coin}bb_lowerband-period_{t}"]
            )

            informative[f"%-{coin}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

            informative[f"%-{coin}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

        informative[f"%-{coin}pct-change"] = informative["close"].pct_change()
        informative[f"%-{coin}raw_volume"] = informative["volume"]
        informative[f"%-{coin}raw_price"] = informative["close"]

        # FIXME: add these outside the user strategy?
        # The following columns are necessary for RL models.
        informative[f"%-{coin}raw_close"] = informative["close"]
        informative[f"%-{coin}raw_open"] = informative["open"]
        informative[f"%-{coin}raw_high"] = informative["high"]
        informative[f"%-{coin}raw_low"] = informative["low"]

        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:
            df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

            # For RL, there are no direct targets to set. This is filler (neutral)
            # until the agent sends an action.
            df["&-action"] = 0

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [df["do_predict"] == 1, df["&-action"] == 1]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [df["do_predict"] == 1, df["&-action"] == 3]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["&-action"] == 2]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [df["do_predict"] == 1, df["&-action"] == 4]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
