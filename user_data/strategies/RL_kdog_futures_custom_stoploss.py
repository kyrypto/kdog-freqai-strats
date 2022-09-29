import logging
from functools import reduce
from datetime import datetime
import numpy as np  # noqa
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.persistence import Trade

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)


class RL_kdog_futures_custom_stoploss(IStrategy):
    """
Here be stonks
1. freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy RL_kdog_futures_custom_stoploss --freqaimodel ReinforcementLearner --spaces sell roi --timerange "$(date --date='-1 week' '+%Y%m%d')"-"$(date '+%Y%m%d')" -e 1000
2. freqtrade trade --logfile ./logs --freqaimodel ReinforcementLearner_multiproc --strategy RL_kdog_futures_custom_stoploss
    """
    INTERFACE_VERSION: int = 3
    risk_reward_optimize = True
    risk_reward_ratio: DecimalParameter(2, 10, default=3.5, decimals=1, space='sell', optimize=risk_reward_optimize)
    break_even_optimize = True
    set_to_break_even_at_profit: DecimalParameter(1, 10, default=1, decimals=1, space='sell', optimize=break_even_optimize)
    custom_info = {
        'risk_reward_ratio': risk_reward_ratio.value,
        'set_to_break_even_at_profit': set_to_break_even_at_profit.value,
    }
    use_custom_stoploss = True
    stoploss = -0.9
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

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True
    # Specific variables
    leverage_optimize = True
    leverage_num = DecimalParameter(1, 20, default=5, decimals=2, space='sell', optimize=leverage_optimize)
    linear_roi_offset = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=True, load=True
    )
    max_roi_time_long = IntParameter(0, 800, default=400, space="sell", optimize=True, load=True)
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        """
            custom_stoploss using a risk/reward ratio
        """
        result = break_even_sl = takeprofit_sl = -1
        custom_info_pair = self.custom_info.get(pair)
        if custom_info_pair is not None:
            # using current_time/open_date directly via custom_info_pair[trade.open_daten]
            # would only work in backtesting/hyperopt.
            # in live/dry-run, we have to search for nearest row before it
            open_date_mask = custom_info_pair.index.unique().get_loc(trade.open_date_utc, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            # trade might be open too long for us to find opening candle
            if(len(open_df) != 1):
                return -1 # won't update current stoploss

            initial_sl_abs = open_df['stoploss_rate']

            # calculate initial stoploss at open_date
            initial_sl = initial_sl_abs/current_rate-1

            # calculate take profit treshold
            # by using the initial risk and multiplying it
            risk_distance = trade.open_rate-initial_sl_abs
            reward_distance = risk_distance*self.risk_reward_ratio.value
            # take_profit tries to lock in profit once price gets over
            # risk/reward ratio treshold
            take_profit_price_abs = trade.open_rate+reward_distance
            # take_profit gets triggerd at this profit
            take_profit_pct = take_profit_price_abs/trade.open_rate-1

            # break_even tries to set sl at open_rate+fees (0 loss)
            # break_even gets triggerd at this profit
            break_even_profit_distance = risk_distance*self.set_to_break_even_at_profit.value
            break_even_profit_pct = (break_even_profit_distance+current_rate)/current_rate-1

            result = initial_sl
            if(current_profit >= break_even_profit_pct):
                break_even_sl = (trade.open_rate*(1+trade.fee_open+trade.fee_close) / current_rate)-1
                result = break_even_sl

            if(current_profit >= take_profit_pct):
                takeprofit_sl = take_profit_price_abs/current_rate-1
                result = takeprofit_sl

        return result
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
            # DEMA - Double Exponential Moving Average
            informative[f"%-{coin}dema-period_{t}"] = ta.DEMA(informative, timeperiod=t)
            # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
            informative[f"%-{coin}ht_trendline"] = ta.HT_TRENDLINE(informative)
            # Linear Regression
            informative[f"%-{coin}linearreg-period_{t}"] = ta.LINEARREG(informative, timeperiod=t)
            # CORREL - Pearson's Correlation Coefficient (r)
            informative[f"%-{coin}correl-period_{t}"] = ta.CORREL(informative, timeperiod=t)
            # STDDEV - Standard Deviation
            informative[f"%-{coin}stddev-period_{t}"] = ta.STDDEV(informative, timeperiod=t)
            # TSF - Time Series Forecast
            informative[f"%-{coin}tsf-period_{t}"] = ta.TSF(informative, timeperiod=t)
            # VAR - Variance
            informative[f"%-{coin}var-period_{t}"] = ta.VAR(informative, timeperiod=t)
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
            # Kaufman's Adaptive Moving Average (KAMA)
            informative[f"%-{coin}kama-period_{t}"] = ta.KAMA(informative, window=t)
            # SMA
            informative[f"%-{coin}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            # EMA
            informative[f"%-{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)
            # TEMA
            informative[f"%-{coin}tema-period_{t}"] = ta.TEMA(informative, timeperiod=t)
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
        dataframe['atr'] = ta.ATR(dataframe)
        dataframe['stoploss_rate'] = dataframe['close']-(dataframe['atr']*2)
        self.custom_info[metadata['pair']] = dataframe[['date', 'stoploss_rate']].copy().set_index('date')

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
