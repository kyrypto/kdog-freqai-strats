import logging
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type, Union

import gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
import torch.multiprocessing
from pandas import DataFrame
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from freqtrade.exceptions import OperationalException
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.Base5ActionRLEnvSpot import Actions, Base5ActionRLEnvSpot
from freqtrade.freqai.RL.BaseEnvironment import Positions
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')

SB3_MODELS = ['PPO', 'A2C', 'DQN']
SB3_CONTRIB_MODELS = ['TRPO', 'ARS', 'RecurrentPPO', 'MaskablePPO']


class SpotReinforcementLearningModel(IFreqaiModel):
    """
    User created Reinforcement Learning Model prediction class
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs['config'])
        self.max_threads = min(self.freqai_info['rl_config'].get(
            'cpu_count', 1), max(int(self.max_system_threads / 2), 1))
        th.set_num_threads(self.max_threads)
        self.reward_params = self.freqai_info['rl_config']['model_reward_parameters']
        self.train_env: Union[SubprocVecEnv, gym.Env] = None
        self.eval_env: Union[SubprocVecEnv, gym.Env] = None
        self.eval_callback: EvalCallback = None
        self.model_type = self.freqai_info['rl_config']['model_type']
        self.rl_config = self.freqai_info['rl_config']
        self.continual_learning = self.freqai_info.get('continual_learning', False)
        if self.model_type in SB3_MODELS:
            import_str = 'stable_baselines3'
        elif self.model_type in SB3_CONTRIB_MODELS:
            import_str = 'sb3_contrib'
        else:
            raise OperationalException(f'{self.model_type} not available in stable_baselines3 or '
                                       f'sb3_contrib. please choose one of {SB3_MODELS} or '
                                       f'{SB3_CONTRIB_MODELS}')

        mod = __import__(import_str, fromlist=[
                         self.model_type])
        self.MODELCLASS = getattr(mod, self.model_type)
        self.policy_type = self.freqai_info['rl_config']['policy_type']
        self.unset_outlier_removal()

    def unset_outlier_removal(self):
        """
        If user has activated any function that may remove training points, this
        function will set them to false and warn them
        """
        if self.ft_params.get('use_SVM_to_remove_outliers', False):
            self.ft_params.update({'use_SVM_to_remove_outliers': False})
            logger.warning('User tried to use SVM with RL. Deactivating SVM.')
        if self.ft_params.get('use_DBSCAN_to_remove_outliers', False):
            self.ft_params.update({'use_SVM_to_remove_outliers': False})
            logger.warning('User tried to use DBSCAN with RL. Deactivating DBSCAN.')
        if self.freqai_info['data_split_parameters'].get('shuffle', False):
            self.freqai_info['data_split_parameters'].update('shuffle', False)
            logger.warning('User tried to shuffle training data. Setting shuffle to False')

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :returns:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info("--------------------Starting training " f"{pair} --------------------")

        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        data_dictionary: Dict[str, Any] = dk.make_train_test_datasets(
            features_filtered, labels_filtered)
        dk.fit_labels()  # FIXME useless for now, but just satiating append methods

        # normalize all data based on train_dataset only
        prices_train, prices_test = self.build_ohlc_price_dataframes(dk.data_dictionary, pair, dk)
        data_dictionary = dk.normalize_data(data_dictionary)

        # data cleaning/analysis
        self.data_cleaning_train(dk)

        logger.info(
            f'Training model on {len(dk.data_dictionary["train_features"].columns)}'
            f' features and {len(data_dictionary["train_features"])} data points'
        )

        self.set_train_and_eval_environments(data_dictionary, prices_train, prices_test, dk)

        model = self.fit(data_dictionary, dk)

        logger.info(f"--------------------done training {pair}--------------------")

        return model

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, DataFrame],
                                        prices_train: DataFrame, prices_test: DataFrame,
                                        dk: FreqaiDataKitchen):
        """
        User can override this if they are using a custom MyRLEnv
        :params:
        data_dictionary: dict = common data dictionary containing train and test
            features/labels/weights.
        prices_train/test: DataFrame = dataframe comprised of the prices to be used in the
            environment during training
        or testing
        dk: FreqaiDataKitchen = the datakitchen for the current pair
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        self.train_env = self.MyRLEnv(df=train_df, prices=prices_train, window_size=self.CONV_WIDTH,
                                      reward_kwargs=self.reward_params, config=self.config)
        self.eval_env = Monitor(self.MyRLEnv(df=test_df, prices=prices_test,
                                window_size=self.CONV_WIDTH,
                                reward_kwargs=self.reward_params, config=self.config))
        self.eval_callback = EvalCallback(self.eval_env, deterministic=True,
                                          render=False, eval_freq=len(train_df),
                                          best_model_save_path=str(dk.data_path))

    @abstractmethod
    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        Agent customizations and abstract Reinforcement Learning customizations
        go in here. Abstract method, so this function must be overridden by
        user class.
        """
        return

    def get_state_info(self, pair: str) -> Tuple[float, float, int]:
        """
        State info during dry/live/backtesting which is fed back
        into the model.
        :param:
        pair: str = COIN/STAKE to get the environment information for
        :returns:
        market_side: float = representing short, long, or neutral for
            pair
        trade_duration: int = the number of candles that the trade has
            been open for
        """
        open_trades = Trade.get_trades_proxy(is_open=True)
        market_side = 0.5
        current_profit: float = 0
        trade_duration = 0
        for trade in open_trades:
            if trade.pair == pair:
                if self.strategy.dp._exchange is None:  # type: ignore
                    logger.error('No exchange available.')
                else:
                    current_value = self.strategy.dp._exchange.get_rate(  # type: ignore
                                pair, refresh=False, side="exit", is_short=trade.is_short)
                openrate = trade.open_rate
                now = datetime.now(timezone.utc).timestamp()
                trade_duration = int((now - trade.open_date.timestamp()) / self.base_tf_seconds)
                if 'long' in str(trade.enter_tag):
                    market_side = 1
                    current_profit = (current_value - openrate) / openrate
                else:
                    market_side = 0
                    current_profit = (openrate - current_value) / openrate

        return market_side, current_profit, int(trade_duration)

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param: unfiltered_dataframe: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """

        dk.find_features(unfiltered_df)
        filtered_dataframe, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        filtered_dataframe = dk.normalize_data_from_metadata(filtered_dataframe)
        dk.data_dictionary["prediction_features"] = filtered_dataframe

        # optional additional data cleaning/analysis
        self.data_cleaning_predict(dk)

        pred_df = self.rl_model_predict(
            dk.data_dictionary["prediction_features"], dk, self.model)
        pred_df.fillna(0, inplace=True)

        return (pred_df, dk.do_predict)

    def rl_model_predict(self, dataframe: DataFrame,
                         dk: FreqaiDataKitchen, model: Any) -> DataFrame:
        """
        A helper function to make predictions in the Reinforcement learning module.
        :params:
        dataframe: DataFrame = the dataframe of features to make the predictions on
        dk: FreqaiDatakitchen = data kitchen for the current pair
        model: Any = the trained model used to inference the features.
        """
        output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)

        def _predict(window):
            market_side, current_profit, trade_duration = self.get_state_info(dk.pair)
            observations = dataframe.iloc[window.index]
            observations['current_profit_pct'] = current_profit
            observations['position'] = market_side
            observations['trade_duration'] = trade_duration
            res, _ = model.predict(observations, deterministic=True)
            return res

        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

        return output

    def build_ohlc_price_dataframes(self, data_dictionary: dict,
                                    pair: str, dk: FreqaiDataKitchen) -> Tuple[DataFrame,
                                                                               DataFrame]:
        """
        Builds the train prices and test prices for the environment.
        """

        coin = pair.split('/')[0]
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        # price data for model training and evaluation
        tf = self.config['timeframe']
        ohlc_list = [f'%-{coin}raw_open_{tf}', f'%-{coin}raw_low_{tf}',
                     f'%-{coin}raw_high_{tf}', f'%-{coin}raw_close_{tf}']
        rename_dict = {f'%-{coin}raw_open_{tf}': 'open', f'%-{coin}raw_low_{tf}': 'low',
                       f'%-{coin}raw_high_{tf}': ' high', f'%-{coin}raw_close_{tf}': 'close'}

        prices_train = train_df.filter(ohlc_list, axis=1)
        prices_train.rename(columns=rename_dict, inplace=True)
        prices_train.reset_index(drop=True)

        prices_test = test_df.filter(ohlc_list, axis=1)
        prices_test.rename(columns=rename_dict, inplace=True)
        prices_test.reset_index(drop=True)

        return prices_train, prices_test

    def load_model_from_disk(self, dk: FreqaiDataKitchen) -> Any:
        """
        Can be used by user if they are trying to limit_ram_usage *and*
        perform continual learning.
        For now, this is unused.
        """
        exists = Path(dk.data_path / f"{dk.model_filename}_model").is_file()
        if exists:
            model = self.MODELCLASS.load(dk.data_path / f"{dk.model_filename}_model")
        else:
            logger.info('No model file on disk to continue learning from.')

        return model

    # Nested class which can be overridden by user to customize further
    class MyRLEnv(Base5ActionRLEnvSpot):
        """
        User can override any function in BaseRLEnv and gym.Env. Here the user
        sets a custom reward based on profit and trade duration.
        """

        def calculate_reward(self, action: int) -> float:
            """
            An example reward function. This is the one function that users will likely
            wish to inject their own creativity into.
            :params:
            action: int = The action made by the agent for the current candle.
            :returns:
            float = the reward to give to the agent for current step (used for optimization
                of weights in NN)
            """
            # first, penalize if the action is not valid
            if not self._is_valid(action):
                return -5

            pnl = self.get_unrealized_profit()
            rew = np.sign(pnl) * (pnl + 1)
            factor = 100.

            # reward agent for entering trades
            if (action in (Actions.Long_enter.value, Actions.Long_enter_1.value, Actions.Long_enter_2.value)
                    and self._position == Positions.Neutral):
                return 25
            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            if self._last_trade_tick:
                trade_duration = self._current_tick - self._last_trade_tick
            else:
                trade_duration = 0

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # discourage sitting in position
            if (self._position == (Positions.Long) and
               action == Actions.Neutral.value):
                return -1 * trade_duration / max_trade_duration

            # close long
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(rew * factor)

            return 0.


def make_env(MyRLEnv: Type[gym.Env], env_id: str, rank: int,
             seed: int, train_df: DataFrame, price: DataFrame,
             reward_params: Dict[str, int], window_size: int, monitor: bool = False,
             config: Dict[str, Any] = {}) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:

        env = MyRLEnv(df=train_df, prices=price, window_size=window_size,
                      reward_kwargs=reward_params, id=env_id, seed=seed + rank, config=config)
        if monitor:
            env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init
