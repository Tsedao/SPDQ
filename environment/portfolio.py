import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
import gym.spaces
import copy

from matplotlib import pyplot as plt

from utils.data import date_to_index, index_to_date

from icecream import ic
eps=1e-13

def sharpe(returns, freq=30, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)

def ARR(portfolio_value,freq):
    return ((portfolio_value.iloc[-1] - portfolio_value.iloc[0]) / portfolio_value.iloc[0])*(256/freq)

def AVOL(returns,freq):
    return  np.sqrt(np.mean(returns**2)-(np.mean(returns))**2) * np.sqrt(256/freq)

def DDR(returns,arr):
    return arr / np.sqrt(np.mean(np.min(returns,0)**2))


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history,
                       abbreviation,
                       timestamp,
                       steps=730,
                       step_size=1,
                       window_length=50,
                       start_idx=0,
                       start_date=None,
                       feature_num=4,
                       valid_env=False):
        """
        Args:
            history: (num_stocks, timestamp, 5) open, high, low, close, volume
            abbreviation: a list of length num_stocks with assets name
            timestamp: a list of timestamp in format %Y-%m-%d
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        assert history.shape[0] == len(abbreviation), 'Number of stock is not consistent'
        import copy
        self.step_size = step_size
        self.steps = steps
        self.window_length = window_length
        self.start_idx = start_idx
        self.start_date = start_date

        # make immutable class
        self._data = history.copy()[...,:feature_num]  # all data
        self.asset_names = copy.copy(abbreviation)
        self.timestamp_all = timestamp
        self.valid_env = valid_env

    def _step(self):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes. Normalize could be critical here
        self.step += self.step_size
        ic(self.step)
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()
        # normalize obs with open price

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0
        # get data for this episode, each episode might be different.
        if self.start_date is None:
            if self.valid_env:
                self.idx = self.window_length
            else:
                self.idx = np.random.randint(
                    low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date,self.timestamp_all) - self.start_idx
            assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        self._start_date = index_to_date(self.idx, self.timestamp_all)
        self._end_date = index_to_date(self.idx+self.steps, self.timestamp_all)
        print('Start date: %s, End date: %s' %(self._start_date,self._end_date))
        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :]
        # apply augmentation?
        self.data = data
        # self.timestamp = self.timestamp[self.idx - self.window_length:self.idx + self.steps + 1]
        return self.data[:, self.step:self.step + self.window_length, :].copy(), \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=list(), steps=730, trading_cost=0.0025, time_cost=0.0, beta=0.0):
        self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps

        self.beta = beta
        self.onestep_rewards = [0]

    def _step(self, w1_o, v1, v2):
        """
        Step.
        w1_o - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        v1 - price relative vector 1: also called return close/open
            e.g. [1.0, 0.9, 1.1]
        v2 - price relative vector 2: close/ pre_close
            for the purpose of calculate market_value

        """
        assert w1_o.shape == v1.shape, 'w1 and y1 must have the same shape'
        assert v1[0] == 1.0, 'y1[0] must be 1'

        v0 = v2 / v1  # price relative vector 0: open / pre_close

        p0_c = self.p0_c
        w0_c = self.w0_c
        # passive change
        p1_o = p0_c * np.dot(v0,w0_c)     # change of portfolio_value because of open price not equals to pre close price

        w1_c = (v1 * w1_o) / (np.dot(v1, w1_o) + eps)  # arggressive weights evolve into

        dw = np.abs(w1_o[1:] - w0_c[1:]).sum()
        turn_over_ratio = dw
        mu1 = self.cost * dw  # (eq16) cost to change portfolio

        assert mu1 < 1.0, 'Cost is larger than current holding'

        var = np.var(self.onestep_rewards)

        r = np.log((1-mu1)*np.dot(w1_o,v1)) - self.beta * (var) #

        self.onestep_rewards.append(r)

        p1_c = p1_o * (1 - mu1) * np.dot(v1, w1_o)  #  final portfolio value

        p1_c = p1_c * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1_c / p0_c - 1  # rate of returns
        r1 = np.log((p1_c + eps) / (p0_c + eps))  # log rate of return

        # remember for next step
        self.w0_c = w1_c
        self.p0_c = p1_c

        # if we run out of money, we're done (losing all the money)
        done = p1_c == 0

        info = {
            "onestep reward penalized by var": r,
            "log_return": r1,
            "portfolio_value": p1_c,
            "return": v2.mean(),
            "rate_of_return": rho1,
            "weights": w1_o,
            "weights_mean": w1_o.mean(),
            "weights_std": w1_o.std(),
            "cost": mu1,
            "turn_over_ratio":turn_over_ratio
        }
        self.infos.append(info)
        # print('reward: %.4f,turn_over_ratio: %.4f'%(r,turn_over_ratio))
        return r, info, done

    def reset(self):
        self.infos = []
        self.p0_c = 1.0
        self.w0_c = np.array([1.0]+[0 for i in range(len(self.asset_names))])

class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 history,
                 abbreviation,
                 timestamp,
                 steps=730,  # 2 years
                 step_size=1,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0,
                 sample_start_date=None,
                 feature_num=4,
                 valid_env = False,
                 beta = 0.0,
                 name = 'StockTradingEnv_v1'
                 ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            sample_start_date - The start date sampling from the history
        """
        self.window_length = window_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx
        self.timestamp_all = timestamp

        self.src = DataGenerator(history, abbreviation, timestamp,steps=steps,
                                step_size=step_size,window_length=window_length,
                                start_idx=start_idx,start_date=sample_start_date,
                                feature_num=feature_num,valid_env = valid_env)

        self.sim = PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps,
            beta = beta)
        self.name = name
        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.src.asset_names) + 1,), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(abbreviation) + 1, window_length,
                                                                                 history.shape[-1]), dtype=np.float32)

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names) + 1,)
        )
        # ic(action)

        # Env_v1 is based on zhengyao jiang
        if self.name == "StockTradingEnv_v1":
            action = np.clip(action, 0, 1)
            assert ((action>= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
            ## normalise just in case
            weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
            # weights = np.exp(weights) / np.sum(np.exp(weights))
            weights /= (np.sum(weights) + eps)
            weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        # Env_v2 is based on Alphastock
        elif self.name == "StockTradingEnv_v2":
            action = np.clip(action, 0, 1)
            assert ((action>= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
            # buy good sell bad startegy, allow to make short position
            action = np.exp(action) / np.sum(np.exp(action))                     # normalise the action because of exploration
            weights = action
            sell_idx, buy_idx = np.argsort(weights)[:len(weights)//2], np.argsort(weights)[len(weights)//2:]
            sell_prop = np.exp(1-weights[sell_idx]) / np.sum(np.exp(1-weights[sell_idx]))
            buy_prop = np.exp(weights[buy_idx]) / np.sum(np.exp(weights[buy_idx]))
            weights[sell_idx] -= sell_prop
            weights[buy_idx] += buy_prop

        elif self.name in ["StockTradingEnv_v3","StockTradingEnv_v4"]:           # v4 require change the activation of w


            if self.name == "StockTradingEnv_v3":
                # need softmax
                action = np.clip(action, 0, 1)
                assert ((action>= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
                action = np.exp(action) / np.sum(np.exp(action))                  # normalise the action because of exploration using probability
                sell_idx, buy_idx = np.argsort(action)[:len(action)//2], np.argsort(action)[len(action)//2:][::-1]
                sell_prop = np.exp(50*(1-action[sell_idx])) / np.sum(np.exp(50*(1-action[sell_idx])))
                buy_prop = np.exp(100*(action[buy_idx])) / np.sum(np.exp(100*(action[buy_idx])))
            else:
                # don't need softmax
                sell_idx, buy_idx = np.where(action<0)[0], np.where(action>=0)[0]
                buy_prop = (action[buy_idx]) / np.sum((action[buy_idx]))
                sell_prop = (-action[sell_idx]) / np.sum((-action[sell_idx]))

            selled_shares_total = 0
            weights = self.sim.w0_c

            if len(sell_idx) == 0 or len(buy_idx) == 0:
                # print('******Holding*******')
                pass
            else:
                # print('*******Selling******')
                for i,prop in zip(sell_idx,sell_prop):
                    selled_shares = prop*weights[i]
                    weights[i] -= selled_shares                                      # reduce the shares based on holded position
                    selled_shares_total += selled_shares
                    # print(prop,i)

                # print('******Buying********')
                bought_shares_total = 0
                for i,prop in zip(buy_idx,buy_prop):                                 # increase the share based on the number of reduced share
                    bought_shares = min(prop*selled_shares_total,1-weights[i])
                    weights[i] += bought_shares
                    bought_shares_total = bought_shares_total + bought_shares
                    # print(prop,i)
        else:
            raise('%s Not Implemented'%self.name)

        if self.src.valid_env:
            ic(weights)
        weights = np.float32(weights)
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)
        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        # ic(observation.shape)
        # ic(cash_observation.shape)
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/pre_close)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]
        pre_close_price_vector = observation[:, -2, 3]
        y1 = close_price_vector / open_price_vector
        y2 = close_price_vector / pre_close_price_vector

        pre_w = self.sim.w0_c
        self.weights[:,:-1] = self.weights[:,1:]
        self.weights[:,-1] = pre_w                                              # add previous weights to the observation space
        self.weights = np.float32(self.weights)
        onestep_r, info, done2 = self.sim._step(weights, y1, y2)
        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step,self.timestamp_all)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)
        #observation = (observation,pre_w)

        observation = np.concatenate([observation,np.expand_dims(self.weights,axis=-1)],axis=-1)
        return observation, onestep_r, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.weights = np.concatenate([np.ones((1,self.window_length)),
                    np.zeros((self.num_stocks, self.window_length))],axis=0)
        self.sim.reset()
        pre_w = self.sim.w0_c
        observation, ground_truth_obs = self.src.reset()
        self.timestamp_all = self.src.timestamp_all
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        #observation = (observation, pre_w)
        observation = np.concatenate([observation,np.expand_dims(self.weights,axis=-1)],axis=-1)
        return observation, info

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()

    def render(self, mode='human', close=False):
        return self._render(mode='human', close=False)

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        # df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)


class MultiActionPortfolioEnv(PortfolioEnv):
    def __init__(self,
                 history,
                 abbreviation,
                 timestamp,
                 model_names,
                 window_length = 100,
                 steps=730,  # 2 years
                 step_size=1,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 start_idx=0,
                 sample_start_date=None,
                 feature_num = 4
                 ):
        super(MultiActionPortfolioEnv, self).__init__(history, abbreviation, timestamp,
                              steps,step_size,trading_cost, time_cost, window_length,
                              start_idx, sample_start_date,feature_num)
        self.model_names = model_names
        # need to create each simulator for each model
        self.sim = [PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps) for _ in range(len(self.model_names))]


    def _step(self, action):
        """ Step the environment by a vector of actions
        Args:
            action: (num_models, num_stocks + 1)
        Returns:
        """
        assert action.ndim == 2, 'Action must be a two dimensional array with shape (num_models, num_stocks + 1)'
        assert action.shape[1] == len(self.sim[0].asset_names) + 1
        assert action.shape[0] == len(self.model_names)
        # normalise just in case
        action = np.clip(action, 0, 1)
        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (np.sum(weights, axis=1, keepdims=True) + eps)
        # so if weights are all zeros we normalise to [1,0...]
        weights[:, 0] += np.clip(1 - np.sum(weights, axis=1), 0, 1)
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(weights, axis=1), np.ones(shape=(weights.shape[0])), 3,
                                       err_msg='weights should sum to 1. action="%s"' % weights)

        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]
        pre_close_price_vector = observation[:, -2, 3]
        y1 = close_price_vector / open_price_vector
        y2 = close_price_vector / pre_close_price_vector

        rewards = np.empty(shape=(weights.shape[0]))
        info = {}
        dones = np.empty(shape=(weights.shape[0]), dtype=bool)
        for i in range(weights.shape[0]):
            reward, current_info, done2 = self.sim[i]._step(weights[i], y1, y2)
            rewards[i] = reward
            info[self.model_names[i]] = current_info['portfolio_value']
            info[self.model_names[i]+'_rate_of_return'] = current_info['rate_of_return']
            info['return'] = current_info['return']
            dones[i] = done2

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step, self.timestamp_all)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return observation, rewards, np.all(dones) or done1, info

    def _reset(self):
        self.infos = []
        for sim in self.sim:
            sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

    def plot(self):
        df_info = pd.DataFrame(self.infos)
        fig=plt.gcf()
        title = 'Trading Performance of Various Models'
        df_info.set_index('date', inplace=True)
        df_info[self.model_names + ['market_value']].plot(title=title, fig=fig, rot=30)
