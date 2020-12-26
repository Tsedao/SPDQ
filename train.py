import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import json


from environment.portfolio import PortfolioEnv

from model.ddpg.stockactor import StockActor
from model.ddpg.stockcritic import StockCritic
from model.ddpg.ddpg import DDPG
from model.core.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise


if __name__ == '__main__':

    eps=1e-8

    abbreviations = ['AAPL.O',
                     'ADBE.O',
                     'AMZN.O',
                     'DIS.N',
                     'GOOGL.O',
                     'JNJ.N',
                     'JPM.N',
                     'MSFT.O',
                     'NFLX.O',
                     'PG.N']

    # read data
    with h5py.File('./Data/history_stock_price.h5','r') as f:
        history_stock_price = f['stock_price'][...]
        timestamp = [s.decode('utf-8') for s in f['timestamp']]
        abbreviations = [s.decode('utf-8') for s in f['abbreviations']]
        features = [s.decode('utf-8') for s in f['features']]

    env = PortfolioEnv(history=history_stock_price,abbreviation=abbreviations,timestamp=timestamp,steps=3000)

    with open('configs/ddpg_default.json') as f:
        config = json.load(f)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(config['input']['asset_number']+1))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    if config["training"]["device"] == "cpu":
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0
    else:
        tf_config.gpu_options.per_process_gpu_memory_fraction = 1
    stockactor = StockActor(sess, config, action_bound=1)
    stockcritic = StockCritic(sess, config, num_actor_vars=stockactor.get_num_trainable_vars())

    myddpg = DDPG(env,sess,stockactor,critic=stockcritic,
                  actor_noise=actor_noise, config=config)

    myddpg.initialize(load_weights=False)
    myddpg.train()

    print(env.infos[-1])
