import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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


def test_model(env, model):
    observation, info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
    env.render()

def get_path(mode, episode, window_size, use_batch_norm = False, obs_normalizer = False):

    assert mode in ['weights', 'results']
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    if obs_normalizer:
        normailzed_str = "normalized"
    else:
        normailzed_str = "no_normalized"


    return '{}/ddpg/window_{}_{}_{}_eps_{}_checkpoint.ckpt'.format(mode,
                                                            window_size,
                                                            batch_norm_str,
                                                            normailzed_str,
                                                            episode)

def get_variable_scope(window_size, use_batch_norm = False, obs_normalizer = False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    if obs_normalizer:
        normailzed_str = "normalized"
    else:
        normailzed_str = "no_normalized"
    return 'window_{}_{}'.format(window_size, batch_norm_str, normailzed_str)

def obs_normalizer(observation):
    """ Preprocess observation obtained by environment
    Args:
        observation: (nb_classes, window_length, num_features) or with info
    Returns: normalized
    """
    means = np.expand_dims(np.mean(observation, axis=1), axis=1)
    stds = np.expand_dims(np.std(observation, axis=1), axis=1)

    normed_obs = (observation - means) / (stds + 1e-8)

    return normed_obs

if __name__ == '__main__':

    eps=1e-8
    use_batch_norm = False


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


    with open('configs/ddpg_default.json') as f:
        config = json.load(f)

    train_step = timestamp.index('2019-07-01')
    valid_step = timestamp.index('2020-07-01')

    history_stock_price_training = history_stock_price[:,:train_step,:]
    history_stock_price_validating = history_stock_price[:,train_step:valid_step,:]
    history_stock_price_testing = history_stock_price[:,valid_step:,:]
    timestamp_training = timestamp[:train_step]
    timestamp_validating = timestamp[train_step:valid_step]
    timestamp_testing = timestamp[valid_step:]

    env_training = PortfolioEnv(history=history_stock_price_training,
                                abbreviation=abbreviations,
                                timestamp=timestamp_training,
                                steps=3000)


    env_validating = PortfolioEnv(history=history_stock_price_validating,
                                abbreviation=abbreviations,
                                timestamp=timestamp_validating,
                                steps=120,
                                sample_start_date='2020-01-02')

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(config['input']['asset_number']+1))

    model_save_path = get_path(mode = 'weights',
                            episode = config['training']['episode'],
                               window_size = config['input']['window_size'],
                               obs_normalizer = True)

    summary_path = get_path(mode = 'results',
                            episode = config['training']['episode'],
                            window_size = config['input']['window_size'],
                            obs_normalizer = True)

    variable_scope = get_variable_scope(config['input']['window_size'],
                                        obs_normalizer = True)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.variable_scope(variable_scope):

        sess = tf.Session(config=tf_config)
        if config["training"]["device"] == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 1
        stockactor = StockActor(sess, config, action_bound=1)
        stockcritic = StockCritic(sess, config, num_actor_vars=stockactor.get_num_trainable_vars())

        myddpg = DDPG(env_training,sess,actor = stockactor,
                                       critic=stockcritic,
                                       obs_normalizer = obs_normalizer,
                                       actor_noise=actor_noise,
                                       model_save_path=model_save_path,
                                       summary_path=summary_path,
                                       config=config)

        myddpg.initialize(load_weights=False)
        myddpg.train()
