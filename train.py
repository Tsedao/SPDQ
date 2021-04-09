import numba
import numpy as np

def test_model(env, model, policy_delay):
    observation, info = env.reset()
    done = False
    count = 0
    pre_action = None
    while not done:
        action = pre_action
        if count % policy_delay == 0:
            action = model.predict_single(observation)
            pre_action = action
        count += 1
        observation, _, done, _ = env.step(action)

    return env

def test_model_multiple(env, models, policy_delay):
    import numpy as np

    observation, info = env.reset()
    done = False
    count = 0
    pre_actions = []
    while not done:
        actions = pre_actions
        if count % policy_delay == 0:
            actions = []
            for model in models:
                actions.append(model.predict_single(observation))
            actions = np.array(actions)
            pre_actions = actions
        count += 1
        observation, _, done, info = env.step(actions)

    return env

def get_path(mode, model, episode, window_size, use_batch_norm = False, use_obs_normalizer = False):

    assert mode in ['weights', 'results']
    assert model in ['ddpg', 'td3','sac']

    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    if use_obs_normalizer:
        normailzed_str = "normalized"
    else:
        normailzed_str = "no_normalized"


    return '{}/{}/window_{}_{}_{}_eps_{}_checkpoint.ckpt'.format(mode,
                                                            model,
                                                            window_size,
                                                            batch_norm_str,
                                                            normailzed_str,
                                                            episode)

def get_variable_scope(window_size, use_batch_norm = False, use_obs_normalizer = False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'

    if use_obs_normalizer:
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

    import numpy as np
    # means = np.expand_dims(np.mean(observation, axis=1), axis=1)
    # stds = np.expand_dims(np.std(observation, axis=1), axis=1)
    #
    # normed_obs = (observation - means) / (stds + 1e-8)
    normed_obs = observation[...,-4:]
    return normed_obs

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@numba.njit
def shift5_numba(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:,:num,:] = np.full(shape=(arr.shape[0],num,arr.shape[2]),fill_value=fill_value)
        result[:,num:,:] = arr[:,:-num,:]
    return result

def normalize_obs_logdiff(obs):
    """
    Inputs:
        obs:[asset_num,window_length,feature_num] obs must bigger than zero
    Return:
        normalized_obs:[asset_num,window_length-1,feature_num]
    """
    forwoard_shift_obs = shift5_numba(obs,1)
    logdiff = np.log(obs / forwoard_shift_obs)[:,1:,:]
    return np.where(logdiff==-np.inf,0,logdiff)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')

    parser.add_argument('--debug', '-d', help='print debug statement', default=False)
    parser.add_argument('--window_length', '-w', help='observation window length',default=50, type=int,required = True)
    parser.add_argument('--episode','-e', help='number of episodes during training', type=int, default=20, required = True)
    parser.add_argument('--steps','-s',help='number of steps in one episode',type=int, required=True)
    parser.add_argument('--device', help='use gpu to train or cpu', type=str, default='gpu', required=True)
    parser.add_argument('--batchnorm',type=str2bool,default=False)
    parser.add_argument('--gpu', '-g', help='which gpu to use', type=int, default=[6], nargs='+')
    parser.add_argument('--model','-m',help='which model to train',type=int,required=True)
    parser.add_argument('--highfreq',help='whether use highfreq data or not',type=str2bool, default=False)

    args = parser.parse_args()



    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.gpu])
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    import tensorflow as tf
    import pandas as pd
    import h5py
    import json

    from environment.portfolio import PortfolioEnv

    from model.ddpg.stockactor import DDPGActor
    from model.ddpg.stockcritic import DDPGCritic
    from model.sac.stockactor import SACActor
    from model.ddpg.ddpg import DDPG
    from model.td3.td3 import TD3
    from model.sac.sac import SAC
    from model.core.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise


    eps=1e-8
    use_batch_norm = args.batchnorm
    use_obs_normalizer = True
    model_zoo = ['ddpg', 'td3','sac']
    this_model = model_zoo[args.model]

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



    if use_batch_norm:
        config_path = 'configs/{}_batchnorm.json'.format(this_model)
    else:
        config_path = 'configs/{}_default.json'.format(this_model)

    # read data
    if args.highfreq:
        file_path = './HighData/history_stock_price_cn_1min.h5'
        config_file = 'configs/sac_default_high.json'

    else:
        file_path = './Data/history_stock_price.h5'



    with h5py.File(file_path,'r') as f:
        history_stock_price = f['stock_price'][...]
        timestamp = [s.decode('utf-8') for s in f['timestamp']]
        abbreviations = [s.decode('utf-8') for s in f['abbreviations']]
        features = [s.decode('utf-8') for s in f['features']]
    with open(config_path) as f:
        config = json.load(f)

    if args.highfreq:
        train_step = timestamp.index('2020-05-15 15:00:00')
        valid_step = timestamp.index('2020-12-17 15:00:00')
    else:
        train_step = timestamp.index('2019-07-01')
        valid_step = timestamp.index('2020-07-01')

    config['input']['window_size'] = args.window_length
    config['training']['episode'] = args.episode
    config['training']['device'] = args.device
    config['training']['max_step'] = args.steps

    feature_number = config['input']['feature_number']
    window_size = config['input']['window_size']
    asset_number = config['input']['asset_number']

    num_mixtrue = config['training'].get('num_mixtrue',None)
    actor_learning_rate = config['training']['actor_learning_rate']
    critic_learning_rate = config['training']['critic_learning_rate']
    batch_size = config['training']['batch_size']
    tau = config['training']['tau']        # frequency to update target net parameter
    episodes = config['training']['episode']
    device = config['training']['device']
    policy_delay = config['training']['policy_delay']
    max_step = config['training']['max_step']

    actor_layers = config['actor_layers']
    critic_layers = config['critic_layers']




    history_stock_price_training = history_stock_price[:,0:train_step,:]
    history_stock_price_validating = history_stock_price[:,train_step:valid_step,:]
    history_stock_price_testing = history_stock_price[:,valid_step:,:]
    timestamp_training = timestamp[0:train_step]
    timestamp_validating = timestamp[train_step:valid_step]
    timestamp_testing = timestamp[valid_step:]

    env_training = PortfolioEnv(history=history_stock_price_training,
                                abbreviation=abbreviations,
                                timestamp=timestamp_training,
                                window_length = window_size,
                                steps=max_step,
                                feature_num = feature_number)


    env_validating = PortfolioEnv(history=history_stock_price_validating,
                                abbreviation=abbreviations,
                                timestamp=timestamp_validating,
                                window_length = window_size,
                                steps=120,
                                sample_start_date='2020-01-02',
                                feature_num = feature_number)

#    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(asset_number+1),sigma=1, theta=0.15)
    actor_noise = None

    model_save_path = get_path(mode = 'weights',
                               model = this_model,
                               episode = episodes,
                               window_size = window_size,
                               use_batch_norm = use_batch_norm,
                               use_obs_normalizer = use_obs_normalizer)


    summary_path = get_path(mode = 'results',
                            model = this_model,
                            episode = episodes,
                            window_size = window_size,
                            use_batch_norm = use_batch_norm,
                            use_obs_normalizer = use_obs_normalizer)

    variable_scope = get_variable_scope(window_size,
                                        use_batch_norm = use_batch_norm,
                                        use_obs_normalizer = use_obs_normalizer)

    if use_obs_normalizer:
        obs_normalizer = normalize_obs_logdiff
        window_size = window_size - 1
    else:
        obs_normalizer = None

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    dtype = tf.float32

    with tf.variable_scope(variable_scope):

        sess = tf.Session(config=tf_config)

        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 1


        if this_model =='sac':
            dtype = tf.float64
            tf.keras.backend.set_floatx('float64')
            stockactor = SACActor(sess, feature_number = feature_number,
                                          config = config,
                                          action_dim = asset_number + 1,
                                          window_size = window_size,
                                          num_mixtrue = num_mixtrue,
                                          learning_rate = actor_learning_rate,
                                          action_bound=1,
                                          layers = actor_layers,
                                          tau=tau, batch_size=batch_size,dtype=dtype)
        else:

            stockactor = DDPGActor(sess, feature_number = feature_number,
                                          config = config,
                                          action_dim = asset_number + 1,
                                          window_size = window_size,
                                          learning_rate = actor_learning_rate,
                                          action_bound=1,
                                          layers = actor_layers,
                                          tau=tau, batch_size=batch_size)

        stockcritic = DDPGCritic(sess, feature_number = feature_number,
                                        config = config,
                                        action_dim = asset_number+1,
                                        window_size = window_size,
                                        learning_rate = critic_learning_rate,
                                        num_actor_vars = stockactor.get_num_trainable_vars(),
                                        layers = critic_layers,
                                        tau=tau, batch_size=batch_size,dtype=dtype)


        if this_model == 'ddpg':
             model = DDPG(env_training,sess, actor = stockactor,
                                             critic = stockcritic,
                                             obs_normalizer = obs_normalizer,
                                             actor_noise = actor_noise,
                                             model_save_path = model_save_path,
                                             summary_path = summary_path,
                                             config = config)

        elif this_model == 'td3' or this_model == 'sac':

            stockcritic2 = DDPGCritic(sess, feature_number = feature_number,
                                             config = config,
                                             action_dim = asset_number+1,
                                             window_size = window_size,
                                             learning_rate = critic_learning_rate,
                                             num_actor_vars = stockactor.get_num_trainable_vars() + stockcritic.get_num_trainable_vars(),
                                             layers = critic_layers,
                                             tau=tau, batch_size=batch_size,dtype=dtype)
            if this_model == 'td3':
                model = TD3(env_training, sess,  actor = stockactor,
                                                 critic1 = stockcritic,
                                                 critic2 = stockcritic2,
                                                 obs_normalizer = obs_normalizer,
                                                 actor_noise = actor_noise,
                                                 policy_delay = policy_delay,
                                                 model_save_path = model_save_path,
                                                 summary_path = summary_path,
                                                 config = config)
            else:
                model = SAC(env_training, sess,  actor = stockactor,
                                                 critic1 = stockcritic,
                                                 critic2 = stockcritic2,
                                                 obs_normalizer = obs_normalizer,
                                                 actor_noise = actor_noise,
                                                 policy_delay = policy_delay,
                                                 model_save_path = model_save_path,
                                                 summary_path = summary_path,
                                                 config = config)
        else:
            raise("Model not Implemented Error")
        model.initialize(load_weights=False)
        model.train()
