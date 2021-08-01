"""
Actor Network definition, The CNN architecture follows the one in this paper
https://arxiv.org/abs/1706.10059
Author: Patrick Emami, Modified by Chi Zhang
"""

import tensorflow as tf


# ===========================
#   Actor DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, config, feature_number, action_dim, window_size, learning_rate,
                      action_bound, num_vars=0, tau=0.001, batch_size=128,dtype=tf.float32):
        """
        Args:
            sess: a tensorflow session
            config: a general config file
            action_bound: whether to normalize action in the end
        """
        self.sess = sess
        # assert isinstance(state_dim, list), 'state_dim must be a list.'
        # self.s_dim = state_dim
        # assert isinstance(action_dim, list), 'action_dim must be a list.'
        # self.a_dim = action_dim
        self.feature_number = feature_number
        self.action_dim = action_dim
        self.window_size = window_size
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.config = config
        self.dtype = dtype
        # self.feature_number = config['input']['feature_number']
        # self.action_dim = config['input']['asset_number'] + 1
        # self.window_size = config['input']['window_size']
        # self.learning_rate = config['training']['actor learning rate']
        # self.tau = config['training']['tau']
        # self.batch_size = config['training']['batch size']


        # Actor Network
        with tf.name_scope('network'):
            self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()[num_vars:]

        # Target Network
        with tf.name_scope('target_network'):
            self.target_inputs, self.out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     (len(self.network_params)+num_vars):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        raise NotImplementedError('Create actor should return (inputs, out, scaled_out)')

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs[0]: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs[0]: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs[0]: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
