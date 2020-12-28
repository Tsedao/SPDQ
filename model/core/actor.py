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

    def __init__(self, sess, config, action_bound):
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
        self.action_bound = action_bound
        # self.learning_rate = learning_rate
        # self.tau = tau
        # self.batch_size = batch_size

        self.config = config
        self.feature_number = config['input']['feature_number']
        self.action_dim = config['input']['asset_number'] + 1
        self.window_size = config['input']['window_size']
        self.learning_rate = config['training']['actor learning rate']
        self.tau = config['training']['tau']
        self.batch_size = config['training']['batch size']


        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None] + [self.action_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max step'],
                                                    decay_rate=0.96,
                                                    staircase=True)

        self.optimize = tf.keras.optimizers.Adam(self.lr_schedule). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        raise NotImplementedError('Create actor should return (inputs, out, scaled_out)')

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
