"""
Critic Network definition, the input is (o, a_{t-1}, a_t) since (o, a_{t-1}) is the state.
Basically, it evaluates the value of (current action, previous action and observation) pair
"""

import tensorflow as tf


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, config, feature_number, action_dim, window_size, learning_rate,
                       num_actor_vars, tau=0.001, batch_size=128,dtype=tf.float32):
        self.sess = sess
        # assert isinstance(state_dim, list), 'state_dim must be a list.'
        # self.s_dim = state_dim
        # assert isinstance(action_dim, list), 'action_dim must be a list.'
        # self.a_dim = action_dim

        self.feature_number = feature_number
        self.action_dim = action_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.dtype = dtype
        self.config = config
        # self.feature_number = config['input']['feature_number']
        # self.action_dim = config['input']['asset_number'] + 1
        # self.window_size = config['input']['window_size']
        # self.learning_rate = config['training']['critic learning rate']
        # self.tau = config['training']['tau']
        # self.batch_size = config['training']['batch size']

        # Create the critic network
        with tf.name_scope('network'):
            self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        with tf.name_scope('target_network'):
            self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.target_q_value = tf.placeholder(self.dtype, [None, 1])

        # Sampling bias (w_i * TD_error_i)
        self.bias = tf.placeholder(self.dtype, [None,1])

        # Define loss and optimization Op
        self.loss = tf.keras.losses.MeanSquaredError()(self.target_q_value, self.out,
                                                        sample_weight=self.bias)

        self.TD_error = tf.math.abs(self.target_q_value-self.out)

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max_step'],
                                                    decay_rate=config['training']['critic_lr_decay'],
                                                    staircase=True)

        self.loss_gradients = tf.gradients(self.loss, self.network_params)
        self.optimize = tf.keras.optimizers.Adam(
            self.lr_schedule,name='mse_adam').apply_gradients(zip(self.loss_gradients,self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_critic_network(self):
        raise NotImplementedError('Create critic should return (inputs, action, out)')

    def train(self, inputs, action, target_q_value):
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.target_q_value: target_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
