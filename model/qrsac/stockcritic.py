import tensorflow as tf
import numpy as np

from ..core import nn

class QRCritic(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, config, feature_number, action_dim, window_size, num_quart, learning_rate,
                       num_actor_vars, layers, tau=0.001, batch_size=128,dtype=tf.float32):
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
        self.num_quart = num_quart
        self.tau = tau
        self.dtype = dtype
        self.config = config
        # self.feature_number = config['input']['feature_number']
        # self.action_dim = config['input']['asset_number'] + 1
        # self.window_size = config['input']['window_size']
        # self.learning_rate = config['training']['critic learning rate']
        # self.tau = config['training']['tau']
        # self.batch_size = config['training']['batch size']
        self.layers = layers
        self.qrtau = tf.constant(np.expand_dims((2*np.arange(self.num_quart)+1)/(2.0*self.num_quart),axis=0),dtype=self.dtype)
        # Create the critic network
        with tf.name_scope('network'):
            self.inputs, self.action, self.quart_out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        with tf.name_scope('target_network'):
            self.target_inputs, self.target_action, self.target_quart_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.logalpha = tf.Variable(0,dtype=self.dtype,trainable=True,name='alpha')
        self.alpha = tf.math.exp(self.logalpha)


        self.target_entropy = tf.constant(-self.action_dim**2,dtype=self.dtype,name='target_entropy')

        self.logprob = tf.placeholder(self.dtype, [None,1],name='logprob')
        # Network target (y_i)
        self.target_q_value = tf.placeholder(self.dtype, [None, self.num_quart])

        # Sampling bias (w_i * TD_error_i)
        self.bias = tf.placeholder(self.dtype, [None,1],name='bias')

        self.out = self.quart_out - self.alpha*self.logprob
        # self.target_out = self.target_quart_out - self.alpha*self.logprob

        # Define loss and optimization Op
        diff = self.target_q_value - self.quart_out
        loss = self.huber(diff) * tf.math.abs(self.qrtau - tf.cast(diff<0,self.dtype))
        self.loss = tf.math.reduce_mean(self.bias*loss, axis=1,keepdims=True)


        self.TD_error = tf.math.abs(tf.math.reduce_mean(diff,axis=1,keepdims=True))

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max_step'],
                                                    decay_rate=config['training']['critic_lr_decay'],
                                                    staircase=True)

        self.loss_gradients = tf.gradients(self.loss, self.network_params)
        self.optimize = tf.keras.optimizers.Adam(
            self.lr_schedule).apply_gradients(zip(self.loss_gradients,self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

        self.action_grads = tf.gradients(self.out, self.action)[0]
        self.logprob_grads = tf.gradients(self.out, self.logprob)[0]


        self.alpha_loss = tf.reduce_mean(-self.alpha*(self.logprob + self.target_entropy))
        self.alpha_gradient = tf.gradients(self.alpha_loss,[self.logalpha])
        global_step = tf.Variable(0,trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        self.lr_schedule = tf.train.natural_exp_decay(
                                                    1e-3,
                                                    global_step = global_step,
                                                    decay_steps=config['training']['episode']*config['training']['max_step'],
                                                    decay_rate=0.9,
                                                    staircase=True)
        self.optimize_alpha = tf.train.AdamOptimizer(self.lr_schedule,name='alpha_adam').apply_gradients(zip(self.alpha_gradient,[self.logalpha]))

    def create_critic_network(self):

        self.critic_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)

        inputs = self.critic_net.input_tensor
        action = self.critic_net.predicted_w
        out = self.critic_net.output
        quart_out = tf.keras.layers.Dense(self.num_quart,activation=None,dtype=self.dtype)(out)
        return inputs, action, quart_out

    def train(self, inputs, action, target_q_value, bias):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = True
        return self.sess.run([self.quart_out, self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.target_q_value: target_q_value,
            self.bias: bias
        })

    def val(self, inputs, action, logprob, target_q_value, bias):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = False
        return self.sess.run([self.quart_out, self.loss, self.alpha_loss], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.logprob:logprob,
            self.target_q_value: target_q_value,
            self.bias: bias
        })

    def compute_TDerror(self, inputs, action, target_q_value):
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = False
        return self.sess.run(self.TD_error, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.target_q_value: target_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.quart_out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_quart_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_logprob_gradients(self, inputs, actions,logprob):
        return self.sess.run([self.action_grads,self.logprob_grads], feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.logprob: logprob
        })

    def train_alpha(self,logprob):

        return self.sess.run([self.alpha_loss, self.optimize_alpha],feed_dict={
            self.logprob:logprob
        })


    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


    def huber(self, x, k=1.0):
        return tf.where(tf.math.abs(x) < k, 0.5 * tf.math.pow(x,2), k * (tf.math.abs(x) - 0.5 * k))
