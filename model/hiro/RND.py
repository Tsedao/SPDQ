import tensorflow as tf
import numpy as np

from ..core import nn

class RND(object):
    def __init__(self,sess,
                      layers,
                      feature_number,
                      action_dim,
                      window_size,
                      learning_rate,
                      num_vars,
                      dtype):
        self.sess = sess
        self.layers = layers
        self.feature_number = feature_number
        self.action_dim = action_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.dtype = dtype

        with tf.name_scope("predictor"):
            self.predictor_inputs, self.predictor_outs = self.create_rnd_network()

            self.predictor_outs = tf.keras.layers.Dense(units=self.action_dim)(self.predictor_outs)
            self.predictor_outs = tf.keras.layers.LeakyReLU()(self.predictor_outs)
            self.predictor_outs = tf.keras.layers.Dense(units=self.action_dim)(self.predictor_outs)

        self.network_params = tf.trainable_variables()[num_vars:]

        with tf.name_scope('target'):
            self.target_inputs, self.target_outs = self.create_rnd_network()
            self.target_outs = tf.keras.layers.Dense(units=self.action_dim)(self.target_outs)
        self.target_network_params = tf.trainable_variables()[(len(self.network_params)+num_vars):]

        self.loss = tf.keras.losses.MeanSquaredError()(self.predictor_outs,self.target_outs)
        self.intrinsic_reward = tf.math.reduce_sum(tf.math.pow((self.predictor_outs-self.target_outs),2),axis=-1)/2

        self.loss_gradients = tf.gradients(self.loss, self.network_params)
        self.optimize = tf.keras.optimizers.Adam(
            self.learning_rate).apply_gradients(zip(self.loss_gradients,self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_rnd_network(self):
        self.rnd_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)
        inputs = self.rnd_net.input_tensor
        outs = self.rnd_net.output

        return inputs, outs

    def train(self,inputs):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run([self.optimize,self.loss],feed_dict={
                self.predictor_inputs: inputs,
                self.target_inputs: inputs
        })

    def compute_intrinsic_reward(self,inputs):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.intrinsic_reward, feed_dict={
                self.predictor_inputs: inputs,
                self.target_inputs:    inputs
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
