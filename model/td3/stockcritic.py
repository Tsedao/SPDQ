import tensorflow as tf

from ..core.critic import CriticNetwork
from ..core import nn

class TD3Critics(object):

    def __init__(self,sess,critic1, critic2):
        self.critic1 = critic1
        self.critic2 = critic2
        self.sess = sess
        self.dtype = critic1.dtype

        self.target_q_value = tf.placeholder(self.dtype, [None, 1])
        self.bias = tf.placeholder(self.dtype, [None,1])

        self.target_out = tf.math.minimum(self.critic1.target_out,self.critic2.target_out)
        self.out = tf.math.minimum(self.critic1.out,self.critic2.out)

        self.action_grads_q1 = tf.gradients(self.out, self.critic1.action)
        self.action_grads_q2 = tf.gradients(self.out, self.critic2.action)
        self.action_grads = [tf.math.add(x,y) for x,y in zip(self.action_grads_q1, self.action_grads_q2)]

        self.TD_error_1 = tf.math.abs(self.target_q_value-self.critic1.out)
        self.TD_error_2 = tf.math.abs(self.target_q_value-self.critic2.out)
        self.TD_error = tf.math.add(self.TD_error_1, self.TD_error_2)

        self.loss_q1 = tf.keras.losses.MeanSquaredError()(self.target_q_value, self.critic1.out,
                                                                sample_weight=self.bias)

        self.loss_gradients_q1 = tf.gradients(self.loss_q1, self.critic1.network_params)
        self.optimize_q1 = tf.keras.optimizers.Adam(
            self.critic1.lr_schedule).apply_gradients(zip(self.loss_gradients_q1,self.critic1.network_params))

        self.loss_q2 = tf.keras.losses.MeanSquaredError()(self.target_q_value, self.critic2.out,
                                                                sample_weight=self.bias)

        self.loss_gradients_q2 = tf.gradients(self.loss_q2, self.critic2.network_params)
        self.optimize_q2 = tf.keras.optimizers.Adam(
            self.critic2.lr_schedule).apply_gradients(zip(self.loss_gradients_q2,self.critic2.network_params))

    def train(self, inputs, action, target_q_value, bias):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = True
        self.critic2.training = True
        return self.sess.run([self.out, self.loss_q1, self.loss_q2,self.optimize_q1,self.optimize_q2], feed_dict={
            self.critic1.inputs: inputs,
            self.critic1.action: action,
            self.critic2.inputs: inputs,
            self.critic2.action: action,
            self.target_q_value: target_q_value,
            self.bias: bias
        })

    def compute_TDerror(self, inputs, action, target_q_value):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run(self.TD_error, feed_dict={
            self.critic1.inputs: inputs,
            self.critic1.action: action,
            self.critic2.inputs: inputs,
            self.critic2.action: action,
            self.target_q_value: target_q_value
        })

    def predict(self, inputs, action):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run(self.out, feed_dict={
            self.critic1.inputs: inputs,
            self.critic1.action: action,
            self.critic2.inputs: inputs,
            self.critic2.action: action
        })

    def predict_target(self, inputs, action):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run(self.target_out, feed_dict={
            self.critic1.target_inputs: inputs,
            self.critic1.target_action: action,
            self.critic2.target_inputs: inputs,
            self.critic2.target_action: action
        })

    def action_gradients(self, inputs, action):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run(self.action_grads, feed_dict={
            self.critic1.inputs: inputs,
            self.critic1.action: action,
            self.critic2.inputs: inputs,
            self.critic2.action: action
        })
