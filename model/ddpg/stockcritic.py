import tensorflow as tf

from ..core.critic import CriticNetwork
from ..core import nn

class StockCritic(CriticNetwork):
    def __init__(self, sess, config, num_actor_vars):
        CriticNetwork.__init__(self, sess, config, num_actor_vars)

    def create_critic_network(self):

        critic_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.config['critic_layers'])

        inputs = critic_net.input_tensor
        action = critic_net.predicted_w
        out = critic_net.output

        return inputs, action, out

    def train(self, inputs, action, target_q_value):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.target_q_value: target_q_value
        })

    def predict(self, inputs, action):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })
