import tensorflow as tf

from ..core.actor import ActorNetwork
from ..core import nn


class StockActor(ActorNetwork):
    def __init__(self, sess, config, action_bound):
        ActorNetwork.__init__(self, sess, config, action_bound)


    def create_actor_network(self):

        actor_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.config['actor_layers'])
        inputs = actor_net.input_tensor
        out = actor_net.output

        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        """
        Args:
            inputs: a observation with shape [None, action_dim, window_length, feature_number]
            a_gradient: action gradients flow from the critic network
        """
        inputs = inputs[:, :, -self.window_size:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })
