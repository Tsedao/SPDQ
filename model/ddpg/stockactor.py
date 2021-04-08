import tensorflow as tf

from ..core.actor import ActorNetwork
from ..core import nn


class DDPGActor(ActorNetwork):
    def __init__(self, sess, config, feature_number, action_dim, window_size, learning_rate,
                      action_bound, layers, tau=0.001, batch_size=128,dtype=tf.float32):
        self.layers = layers
        ActorNetwork.__init__(self, sess, config, feature_number, action_dim,
                   window_size, learning_rate, action_bound, tau, batch_size,dtype=dtype)
        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(self.dtype, [None] + [self.action_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        # new_lr = initial_learning_rate * decay_rate ^ (step / decay_steps)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max_step'],
                                                    decay_rate=config['training']['actor_lr_decay'],
                                                    staircase=True)

        self.optimize = tf.keras.optimizers.Adam(self.lr_schedule). \
            apply_gradients(zip(-1*self.actor_gradients, self.network_params))


    def create_actor_network(self):

        self.actor_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)
        inputs = self.actor_net.input_tensor
        out = self.actor_net.output

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
        self.actor_net.training = True
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]
        self.actor_net.training = False
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        self.actor_net.training = False
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })
