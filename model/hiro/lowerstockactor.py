import tensorflow as tf

from ..ddpg.stockactor import DDPGActor
from ..core import nn

class LowerDDPGActor(DDPGActor):
    """docstring for LowerDDPGActor."""

    def __init__(self, sess,
                       config,
                       feature_number,
                       action_dim,
                       window_size,
                       learning_rate,
                       action_bound,
                       layers,
                       num_vars,
                       tau=0.001,
                       batch_size=128,
                       dtype=tf.float32):
        DDPGActor.__init__(self, sess=sess, config=config,
                        feature_number=feature_number, action_dim=action_dim,
                        window_size=window_size, learning_rate=learning_rate,
                        action_bound=action_bound,layers=layers,
                        num_vars=num_vars, tau=tau, batch_size=batch_size,dtype=dtype)

    def create_actor_network(self):


        self.actor_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)
        inputs = self.actor_net.input_tensor
        out = self.actor_net.output                                             # [None, asset_num, window_size]
        subgoal = tf.placeholder(shape=[None,self.action_dim],dtype=self.dtype,
                                    name='subgoal')
        embeded_sg = tf.keras.layers.Dense(self.window_size,
                                            use_bias=False)(subgoal)            # [None, window_size]
        out = out @ tf.expand_dims(embeded_sg,axis=-1)
        out = tf.squeeze(out, axis=-1)
        out = tf.keras.layers.LayerNormalization()(out)
        out = tf.keras.activations.tanh(out)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)

        return [inputs,subgoal],  out, scaled_out

    def train(self, inputs,subgoal, a_gradient):
        """
        Args:
            inputs: a observation with shape [None, action_dim, window_length, feature_number]
            a_gradient: action gradients flow from the critic network
        """
        inputs = inputs[:, :, -self.window_size:, :]
        self.actor_net.training = True
        self.sess.run(self.optimize, feed_dict={
            self.inputs[0]: inputs,
            self.inputs[1]:subgoal,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs,subgoal):
        inputs = inputs[:, :, -self.window_size:, :]
        self.actor_net.training = False
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs[0]: inputs,
            self.inputs[1]:subgoal
        })

    def predict_target(self, inputs, subgoal):
        self.actor_net.training = False
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs[0]: inputs,
            self.target_inputs[1]: subgoal
        })
