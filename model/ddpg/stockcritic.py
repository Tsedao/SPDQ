import tensorflow as tf

from ..core.critic import CriticNetwork
from ..core import nn

class DDPGCritic(CriticNetwork):
    def __init__(self, sess, config, feature_number, action_dim, window_size, learning_rate,
                       num_actor_vars, layers, tau=0.001, batch_size=128,dtype=tf.float32):
        self.layers = layers
        CriticNetwork.__init__(self, sess, config, feature_number, action_dim,
                    window_size, learning_rate,num_actor_vars, tau, batch_size,dtype)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.inputs[1])

    def create_critic_network(self):

        self.critic_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)

        inputs = self.critic_net.input_tensor
        action = self.critic_net.predicted_w
        out = self.critic_net.output

        return [inputs,action], out

    def train(self, inputs, action, target_q_value, bias):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = True
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
            self.inputs[0]: inputs,
            self.inputs[1]: action,
            self.target_q_value: target_q_value,
            self.bias: bias
        })

    def val(self, inputs, action, target_q_value, bias):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = False
        return self.sess.run([self.out, self.loss], feed_dict={
            self.inputs[0]: inputs,
            self.inputs[1]: action,
            self.target_q_value: target_q_value,
            self.bias: bias
        })

    def compute_TDerror(self, inputs, action, target_q_value):
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = False
        return self.sess.run([self.TD_error], feed_dict={
            self.inputs[0]: inputs,
            self.inputs[1]: action,
            self.target_q_value: target_q_value
        })

    def predict(self, inputs, action):
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = False
        return self.sess.run(self.out, feed_dict={
            self.inputs[0]: inputs,
            self.inputs[1]: action
        })

    def predict_target(self, inputs, action):
        inputs = inputs[:, :, -self.window_size:, :]
        self.critic_net.training = False
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs[0]: inputs,
            self.target_inputs[1]: action
        })

    def action_gradients(self, inputs, actions):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs[0]: inputs,
            self.inputs[1]: actions
        })
