import tensorflow as tf

from ..core import nn

class PPOCritic(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess,
                       config,
                       feature_number,
                       action_dim,
                       window_size,
                       layers,
                       num_actor_vars,
                       learning_rate=0.001,
                       batch_size=128,
                       dtype=tf.float32):
        self.sess = sess

        self.feature_number = feature_number
        self.action_dim = action_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dtype = dtype
        self.config = config
        self.layers = layers

        # Create the critic network
        with tf.name_scope('network'):
            self.inputs, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]


        # Network target (y_i)
        self.target_q_value = tf.placeholder(self.dtype, [None, 1],name='target_q')

        # Define loss and optimization Op
        self.loss = tf.keras.losses.MeanSquaredError()(self.target_q_value, self.out)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max_step'],
                                                    decay_rate=config['training']['critic_lr_decay'],
                                                    staircase=True)
        self.loss_gradients = tf.gradients(self.loss, self.network_params)
        self.loss_gradients, _ = tf.clip_by_global_norm(self.loss_gradients,5.0)
        self.optimize = tf.keras.optimizers.Adam(
            self.lr_schedule,name='mse_adam').apply_gradients(zip(self.loss_gradients,self.network_params))
        self.num_trainable_vars = len(self.network_params)

    def create_critic_network(self):

        self.critic_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)

        inputs = self.critic_net.input_tensor
        out = self.critic_net.output

        return inputs, out

    def train(self, inputs, target_q_value):
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.target_q_value: target_q_value
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
