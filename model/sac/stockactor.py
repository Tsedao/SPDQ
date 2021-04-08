import tensorflow as tf
import numpy as np

from ..core.actor import ActorNetwork
from ..core import nn

class SACActor(object):

    def __init__(self, sess, config, feature_number, action_dim, window_size,num_mixtrue,learning_rate=0.01,
                      action_bound=None,layers=None,dtype=tf.float64, tau=0.001, batch_size=128):

        self.sess = sess
        self.feature_number = feature_number
        self.action_dim = action_dim
        self.window_size = window_size
        self.num_mixtrue = num_mixtrue
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.config = config
        self.dtype = dtype
        self.layers = layers
        # Actor Network
        self.inputs, self.scaled_out, self.logprob = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_scaled_out, self.target_logprob = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(self.dtype, [None] + [self.action_dim],name='action_gradient')
        self.logprob_gradient = tf.placeholder(self.dtype,[None,1],name='logprob_gradient')

        # Combine the gradients here
        self.unnormalized_action_gradients = tf.gradients(
            self.scaled_out, self.network_params, self.action_gradient)
        self.unnormalized_logprob_gradients = tf.gradients(
            self.logprob, self.network_params, self.logprob_gradient)
        self.unnormalized_actor_gradients = [tf.math.add(x,y) for x, y in zip(
            self.unnormalized_action_gradients,self.unnormalized_logprob_gradients)]
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

        num_mixtrue = self.num_mixtrue
        action_dim = self.action_dim
        time_stamp = self.window_size
        feature_num = self.feature_number
        delta = 1e-8
        dtype = self.dtype

        log_tau_tf = tf.Variable(initial_value=1,trainable=True,dtype=dtype,name='tau')
        tau_tf = tf.math.exp(log_tau_tf)
        
        delta_tf = tf.constant(delta,dtype=dtype,name='delta')

        # feature_input = tf.placeholder(dtype, shape=[None, action_dim,time_stamp,feature_num])
        # out = tf.reshape(feature_input,shape=[-1,action_dim*time_stamp*feature_num])
        # out = tf.keras.layers.Dense(units=128,activation='relu')(out)
        # out = tf.keras.layers.Dense(units=32,activation='relu')(out)
        self.actor_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)
        feature_input = self.actor_net.input_tensor
        out = self.actor_net.output


        with tf.name_scope('std'):
            logvar = tf.keras.layers.Dense(units=action_dim*num_mixtrue)(out)
            logvar = tf.reshape(logvar,shape=[-1,num_mixtrue,action_dim])
            sigma = tf.math.exp(logvar * .5)

        with tf.name_scope('mu'):
            mu = tf.keras.layers.Dense(units=action_dim*num_mixtrue,activation='relu')(out)
            mu = tf.reshape(mu,shape=[-1,num_mixtrue,action_dim])

        with tf.name_scope('mixture_weights'):
            mixture_weights = tf.keras.layers.Dense(units=num_mixtrue,activation='softmax')(out)

        # dist_cat = tfd.Categorical(probs=mixture_weights)
        # dist_multin = []
        # for i in range(num_mixtrue):
        #     multin = tfd.MultivariateNormalDiag(loc=mu[:,i,:],scale_diag=tf.math.exp(logvar[:,i,:] * .5))
        #     dist_multin.append(multin)

        # dist_gmm = tfd.Mixture(cat=dist_cat,components=dist_multin)
        # reparameterize
        with tf.name_scope('x'):
            eps = tf.random.normal(shape = tf.shape(mu),dtype=dtype)
            x = eps * sigma + mu
            x = tf.squeeze(tf.matmul(tf.expand_dims(mixture_weights,axis=1),x),axis=1)
        #    x = dist_gmm.sample()
        with tf.name_scope('z'):
            z = tf.math.exp(x / tau_tf) / (tf.reduce_sum(tf.math.exp(x / tau_tf),axis=1,keepdims=True) + delta_tf)


        with tf.name_scope('det'):
            det = (1-tf.reduce_sum(z,axis=1))*tf.math.reduce_prod(z,axis=1)*(1/(tau_tf**action_dim))

        with tf.name_scope('log-likehood'):
            prob = self.tf_mixtrue_density(pdf=self.tf_norm_pdf_multivariate,x=x,mus=mu,sigmas=sigma,weights=mixture_weights)
            logprob_z = tf.math.log(prob) - tf.math.log(det)
            logprob_z = tf.expand_dims(logprob_z,axis=-1)
            # logprob = tf.expand_dims(tf.math.log(prob),axis=-1)


        return feature_input, z, logprob_z

    def train(self, inputs, a_gradient,l_gradient):
        """
        Args:
            inputs: a observation with shape [None, action_dim, window_length, feature_number]
            a_gradient: action gradients flow from the critic network
        """
        inputs = inputs[:, :, -self.window_size:, :]

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.logprob_gradient: l_gradient
        })

    def predict(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]

        return self.sess.run([self.scaled_out,self.logprob], feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):

        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run([self.target_scaled_out,self.target_logprob], feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def tf_mixtrue_density(self, pdf, x, mus, sigmas, weights):
        """
        mus: [none, num_mixtrue, num_actions]
        sigmas: [none, num_mixtrue, num_actions, num_actions]
        weigths: [none, num_mixtrue]
        x: [none, num_action]

        return: weighted_density [none]
        """
        num_mixtrue = mus.shape.as_list()[1]
        un_weighted_density = tf.stack([pdf(x, mus[:,i,:],sigmas[:,i,:]) for i in range(num_mixtrue)],axis=1)
        weighted_density = tf.reduce_sum(un_weighted_density*weights,axis=1)
        return weighted_density

    def tf_norm_pdf_multivariate(self, x, mu, sigma):
        """
        mu: [none, num_action]
        sigma: [none, num_action]
        x: [none, num_action]

        return: [none]
        """
        size = x.shape.as_list()[-1]
        if size == mu.shape.as_list()[-1] and size == sigma.shape.as_list()[-1]:
            det = tf.math.reduce_prod(sigma,axis=1)
            size = x.shape.as_list()[-1]
            norm_const = 1.0/ ( tf.cast(tf.math.pow((2.0*np.pi),(size)/2),dtype=self.dtype) * tf.math.pow(det,1.0/2) )
    #        norm_const = tf.expand_dims(norm_const,axis=1)
            x_mu = x-mu

            inv = 1 / sigma
    #        exponential_term = tf.math.exp( -0.5 * (tf.matmul(tf.matmul(x_mu, inv),tf.transpose(x_mu,perm=[0,2,1]))))
            exponential_term = tf.math.exp( -0.5 * tf.reduce_sum(x_mu*inv*x_mu,axis=1))
    #        exponential_term = tf.squeeze(exponential_term,axis=-1)
            return tf.multiply(norm_const,exponential_term)
        else:
            raise NameError("The dimensions of the input don't match")
