import tensorflow as tf
import numpy as np

from ..core.actor import ActorNetwork
from ..core import nn

class SACActor(object):

    def __init__(self, sess, config, feature_number, action_dim, window_size,num_mixture,learning_rate=0.01,
                      action_bound=None,layers=None,num_vars=0, tau=0.001,tau_softmax=0.01,batch_size=128,dtype=tf.float64):

        self.sess = sess
        self.feature_number = feature_number
        self.action_dim = action_dim
        self.window_size = window_size
        self.num_mixture = num_mixture
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.config = config
        self.dtype = dtype
        self.layers = layers
        self.tau_softmax = tau_softmax
        # Actor Network
        self.inputs, self.scaled_out, self.logprob, self.test_prob, self.test_det, \
            self.test_mu, self.test_sigma, self.test_eps,self.test_x, self.test_mw = self.create_actor_network()

        self.network_params = tf.trainable_variables()[num_vars:]

        # Target Network
        self.target_inputs, self.target_scaled_out, self.target_logprob, self.test_target_prob, self.test_target_det, \
          self.test_target_mu, self.test_target_sigma, self.test_target_eps, self.test_target_x,self.test_target_mw = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     (len(self.network_params))+num_vars:]

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
        self.unnormalized_actor_gradients += self.unnormalized_action_gradients[-2:]
        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), self.unnormalized_actor_gradients))

        self.actor_gradients = tf.clip_by_global_norm(self.actor_gradients,5.0)[0]

        # Optimization Op
        # new_lr = initial_learning_rate * decay_rate ^ (step / decay_steps)
        global_step = tf.Variable(0,trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        self.lr_schedule = tf.train.natural_exp_decay(
                                                    self.learning_rate,
                                                    global_step = global_step,
                                                    decay_steps=config['training']['episode']*config['training']['max_step'],
                                                    decay_rate=config['training']['actor_lr_decay'],
                                                    staircase=True)

        self.optimize = tf.train.AdamOptimizer(self.lr_schedule). \
            apply_gradients(zip([-1*g for g in self.actor_gradients], self.network_params),global_step=global_step)

        self.inputs_grad = tf.gradients(self.scaled_out,self.inputs)
    def create_actor_network(self):

        num_mixture = self.num_mixture
        action_dim = self.action_dim
        time_stamp = self.window_size
        feature_num = self.feature_number
        delta = 1e-15
        dtype = self.dtype


        EPS_TF = tf.constant(delta,dtype=dtype,name='delta')
        self.EPS_TF = EPS_TF
        # feature_input = tf.placeholder(dtype, shape=[None, action_dim,time_stamp,feature_num])
        # out = tf.reshape(feature_input,shape=[-1,action_dim*time_stamp*feature_num])
        # out = tf.keras.layers.Dense(units=128,activation='relu')(out)
        # out = tf.keras.layers.Dense(units=32,activation='relu')(out)
        self.actor_net = nn.CNN(self.feature_number,self.action_dim,
                            self.window_size, self.layers,self.dtype)
        feature_input = self.actor_net.input_tensor
        out = self.actor_net.output
        # with tf.name_scope('tau'):
        #     log_tau = tf.keras.layers.Dense(1)(out)
        #     self.tau_softmax_tf = tf.math.exp(log_tau)
        self.tau_softmax_tf = tf.constant(self.tau_softmax,dtype=self.dtype)
        with tf.name_scope('std'):
            logvar = tf.keras.layers.Dense(32,activation='relu')(out)
            logvar = tf.keras.layers.Dense(units=action_dim*num_mixture,activation=None)(logvar)
            logvar = tf.reshape(logvar,shape=[-1,num_mixture,action_dim])
            # logvar = logvar / tf.reduce_sum(logvar, axis=-1,keepdims=True)
            sigma = tf.math.exp(logvar * .5)
            #sigma = tf.keras.layers.Softmax(axis=-1)(sigma)

        with tf.name_scope('mu'):
            mu = tf.keras.layers.Dense(32,activation='relu')(out)
            mu = tf.keras.layers.Dense(units=action_dim*num_mixture,activation=None)(mu)
            mu = tf.reshape(mu,shape=[-1,num_mixture,action_dim])
            # mu = mu / tf.reduce_sum(mu, axis=-1,keepdims=True)
        with tf.name_scope('mixture_weights'):
            mw = tf.keras.layers.Dense(32,activation='relu')(out)
            mixture_weights = tf.keras.layers.Dense(units=num_mixture,activation='sigmoid')(mw)
            mixture_weights = (mixture_weights+EPS_TF) / (tf.reduce_sum(mixture_weights, axis=-1,keepdims=True)+EPS_TF)
        # with tf.name_scope('tau'):
        #     log_tau = tf.keras.layers.Dense(32,activation='relu')(out)
        #     log_tau = tf.keras.layers.Dense(1)(log_tau)
        #     self.tau_softmax_tf = tf.math.exp(log_tau)
        # log_tau_tf = tf.Variable(initial_value=0,trainable=True,dtype=dtype,name='tau')
        # self.tau_softmax_tf = tf.math.exp(log_tau_tf)
        # reparameterize
        with tf.name_scope('x'):
            eps = tf.random.normal(shape = [tf.shape(mu)[0],1],dtype=dtype)
            m_sigma = tf.math.sqrt(tf.squeeze(tf.matmul(tf.expand_dims(mixture_weights**2,axis=1),sigma**2),axis=1))
            m_mu = tf.squeeze(tf.matmul(tf.expand_dims(mixture_weights,axis=1),mu),axis=1)
            x = eps * m_sigma + m_mu
            print(x.shape)

        #    x = dist_gmm.sample()
        with tf.name_scope('z'):
            z = tf.math.exp(x / self.tau_softmax_tf) / (tf.reduce_sum(tf.math.exp(x / self.tau_softmax_tf),axis=1,keepdims=True) + EPS_TF)
            # z = tf.keras.activations.sigmoid(x)
            # z1 = z / tf.math.reduce_sum(z,axis=-1, keepdims=True)
            print(z.shape)

        with tf.name_scope('det'):
            logabsdet = tf.math.reduce_sum(tf.math.log(z),axis=1,keepdims=True)-action_dim*tf.math.log(self.tau_softmax_tf)

            # det = tf.math.reduce_sum(tf.math.log(z*(1-z)+self.EPS_TF),axis=-1,keepdims=True)
            #det = EPS_TF
            print(logabsdet.shape)

        with tf.name_scope('log-likehood'):
            prob = self.tf_mixtrue_density(pdf=self.tf_norm_pdf_multivariate,x=x,mus=mu,sigmas=sigma,weights=mixture_weights)
            print(prob.shape)
            # logprob_z = tf.math.log(prob) - tf.math.log(tf.math.abs(det))
            logprob_z = tf.math.log(prob) - logabsdet
            print(logprob_z.shape)
            # logprob_z = tf.expand_dims(logprob_z,axis=-1)

            # logprob = tf.expand_dims(tf.math.log(prob),axis=-1)


        return feature_input, z, logprob_z, prob, logabsdet, m_mu, m_sigma, eps,x, mixture_weights

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

    def test_predict(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]

        return self.sess.run([self.scaled_out,self.logprob,
                             self.test_prob,self.test_det,
                             self.test_mu, self.test_sigma,self.test_x,self.test_mw], feed_dict={
            self.inputs: inputs
        })


    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def tf_mixtrue_density(self, pdf, x, mus, sigmas, weights):
        """
        mus: [none, num_mixture, num_actions]
        sigmas: [none, num_mixture, num_actions, num_actions]
        weigths: [none, num_mixture]
        x: [none, num_action]

        return: weighted_density [none]
        """
        num_mixture = mus.shape.as_list()[1]
        un_weighted_density = tf.concat([pdf(x, mus[:,i,:],sigmas[:,i,:]) for i in range(num_mixture)],axis=1)
        weighted_density = tf.reduce_sum(un_weighted_density*weights,axis=1,keepdims=True)
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
            det = tf.math.reduce_prod(sigma,axis=1,keepdims=True)
            size = x.shape.as_list()[-1]
            norm_const = 1.0/ (( tf.cast(tf.math.pow((2.0*np.pi),(size)/2),dtype=self.dtype) * tf.math.pow(det,1.0/2) )+self.EPS_TF)
    #        norm_const = tf.expand_dims(norm_const,axis=1)
            x_mu = x-mu

            inv = 1 / (sigma+self.EPS_TF)
    #        exponential_term = tf.math.exp( -0.5 * (tf.matmul(tf.matmul(x_mu, inv),tf.transpose(x_mu,perm=[0,2,1]))))
            exponential_term = tf.math.exp( -0.5 * tf.reduce_sum(x_mu*inv*x_mu,axis=1,keepdims=True))

    #        exponential_term = tf.squeeze(exponential_term,axis=-1)
            return tf.multiply(norm_const,exponential_term) + self.EPS_TF
            # return tf.multiply(norm_const,exponential_term)
        else:
            raise NameError("The dimensions of the input don't match")

    def cal_inputs_grad(self,inputs):
        inputs = inputs[:, :, -self.window_size:, :]

        return self.sess.run(self.inputs_grad, feed_dict={
            self.inputs: inputs
        })
