import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from ..core import nn

tfd = tfp.distributions
class PPOActor(object):

    def __init__(self, sess, config,
                        feature_number,
                        action_dim,
                        window_size,
                        num_mixture,
                        ent_coef,
                        learning_rate=0.01,
                        action_bound=None,
                        layers=None,
                        ppo_eps=0.05,
                        num_vars=0,
                        tau_softmax=1,
                        batch_size=128,
                        dtype=tf.float32):

        self.sess = sess
        self.feature_number = feature_number
        self.action_dim = action_dim
        self.window_size = window_size
        self.num_mixture = num_mixture
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ppo_eps = ppo_eps
        self.ent_coef = ent_coef
        self.config = config
        self.dtype = dtype
        self.layers = layers
        self.tau_softmax = tau_softmax
        # Actor Network
        self.inputs, self.scaled_out, self.logprob, self.test_prob, self.test_det, \
            self.test_mu, self.test_sigma, self.test_eps,self.test_x, self.test_mw = self.create_actor_network()

        self.network_params = tf.trainable_variables()[num_vars:]


        self.num_trainable_vars = len(self.network_params)

        self.advantage = tf.placeholder(shape=[None,1],dtype=self.dtype,name='adv')
        self.behavior_logpi = tf.placeholder(shape=[None,1], dtype=self.dtype,name='behavior_pi')

        self.ratio = tf.math.exp(self.logprob-self.behavior_logpi)
        self.surr1 = self.ratio * self.advantage
        self.surr2 = tf.clip_by_value(
            self.ratio,
            1.0 - self.ppo_eps,
            1.0 + self.ppo_eps) * self.advantage

        self.loss = -tf.math.reduce_mean(tf.math.minimum(self.surr1, self.surr2),axis=0)-self.ent_coef*self.logprob

        self.loss_gradients = tf.gradients(self.loss,self.network_params)
        self.loss_gradients, _ = tf.clip_by_global_norm(self.loss_gradients,5.0)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                    self.learning_rate,
                                                    decay_steps=config['training']['episode']*config['training']['max_step'],
                                                    decay_rate=config['training']['actor_lr_decay'],
                                                    staircase=True)

        self.optimize = tf.keras.optimizers.Adam(self.lr_schedule). \
            apply_gradients(zip(self.loss_gradients,self.network_params))

    def create_actor_network(self):

        num_mixture = self.num_mixture
        action_dim = self.action_dim
        time_stamp = self.window_size
        feature_num = self.feature_number
        delta = 1e-18
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
            mixture_weights = (mixture_weights+EPS_TF) / (tf.reduce_sum((mixture_weights+EPS_TF), axis=-1,keepdims=True))
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
            components_list = [tfd.MultivariateNormalDiag(loc=mu[:,i,:], scale_diag=sigma[:,i,:]) for i in range(num_mixture)]
            mix_gauss = tfd.Mixture(
              cat=tfd.Categorical(probs=mixture_weights),
              components=components_list)

            # x = mix_gauss.sample()

        with tf.name_scope('z'):
            z = tf.math.exp(x / self.tau_softmax_tf) / (tf.reduce_sum(tf.math.exp(x / self.tau_softmax_tf),axis=1,keepdims=True) + EPS_TF)
            # z = tf.keras.activations.sigmoid(x)
            # z1 = z / tf.math.reduce_sum(z,axis=-1, keepdims=True)


        with tf.name_scope('det'):
            logabsdet = tf.math.reduce_sum(tf.math.log(z),axis=1,keepdims=True)-action_dim*tf.math.log(self.tau_softmax_tf)

            # det = tf.math.reduce_sum(tf.math.log(z*(1-z)+self.EPS_TF),axis=-1,keepdims=True)
            #det = EPS_TF


        with tf.name_scope('log-likehood'):
            # prob = self.tf_mixtrue_density(pdf=self.tf_norm_pdf_multivariate,x=x,mus=mu,sigmas=sigma,weights=mixture_weights)
            logprob = mix_gauss.log_prob(x)
            logprob = tf.expand_dims(logprob,axis=-1)
            logprob_z = logprob - logabsdet



        return feature_input, z, logprob_z, mix_gauss.prob(x), logabsdet, m_mu, m_sigma, eps,x, mixture_weights


    def train(self, inputs, adv, behavior_logpi):
        inputs = inputs[:, :, -self.window_size:, :]
        return self.sess.run([self.loss, self.loss_gradients, self.logprob, self.optimize],feed_dict={
                        self.inputs : inputs,
                        self.advantage : adv,
                        self.behavior_logpi : behavior_logpi
        })
    def predict(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]

        return self.sess.run([self.scaled_out,self.logprob], feed_dict={
            self.inputs: inputs
        })

    def test_predict(self, inputs):
        inputs = inputs[:, :, -self.window_size:, :]

        return self.sess.run([self.scaled_out,self.logprob,
                             self.test_prob,self.test_det,
                             self.test_mu, self.test_sigma,self.test_x,self.test_mw], feed_dict={
            self.inputs: inputs
        })


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
