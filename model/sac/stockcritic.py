import tensorflow as tf

from ..td3.stockcritic import TD3Critics


class SACCritics(TD3Critics):

    def __init__(self, sess, critic1, critic2):
        self.critic1 = critic1
        self.critic2 = critic2
        del critic1.optimize
        del critic2.optimize

        self.sess = sess
        self.dtype = critic1.dtype
        self.logalpha = tf.Variable(0,dtype=self.dtype,trainable=True,name='alpha')
        self.alpha = tf.math.exp(self.logalpha)

        self.target_entropy = tf.constant(-critic1.action_dim**2,
                             dtype=self.dtype,name='target_entropy')

        self.target_q_value = tf.placeholder(self.dtype, [None, 1],name='target_q')
        self.bias = tf.placeholder(self.dtype, [None,1],name='bias')
        self.logprob = tf.placeholder(self.dtype, [None,1],name='logprob')

        self.q_out = tf.math.minimum(self.critic1.out,self.critic2.out)
        self.q_target_out = tf.math.minimum(self.critic1.target_out,self.critic2.target_out)
        # self.target_out = self.q_target_out - self.alpha*self.logprob
        # self.out = self.q_out - self.alpha*self.logprob

        self.action_grads_q1 = tf.gradients(self.q_out, self.critic1.action)
        self.action_grads_q2 = tf.gradients(self.q_out, self.critic2.action)
        self.action_grads = [tf.math.add(x,y) for x,y in zip(self.action_grads_q1, self.action_grads_q2)]

        self.action_grads = tf.clip_by_global_norm(self.action_grads,5)[0][0]

        # self.logprob_grads = tf.clip_by_global_norm(tf.gradients(self.out, self.logprob),5)[0][0]

        self.TD_error = tf.math.abs(self.target_q_value-self.q_out)


        self.loss_q1 = tf.keras.losses.MeanSquaredError()(self.target_q_value, self.critic1.out,
                                                                sample_weight=self.bias)

        self.loss_gradients_q1 = tf.clip_by_global_norm(tf.gradients(self.loss_q1, self.critic1.network_params),5)[0]
        self.optimize_q1 = tf.keras.optimizers.Adam(
            self.critic1.lr_schedule,name='mse_adam_1').apply_gradients(zip(self.loss_gradients_q1,self.critic1.network_params))

        self.loss_q2 = tf.keras.losses.MeanSquaredError()(self.target_q_value, self.critic2.out,
                                                                sample_weight=self.bias)

        self.loss_gradients_q2 = tf.clip_by_global_norm(tf.gradients(self.loss_q2, self.critic2.network_params),5)[0]
        self.optimize_q2 = tf.keras.optimizers.Adam(
            self.critic2.lr_schedule,name='mse_adam_2').apply_gradients(zip(self.loss_gradients_q2,self.critic2.network_params))

        self.alpha_loss = tf.reduce_mean(-self.alpha*(self.logprob + self.target_entropy))
        self.alpha_gradient = tf.clip_by_global_norm(tf.gradients(self.alpha_loss,[self.logalpha]),1)[0]
        self.optimize_alpha = tf.train.AdamOptimizer(learning_rate=1e-2,name='alpha_adam').apply_gradients(zip(self.alpha_gradient,[self.logalpha]))

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
        return self.sess.run([self.q_out, self.loss_q1, self.loss_q2,self.optimize_q1,self.optimize_q2], feed_dict={
            self.critic1.inputs[0]: inputs,
            self.critic1.inputs[1]: action,
            self.critic2.inputs[0]: inputs,
            self.critic2.inputs[1]: action,
            self.target_q_value: target_q_value,
            self.bias: bias,
        })

    def val(self, inputs, action, logprob,target_q_value, bias):
        """
        Args:
            inputs: observation
            action: predicted action
            target_q_value:
        """
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run([self.q_out, self.loss_q1, self.loss_q2,self.alpha_loss], feed_dict={
            self.critic1.inputs[0]: inputs,
            self.critic1.inputs[1]: action,
            self.critic2.inputs[0]: inputs,
            self.critic2.inputs[1]: action,
            self.logprob: logprob,
            self.target_q_value: target_q_value,
            self.bias: bias
        })


    def train_alpha(self,logprob):

        return self.sess.run([self.alpha_loss, self.optimize_alpha],feed_dict={
            self.logprob:logprob
        })


    def predict(self, inputs, action):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run(self.q_out, feed_dict={
            self.critic1.inputs[0]: inputs,
            self.critic1.inputs[1]: action,
            self.critic2.inputs[0]: inputs,
            self.critic2.inputs[1]: action
        })

    def predict_target(self, inputs, action):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run(self.q_target_out, feed_dict={
            self.critic1.target_inputs[0]: inputs,
            self.critic1.target_inputs[1]: action,
            self.critic2.target_inputs[0]: inputs,
            self.critic2.target_inputs[1]: action
        })

    def compute_TDerror(self, inputs, action,target_q_value):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run(self.TD_error, feed_dict={
            self.critic1.inputs[0]: inputs,
            self.critic1.inputs[1]: action,
            self.critic2.inputs[0]: inputs,
            self.critic2.inputs[1]: action,
            self.target_q_value: target_q_value
        })

    def action_gradients(self, inputs, action):
        inputs = inputs[:, :, -self.critic1.window_size:, :]
        self.critic1.training = False
        self.critic2.training = False
        return self.sess.run([self.action_grads], feed_dict={
            self.critic1.inputs[0]: inputs,
            self.critic1.inputs[1]: action,
            self.critic2.inputs[0]: inputs,
            self.critic2.inputs[1]: action
        })
