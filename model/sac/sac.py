import os
import traceback
import json
import numpy as np
import tensorflow as tf

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..ddpg.ddpg import DDPG
from ..sac.stockcritic import SACCritics


class SAC(DDPG):
    """docstring for sac."""
    def __init__(self, env,val_env, sess, actor, critic1, critic2,actor_noise, config,
                 policy_delay = 2, obs_normalizer=None, action_processor=None,
                 model_save_path='weights/sac/sac.ckpt',
                 best_model_save_path = 'weights/best_sac/sac.ckpt',
                 summary_path='results/sac/'):
        super().__init__(env, val_env, sess, actor, critic1, actor_noise, config, obs_normalizer, action_processor,
                     model_save_path, best_model_save_path,summary_path)

        self.critic_2 = critic2
        self.policy_delay = policy_delay
        self.saccritics = SACCritics(sess, critic1, critic2)

    def build_summaries(self, scope):

        with tf.variable_scope(scope):
            step_loss_1 = tf.Variable(0.)
            a = tf.summary.scalar("step target1 loss", step_loss_1)
            step_loss_2 = tf.Variable(0.)
            b = tf.summary.scalar("step target2 loss", step_loss_2)
            step_qmax = tf.Variable(0.)
            c = tf.summary.scalar("step Q max", step_qmax)
            alpha_loss = tf.Variable(0.)
            d = tf.summary.scalar("step alpha loss", alpha_loss)

        summary_vars = [step_loss_1, step_loss_2, step_qmax, alpha_loss]
        summary_ops = [a, b, c, d]

        return summary_ops, summary_vars

    def validate(self, epi_counter, verbose=True):
        """
        Do validation on val env
        Args
            env: val
            buffer: simple replay buffer
        """


        val_ep_reward = 0
        val_ep_max_q = 0
        val_ep_loss_1 = 0
        val_ep_loss_2 = 0
        val_ep_alpha_loss = 0
        previous_observation, _ = self.val_env.reset()
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)

        for j in range(self.config['training']['max_step_val']):
            action, logprob = self.actor.predict(np.expand_dims(previous_observation, axis=0))
            action = np.squeeze(action,axis=0)
            logprob = np.squeeze(logprob,axis=0)
            if self.action_processor:
                action_take = self.action_processor(action)
            else:
                action_take = action
            # step forward
            observation, reward, done, _ = self.val_env.step(action_take)

            if self.obs_normalizer:
                observation = self.obs_normalizer(observation)



            # Calculate targets
            a_t, l_t = self.actor.predict_target(np.expand_dims(observation,axis=0))
            target_q = self.saccritics.predict_target(np.expand_dims(observation,axis=0),
                                                      a_t)

            alpha = self.sess.run(self.saccritics.alpha)
            if done:
                y =  reward
            else:
                y = reward + self.gamma * (target_q[0] - alpha * l_t[0])


            print('y',y)
            print(np.reshape(y, (1, 1)).shape)
            predicted_q_value, q1_loss, q2_loss, alpha_loss = self.saccritics.val(
                                                np.expand_dims(previous_observation,axis=0),
                                                np.expand_dims(action,axis=0),
                                                np.expand_dims(logprob,axis=0),
                                                np.reshape(y, (1, 1)),
                                                np.ones((1,1)))

            val_ep_max_q += np.amax(predicted_q_value)
            val_ep_loss_1 += np.mean(q1_loss)
            val_ep_loss_2 += np.mean(q2_loss)
            val_ep_alpha_loss += np.mean(alpha_loss)

            previous_observation = observation
            val_ep_reward += reward

            summaries = self.sess.run(self.summary_ops_val, feed_dict = {
                                        self.summary_vars_val[0] : np.mean(q1_loss),
                                        self.summary_vars_val[1] : np.mean(q2_loss),
                                        self.summary_vars_val[2] : np.amax(predicted_q_value),
                                        self.summary_vars_val[3] : np.mean(alpha_loss)
            })

            [self.writer.add_summary(summary, self.validating_max_step*epi_counter+j) for summary in summaries]
            self.writer.flush()

            if  j == self.validating_max_step - 1:
                print("*"*12+'validaing'+"*"*12)
                print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss1: {:.8f}, loss2: {:.8f}, alpha loss: {:.8f}'.format(
                            epi_counter, val_ep_reward, (val_ep_max_q / float(j+1)),
                             (val_ep_loss_1 / float(j+1)),
                            (val_ep_loss_2 / float(j+1)),
                            (val_ep_alpha_loss / float(j+1))))
                reward_summary = self.sess.run(self.summary_ops_val_r, feed_dict = {
                                            self.summary_vars_val_r : val_ep_reward
                })
                self.writer.add_summary(reward_summary,epi_counter)
                self.writer.flush()
                self.best_val_reward = self.save_best_model(val_ep_reward,self.best_val_reward)
            if done:
                break

    def train(self, save_every_episode=1, verbose=True, debug=False):
        """ Must already call intialize
        Args:
            save_every_episode:
            print_every_step:
            verbose:
            debug:
        Returns:
        """
        self.actor.update_target_network()
        self.critic.update_target_network()
        self.critic_2.update_target_network()
        np.random.seed(self.config['training']['seed'])
        self.buffer = ReplayBuffer(self.config['training']['buffer_size'])
        # main training loop
        for i in range(self.num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation, _ = self.env.reset()
            # with open('test_before_norm.npy','wb') as f:
            #     np.save('test_before_norm.npy',np.expand_dims(previous_observation, axis=0))
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            ep_reward = 0
            ep_max_q = 0
            ep_loss_1 = 0
            ep_loss_2 = 0
            ep_alpha_loss = 0
            # keeps sampling until done
            for j in range(self.training_max_step):
                rewards = 0
                for n in range(self.n_step):
                    # with open('test_logdiff.npy','wb') as f:
                    #     np.save('test_logdiff.npy',np.expand_dims(previous_observation, axis=0))
                    # action, logprob = self.actor.predict(np.expand_dims(previous_observation, axis=0))
                    action, logprob, p_outs, d_outs, mu, sigma,x, mw = self.actor.test_predict(np.expand_dims(previous_observation, axis=0))

                    # print(np.isnan(previous_observation).any())
                    # print(self.sess.run(self.actor.tau_tf))
                    # print(self.sess.run([self.actor.test_x,
                    #                      self.actor.actor_net.output,
                    #                      self.actor.test_mu,
                    #                      self.actor.test_sigma],feed_dict={self.actor.actor_net.input_tensor:np.expand_dims(previous_observation, axis=0)}))
                    print(np.sum(action,axis=1))
                    print('isnan',np.isnan(previous_observation).any())
                    print('logprob')
                    print(logprob)
                    print('prob')
                    print(p_outs)
                    print('det')
                    print(d_outs)
                    action = np.squeeze(action,axis=0)
                    logprob = np.squeeze(logprob,axis=0)

                    if self.action_processor:
                        action_take = self.action_processor(action)
                    else:
                        action_take = action
                    # step forward
                    observation, reward, done, _ = self.env.step(action_take)

                    if self.obs_normalizer:
                        observation = self.obs_normalizer(observation)



                    # TD_error = self.saccritics.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                    #                                        np.expand_dims(action_take, axis=0),
                    #                                        np.expand_dims(logprob,axis=0),
                    #                                        y)[0][0]
                    # add to buffer
                    # self.buffer.store((previous_observation, action,logprob, reward, done, observation),TD_error)

                    previous_observation = observation
                    rewards += np.power(self.gamma,n)*reward
                action_t, logprob_t = self.actor.predict_target(np.expand_dims(observation, axis=0))
                target_q_single = self.saccritics.predict_target(np.expand_dims(observation, axis=0),
                                                                action_t)

                print('**************Critic networks*****************')
                # target_q_list = [target_q_single_1,target_q_single_2]
                # min_q_ix = np.argmin(target_q_list)
                alpha = self.sess.run(self.saccritics.alpha)
                if done:
                    y = np.array([[rewards]])
                else:
                    y = rewards + np.power(self.gamma,self.n_step) * (target_q_single - alpha * logprob_t)
                self.buffer.add(previous_observation, (action, logprob), rewards, done, observation)
                if self.buffer.size() >= self.batch_size:
                    # batch update
                    #s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    #(s_batch, a_batch,l_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(self.batch_size)

                    s_batch, al_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

                    a_batch, l_batch = np.vstack(al_batch[:,0]), np.vstack(al_batch[:,1])

                    print('s2_batch_is_nan',np.isnan(s2_batch).any())
                    a2_t, l2_b = self.actor.predict_target(s2_batch)
                    target_q = self.saccritics.predict_target(s2_batch, a2_t)
                    alpha = self.sess.run(self.saccritics.alpha)
                    print(target_q)
                    y_i = []
                    TD_errors = []

                    for k in range(self.batch_size):
                        if t_batch[k]:
                            y_tmp = r_batch[k]
                        else:
                            y_tmp = r_batch[k] + self.gamma * (target_q[k,:] - alpha * l2_b[k,:])
                        # Update the critic given the targets


                        # TD_error = self.saccritics.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                        #                                        np.expand_dims(a_batch[k],axis=0),
                        #                                        np.expand_dims(l_batch[k],axis=0),
                        #                                        np.array([y_tmp]))[0][0]
                        y_i.append(y_tmp)
                        # TD_errors.append(TD_error)

                    # TD_errors = np.array(TD_errors)

                    # self.buffer.update_priority(indices, TD_errors)

                    # bias = TD_errors * np.array(is_weights)  # importance sampling


                    # predicted_q_value, q1_loss, q2_loss, _, _, = self.saccritics.train(
                    #                                 s_batch, a_batch,l_batch,
                    #                                 np.reshape(y_i, (self.batch_size, 1)),
                    #                                 np.expand_dims(bias, axis=1))
                    predicted_q_value, q1_loss, q2_loss, _, _, = self.saccritics.train(
                                                    s_batch, a_batch,
                                                    np.reshape(y_i, (self.batch_size, 1)),
                                                    np.ones((self.batch_size,1)))
                    print(y_i)
                    print(predicted_q_value, q1_loss, q2_loss)

                    # Update the actor policy using the sampled gradient
                    a_outs, l_outs, p_outs, d_outs, mu, sigma,x, mw = self.actor.test_predict(s_batch)
                    print(a_outs)
                    print(l_outs)
                    print('mixture_weights')
                    print(mw)
                    print('prob')
                    print(p_outs)
                    print('det')
                    print(d_outs)
                    print('mu')
                    print(mu[0,...])
                    print('sigma')
                    print(sigma[0,...])
                    print('x')
                    print(x[0,...])
                    grads = self.saccritics.action_gradients(s_batch, a_outs)
                    # print(self.sess.run(self.saccritics.alpha))
                    if j % self.policy_delay == 0:
                        self.actor.train(s_batch, *grads)

                        # Update target networks
                        self.actor.update_target_network()
                    self.critic.update_target_network()
                    self.critic_2.update_target_network()

                    alpha_loss, _ = self.saccritics.train_alpha(l_outs)
                    print('alpha loss')
                    print(alpha_loss)
                    print('alpha')
                    print(self.sess.run(self.saccritics.alpha))
                    ep_max_q += np.amax(predicted_q_value)

                    ep_loss_1 += np.mean(q1_loss)
                    ep_loss_2 += np.mean(q2_loss)
                    ep_alpha_loss += np.mean(alpha_loss)

                    summaries = self.sess.run(self.summary_ops, feed_dict = {
                                            self.summary_vars[0] : np.mean(q1_loss),
                                            self.summary_vars[1] : np.mean(q2_loss),
                                            self.summary_vars[2] : np.amax(predicted_q_value),
                                            self.summary_vars[3] : np.mean(alpha_loss)
                    })

                    [self.writer.add_summary(summary, self.training_max_step*i+j) for summary in summaries]
                    self.writer.flush()

                ep_reward += reward
                previous_observation = observation

                if done or j == self.training_max_step - 1:
                    # self.buffer.tree.print_tree()
                    # summary_str = self.sess.run(self.summary_ops, feed_dict={
                    #     self.summary_vars[0]: ep_reward,
                    #     self.summary_vars[1]: ep_max_q / float(j),
                    #     self.summary_vars[2]: ep_loss / float(j)
                    # })
                    #
                    # writer.add_summary(summary_str, i)
                    # writer.flush()
                    print("*"*12+'training'+"*"*12)
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss1: {:.8f}, loss2: {:.8f}, alpha loss: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                             (ep_loss_1 / float(j+1)),
                            (ep_loss_2 / float(j+1)),
                            (ep_alpha_loss / float(j+1))))
                    reward_summary = self.sess.run(self.summary_ops_r, feed_dict = {
                                                self.summary_vars_r : ep_reward
                    })
                    self.writer.add_summary(reward_summary,i)
                    self.writer.flush()
                    break
            self.validate(i,verbose=verbose)
        self.save_model(verbose=True)
        print('Finish.')


    def predict_single(self, observation):
        """ Predict the action of a single observation
        Args:
            observation: (num_stocks + 1, window_length)
        Returns: a single action array with shape (num_stocks + 1,)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)

        action, logprob = self.actor.predict_target(np.expand_dims(observation, axis=0))
        action = np.squeeze(action,axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action
