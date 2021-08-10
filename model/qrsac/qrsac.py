import os
import traceback
import json
import numpy as np
import tensorflow as tf

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..ddpg.ddpg import DDPG


class QRSAC(DDPG):
    """docstring for qrsac."""
    def __init__(self, env, val_env,sess, actor, critic, actor_noise, config,
                 policy_delay = 2, obs_normalizer=None, action_processor=None,
                 model_save_path='weights/qrsac/qrsac.ckpt',
                 best_model_save_path='weights/best_qrsac/qrsac.ckpt',
                 summary_path='results/qrsac/'):
        super().__init__(env, val_env, sess, actor, critic, actor_noise, config, obs_normalizer, action_processor,
                     model_save_path, best_model_save_path, summary_path)

        self.policy_delay = policy_delay
        # self.buffer = rank_based.Experience(self.config['training']['buffer_size'])

    def build_summaries(self,scope):
        with tf.variable_scope(scope):
            step_loss = tf.Variable(0.)
            a = tf.summary.scalar("step target1 loss", step_loss)
            step_qmax = tf.Variable(0.)
            b = tf.summary.scalar("step Q max", step_qmax)
            alpha_loss = tf.Variable(0.)
            c = tf.summary.scalar("step alpha loss", alpha_loss)

        summary_vars = [step_loss, step_qmax, alpha_loss]
        summary_ops = [a, b, c]

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
        val_ep_loss = 0
        val_ep_alpha_loss = 0
        previous_observation, _ = self.val_env.reset()
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)
        val_ep_reward_list = []
        for j in range(self.validating_max_step):
            action, logprob = self.actor.predict(np.expand_dims(previous_observation, axis=0))
            action = np.squeeze(action,axis=0)
            logprob = np.squeeze(logprob,axis=0)

            # step forward
            observation, reward, done, _ = self.val_env.step(action)

            if self.obs_normalizer:
                observation = self.obs_normalizer(observation)

            a2_t, lp2_t = self.actor.predict_target(np.expand_dims(observation,axis=0))
            # Calculate targets
            target_q = self.critic.predict_target(np.expand_dims(observation,axis=0),
                                                a2_t)
            alpha = self.sess.run(self.critic.alpha)
            if done:
                y = reward*np.ones(self.critic.num_quart)
            else:
                y = reward + self.gamma * (target_q[0,:] - alpha*lp2_t)


            val_ep_reward_list.append(reward)
            predicted_q_value, q_loss,alpha_loss = self.critic.val(
                                                    np.expand_dims(previous_observation,axis=0),
                                                    np.expand_dims(action,axis=0),
                                                    np.expand_dims(logprob,axis=0),
                                                    np.reshape(y, (1, self.critic.num_quart)),
                                                    np.ones((1,1)))

            val_ep_max_q += np.amax(np.mean(predicted_q_value,axis=1))
            val_ep_loss += np.mean(q_loss)
            val_ep_alpha_loss += np.mean(alpha_loss)

            val_ep_reward += reward
            previous_observation = observation

            summaries = self.sess.run(self.summary_ops_val, feed_dict = {
                                        self.summary_vars_val[0] : np.mean(q_loss),
                                        self.summary_vars_val[1] : np.amax(np.mean(predicted_q_value,axis=1)),
                                        self.summary_vars_val[2] : np.mean(alpha_loss)
            })

            [self.writer.add_summary(summary, self.validating_max_step*epi_counter+j) for summary in summaries]
            self.writer.flush()

            if  j == self.validating_max_step - 1 or done:
                print("*"*12+'validaing'+"*"*12)
                print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss: {:.8f}, alpha loss: {:.8f}'.format(
                            epi_counter, val_ep_reward, (val_ep_max_q / float(j+1)),
                             (val_ep_loss / float(j+1)),
                            (val_ep_alpha_loss / float(j+1))))
                reward_summary = self.sess.run(self.summary_ops_val_r, feed_dict = {
                                            self.summary_vars_val_r : val_ep_reward
                })
                self.writer.add_summary(reward_summary,epi_counter)
                self.writer.flush()

                val_ep_reward_avg = np.mean(val_ep_reward_list)
                val_ep_reward_std = np.std(val_ep_reward_list)
                self.reward_summary_dict['epi'].append(epi_counter)
                self.reward_summary_dict['reward_avg'].append(val_ep_reward_avg)
                self.reward_summary_dict['reward_std'].append(val_ep_reward_std)
                self.reward_summary_dict['reward_sum'].append(val_ep_reward)
                self.best_val_reward = self.save_best_model(val_ep_reward,self.best_val_reward)
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
        self.writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

        self.actor.update_target_network()
        self.critic.update_target_network()
        np.random.seed(self.config['training']['seed'])

        #self.buffer = proportional.Experience(self.config['training']['buffer_size'])
        # self.buffer = rank_based.Experience(self.config['training']['buffer_size'])

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
            self.ep_reward_list = []
            ep_max_q = 0
            ep_loss = 0
            ep_alpha_loss = 0
            # keeps sampling until done
            for j in range(self.training_max_step):
                start_obs, start_a,start_l, rewards,done, obs, TD_errors, ep_reward, ep_max_q, ep_loss,ep_alpha_loss = self.train_one_step(
                                                                        previous_observation,
                                                                        ep_reward,
                                                                        ep_max_q,
                                                                        ep_loss,
                                                                        ep_alpha_loss,
                                                                        i,j)
                ep_reward += rewards
                self.buffer.store((start_obs, start_a,start_l, rewards, done, obs),TD_errors)
                previous_observation = obs
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

                    reward_avg = np.mean(self.ep_reward_list)
                    reward_std = np.std(self.ep_reward_list)
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss: {:.8f},  alpha loss: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                             (ep_loss / float(j+1)),
                            (ep_alpha_loss / float(j+1))))

                    reward_summary = self.sess.run(self.summary_ops_r, feed_dict = {
                                                self.summary_vars_r : ep_reward
                    })
                    self.writer.add_summary(reward_summary,i)
                    self.writer.flush()
                    break
            self.validate(i)
        self.save_model(verbose=True)
        print('Finish.')

    def train_one_step(self,previous_observation,ep_reward,ep_max_q, ep_loss,ep_alpha_loss,
                       epi_counter, step_counter):
        rewards = 0
        for n in range(self.n_step):
            # with open('test_logdiff.npy','wb') as f:
            #     np.save('test_logdiff.npy',np.expand_dims(previous_observation, axis=0))
            print(np.isnan(previous_observation).any())
            print(np.isinf(previous_observation).any())
            print('obs',previous_observation[:,:,-1].T)
            action, logprob, p_outs, d_outs, mu, sigma,x, mw = self.actor.test_predict(np.expand_dims(previous_observation, axis=0))
            # action, logprob = self.actor.predict(np.expand_dims(previous_observation, axis=0))
            print('action',action)
            print('logprob',logprob)
            # print('eps',eps)
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
            action = np.squeeze(action,axis=0)
            logprob = np.squeeze(logprob,axis=0)

            if n == 0:
                start_observation = previous_observation
                start_action = action
                start_logprob = logprob

            # step forward
            observation, reward, done, _ = self.env.step(action)

            if self.obs_normalizer:
                observation = self.obs_normalizer(observation)

            rewards += np.power(self.gamma,n)*reward
            previous_observation = observation


        action_t, logp_t = self.actor.predict_target(np.expand_dims(observation, axis=0))
        target_q_single = self.critic.predict_target(np.expand_dims(observation, axis=0),
                            action_t)


        # target_q_list = [target_q_single_1,target_q_single_2]
        # min_q_ix = np.argmin(target_q_list)
        alpha = self.sess.run(self.critic.alpha)
        print('alpha',alpha)
        if done:
            y = np.ones(shape=(1,self.critic.num_quart))
        else:
            y = rewards + np.power(self.gamma,self.n_step) * (target_q_single - alpha * logp_t)


        TD_errors = self.critic.compute_TDerror(np.expand_dims(start_observation,axis=0),
                                               np.expand_dims(start_action, axis=0),
                                               y)[0]
        self.ep_reward_list.append(rewards)
        # add to buffer
        # self.buffer.add(previous_observation, action, reward, done, observation)
        # self.buffer.store((start_observation, start_action,start_logprob, rewards, done, observation),TD_error)

        if self.buffer.size() >= self.batch_size:
            # batch update
            #s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
            (s_batch, a_batch,l_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(self.batch_size)

            a2_t, lp2_t = self.actor.predict_target(s2_batch)
            target_q = self.critic.predict_target(s2_batch, a2_t)
            y_i = []
            TD_errors_list = []
            alpha = self.sess.run(self.critic.alpha)
            for k in range(self.batch_size):

                if t_batch[k]:
                    y_tmp = r_batch[k]*np.ones(shape=(self.critic.num_quart))
                else:
                    y_tmp = r_batch[k] + self.gamma * (target_q[k,...] - alpha * lp2_t[k,...])
                # Update the critic given the targets


                TD_error = self.critic.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                                                       np.expand_dims(a_batch[k],axis=0),
                                                       np.array([y_tmp]))[0]
                y_i.append(y_tmp)
                TD_errors_list.append(TD_error)

            TD_errors_batch = np.array(TD_errors_list)

            self.buffer.update_priority(indices, TD_errors_batch)

            bias = TD_errors * np.expand_dims(np.array(is_weights),axis=1)  # importance sampling


            predicted_q_value, q_loss,  _ = self.critic.train(
                                            s_batch, a_batch,
                                            np.reshape(y_i, (self.batch_size, self.critic.num_quart)),
                                            bias)


            # Update the actor policy using the sampled gradient
            a_outs, l_outs, p_outs, d_outs, mu, sigma,x, mw = self.actor.test_predict(s_batch)
            # print(a_outs)
            # print(l_outs)
            # print('mixture_weights')
            # print(mw)
            # print('prob')
            # print(p_outs)
            # print('det')
            # print(d_outs)
            # print('mu')
            # print(mu[0,...])
            # print('sigma')
            # print(sigma[0,...])
            # print('x')
            # print(x[0,...])
            grads = self.critic.action_logprob_gradients(s_batch, a_outs,l_outs)
            # print(self.sess.run(self.saccritics.alpha))
            if step_counter % self.policy_delay == 0:
                self.actor.train(s_batch, *grads)

                # Update target networks
                self.actor.update_target_network()
            self.critic.update_target_network()

            alpha_loss, _ = self.critic.train_alpha(l_outs)

            ep_max_q += np.amax(np.mean(predicted_q_value,axis=1))

            ep_loss += np.mean(q_loss)
            ep_alpha_loss += np.mean(alpha_loss)

            summaries = self.sess.run(self.summary_ops, feed_dict = {
                                    self.summary_vars[0] : np.mean(q_loss),
                                    self.summary_vars[1] : np.amax(np.mean(predicted_q_value,axis=1)),
                                    self.summary_vars[2] : np.mean(alpha_loss)
            })

            [self.writer.add_summary(summary, self.training_max_step*epi_counter+step_counter) for summary in summaries]
            self.writer.flush()

        return start_observation, start_action, start_logprob, rewards, observation, done, TD_errors, ep_reward, ep_max_q, ep_loss,ep_alpha_loss

    def predict_single(self, observation):
        """ Predict the action of a single observation
        Args:
            observation: (num_stocks + 1, window_length)
        Returns: a single action array with shape (num_stocks + 1,)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action, logprob = self.actor.predict(np.expand_dims(observation, axis=0))
        action = np.squeeze(action,axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action
