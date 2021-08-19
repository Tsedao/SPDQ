import os
import traceback
import json
import time
import numpy as np
import tensorflow as tf

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..ddpg.ddpg import DDPG
from ..td3.stockcritic import TD3Critics

from icecream import ic

class TD3(DDPG):
    """docstring for TD3."""
    def __init__(self, env, val_env, sess, actor, critic1, critic2,actor_noise, config,
                 policy_delay = 2, obs_normalizer=None, action_processor=None,
                 model_save_path='weights/td3/td3.ckpt',
                 best_model_save_path = 'weights/best_td3/td3.ckpt',
                 summary_path='results/td3/'):
        super().__init__(env, val_env, sess, actor, critic1, actor_noise, config, obs_normalizer, action_processor,
                     model_save_path, best_model_save_path, summary_path)

        self.critic_2 = critic2
        self.policy_delay = policy_delay
        self.td3critics = TD3Critics(sess, critic1, critic2)

    def build_summaries(self,scope):

        with tf.variable_scope(scope):
            step_loss_1 = tf.Variable(0.)
            a = tf.summary.scalar("step_target1_loss", step_loss_1)
            step_loss_2 = tf.Variable(0.)
            b = tf.summary.scalar("step_target2_loss", step_loss_2)
            step_qmax = tf.Variable(0.)
            c = tf.summary.scalar("step_Q_max", step_qmax)


        summary_vars = [step_loss_1, step_loss_2, step_qmax]
        summary_ops = [a, b, c]

        return summary_ops, summary_vars

    def validate_verbose(self, epi_counter, verbose=True):
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
        previous_observation, _ = self.val_env.reset()
        # previous_observation = self.val_env.reset()
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)

        for j in range(0,self.config['training']['max_step_val'],self.config['training']['max_step_val_size']):
            action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                axis=0)

            observation, reward, done, _ = self.step(self.val_env,previous_observation,action)
            val_ep_reward += reward


            # Calculate targets
            target_q = self.td3critics.predict_target(np.expand_dims(observation, axis=0),
                                        self.actor.predict_target(np.expand_dims(observation, axis=0)))

            if done:
                y = reward
            else:
                y = reward + self.gamma * target_q[0]


            predicted_q_value, q1_loss, q2_loss = self.td3critics.val(
                                                np.expand_dims(previous_observation, axis=0),
                                                np.expand_dims(action,axis=0),
                                                np.reshape(y, (1, 1)),
                                                np.ones((1,1)))

            val_ep_max_q += np.amax(predicted_q_value)
            val_ep_loss_1 += np.mean(q1_loss)
            val_ep_loss_2 += np.mean(q2_loss)

            previous_observation = observation                                  # reassign obs

            summaries = self.sess.run(self.summary_ops_val, feed_dict = {
                                    self.summary_vars_val[0] : np.mean(q1_loss),
                                    self.summary_vars_val[1] : np.mean(q2_loss),
                                    self.summary_vars_val[2] : np.amax(predicted_q_value),
            })

            [self.writer.add_summary(summary, self.config['training']['max_step_val']*epi_counter+j) for summary in summaries]
            self.writer.flush()

            if  j == self.config['training']['max_step_val'] - self.config['training']['max_step_val_size'] or done:

                print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss1: {:.8f}, loss2: {:.8f}'.format(
                            epi_counter, val_ep_reward, (val_ep_max_q / float(j+1)),
                             (val_ep_loss_1 / float(j+1)),
                            (val_ep_loss_2 / float(j+1))))
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
        # self.writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        self.actor.update_target_network()
        self.critic.update_target_network()
        self.critic_2.update_target_network()

        # np.random.seed(self.config['training']['seed'])
        # num_episode = self.config['training']['episode']
        # self.batch_size = self.config['training']['batch_size']
        # self.gamma = self.config['training']['gamma']
        # self.buffer_val = ReplayBuffer(self.config['training']['buffer_size_val'])
        #self.buffer = proportional.Experience(self.config['training']['buffer_size'])
        # self.buffer = rank_based.Experience(self.config['training']['buffer_size'])
        # self.buffer = ReplayBuffer(self.config['training']['buffer_size'])
        self.best_val_reward = 0
        # main training loop
        for i in range(self.num_episode):
            print("*"*12+'training'+"*"*12)
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation, _ = self.env.reset()
            # previous_observation = self.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            ep_reward = 0
            ep_max_q = 0
            ep_loss_1 = 0
            ep_loss_2 = 0
            # keeps sampling until done
            for j in range(0,self.training_max_step):
                start_time = time.time()

                rewards = 0
                TD_errors = 0

                obs, start_action, rewards, done, TD_errors, ep_max_q, ep_loss_1, ep_loss_2 = self.train_one_step(
                    previous_observation,ep_max_q,ep_loss_1,ep_loss_2,i,j)

                end_time = time.time()
                print("elapsed time {:.4f}s".format(end_time-start_time))
                # prioritise experience replay
                self.buffer.store((previous_observation, start_action, rewards, done, obs),TD_errors)

                # random experience replay
                # self.buffer.add(previous_observation, action, reward, done, observation)
                ep_reward += rewards
                previous_observation = obs

                if j == self.training_max_step - 1 or done:


                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss1: {:.8f}, loss2: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                             (ep_loss_1 / float(j+1)),
                            (ep_loss_2 / float(j+1))))
                    reward_summary = self.sess.run(self.summary_ops_r, feed_dict = {
                                                self.summary_vars_r : ep_reward
                    })
                    # print('g1',g1)
                    # print('g2',g2)
                    # print('actor_g_critic',grads)
                    # print('actor_g',actor_g)
                    self.writer.add_summary(reward_summary,i)
                    self.writer.flush()
                if done:
                    break
            print("*"*12+'validaing'+"*"*12)
            self.validate(i)
        self.save_model(verbose=True)
        print('Finish.')

    def train_one_step(self, previous_observation, ep_max_q, ep_loss_1, ep_loss_2,
                        epi_counter, step_counter):
        rewards = 0
        TD_errors = 0
        # n-step TD-learning
        for n in range(self.n_step):
            action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                axis=0) + self.actor_noise()
            if n == 0:
                start_action = action
                start_obs = previous_observation

            # step forward
            observation, reward, done, _ = self.step(self.env, previous_observation,action)

            previous_observation = observation
            rewards += np.power(self.gamma,n)*reward

        target_q_single = self.td3critics.predict_target(np.expand_dims(observation, axis=0),
                            self.actor.predict_target(np.expand_dims(observation, axis=0)) + np.expand_dims(self.actor_noise(),axis=0))

        if done:
            y = np.expand_dims(np.array([rewards]),axis=0)
        else:
            y = rewards + np.power(self.gamma,self.n_step) * target_q_single

        TD_errors = self.td3critics.compute_TDerror(np.expand_dims(start_obs,axis=0),
                                               np.expand_dims(start_action, axis=0),
                                               y)[0][0]


        if self.buffer.size() >= self.batch_size:

            (s_batch, a_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(self.batch_size)
            # s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

            # Calculate targets
            noise = np.vstack([self.actor_noise() for i in range(self.batch_size)])
            target_q = self.td3critics.predict_target(s2_batch, self.actor.predict_target(s2_batch)+noise)
            # critic1_weights, critic2_weights,w1,w2, t_out, t_out_2, out, out2 = self.td3critics.inspect_targets_weights(s2_batch,
            #                     self.actor.predict_target(s2_batch))
            y_i = []
            TD_errors_list = []
            for k in range(self.batch_size):
                if t_batch[k]:
                    y_tmp = np.array([r_batch[k]])
                else:
                    y_tmp = r_batch[k] + self.gamma * target_q[k,:]
                # Update the critic given the targets
                TD_error = self.td3critics.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                                                       np.expand_dims(a_batch[k],axis=0),
                                                       np.array([y_tmp]))[0][0]
                y_i.append(y_tmp)
                TD_errors_list.append(TD_error)

            TD_errors_batch = np.array(TD_errors_list)
            self.buffer.update_priority(indices, TD_errors_batch)

            bias = TD_errors_batch * np.array(is_weights)  # importance sampling

            # bias depends on the temporal difference
            predicted_q_value, q1_loss, q2_loss, _, _ = self.td3critics.train(
                                            s_batch, a_batch,
                                            np.reshape(y_i, (self.batch_size, 1)),
                                            np.expand_dims(bias, axis=1))
            # bias is one
            # predicted_q_value, q1_loss, q2_loss, g1,g2,_, _ = self.td3critics.train(
            #                                 s_batch, a_batch,
            #                                 np.reshape(y_i, (self.batch_size, 1)),
            #                                 np.ones((self.batch_size,1)))

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.td3critics.action_gradients(s_batch, a_outs)
            if step_counter % self.policy_delay == 0:
                self.actor.train(s_batch, grads[0])
                # Update target networks
                self.actor.update_target_network()
            self.critic.update_target_network()
            self.critic_2.update_target_network()

            ep_max_q += np.amax(predicted_q_value)

            ep_loss_1 += np.mean(q1_loss)
            ep_loss_2 += np.mean(q2_loss)
            summaries = self.sess.run(self.summary_ops, feed_dict = {
                                    self.summary_vars[0] : np.mean(q1_loss),
                                    self.summary_vars[1] : np.mean(q2_loss),
                                    self.summary_vars[2] : np.amax(predicted_q_value),
            })

            [self.writer.add_summary(summary, self.training_max_step*epi_counter+self.n_step*step_counter
                            ) for summary in summaries]
            self.writer.flush()

        return observation, start_action, rewards, done, TD_errors, ep_max_q, ep_loss_1, ep_loss_2
