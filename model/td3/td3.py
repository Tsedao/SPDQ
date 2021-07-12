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

    def validate(self, episode_counter, verbose=True):
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
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)

        for j in range(0,self.config['training']['max_step_val'],self.config['training']['max_step_val_size']):
            action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                axis=0)
            if self.action_processor:
                action_take = self.action_processor(action)
            else:
                action_take = action

            #ic(action_take)
            # step forward
            observation, reward, done, _ = self.val_env.step(action_take)
            if self.obs_normalizer:
                observation = self.obs_normalizer(observation)


            # add to buffer
            # self.buffer.add(previous_observation, action, reward, done, observation)
            self.buffer_val.add(previous_observation, action, reward, done, observation)
            val_ep_reward += reward

            if self.buffer_val.size() >= self.batch_size:
                # batch update
                s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer_val.sample_batch(self.batch_size)

                # Calculate targets
                target_q = self.td3critics.predict_target(s2_batch, self.actor.predict_target(s2_batch))

                y_i = []
                for k in range(self.batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + self.gamma * target_q[k,:])



                predicted_q_value, q1_loss, q2_loss = self.td3critics.val(
                                                    s_batch, a_batch,
                                                    np.reshape(y_i, (self.batch_size, 1)),
                                                    np.ones((self.batch_size,1)))
                val_ep_max_q += np.amax(predicted_q_value)
                val_ep_loss_1 += np.mean(q1_loss)
                val_ep_loss_2 += np.mean(q2_loss)

                summaries = self.sess.run(self.summary_ops_val, feed_dict = {
                                        self.summary_vars_val[0] : np.mean(q1_loss),
                                        self.summary_vars_val[1] : np.mean(q2_loss),
                                        self.summary_vars_val[2] : np.amax(predicted_q_value),
                })

                [self.writer.add_summary(summary, self.config['training']['max_step_val']*episode_counter+j) for summary in summaries]
                self.writer.flush()

            if  j == self.config['training']['max_step_val'] - self.config['training']['max_step_val_size']:

                print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss1: {:.8f}, loss2: {:.8f}'.format(
                            episode_counter, val_ep_reward, (val_ep_max_q / float(j+1)),
                             (val_ep_loss_1 / float(j+1)),
                            (val_ep_loss_2 / float(j+1))))
                reward_summary = self.sess.run(self.summary_ops_val_r, feed_dict = {
                                            self.summary_vars_val_r : val_ep_reward
                })
                self.writer.add_summary(reward_summary,episode_counter)
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
        self.writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        self.actor.update_target_network()
        self.critic.update_target_network()
        self.critic_2.update_target_network()

        np.random.seed(self.config['training']['seed'])
        num_episode = self.config['training']['episode']
        self.batch_size = self.config['training']['batch_size']
        self.gamma = self.config['training']['gamma']
        self.buffer_val = ReplayBuffer(self.config['training']['buffer_size_val'])
        #self.buffer = proportional.Experience(self.config['training']['buffer_size'])
        self.buffer = rank_based.Experience(self.config['training']['buffer_size'])
        # self.buffer_vanila = ReplayBuffer(self.config['training']['buffer_size'])
        self.best_val_reward = 0
        # main training loop
        for i in range(num_episode):
            print("*"*12+'training'+"*"*12)
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation, _ = self.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            ep_reward = 0
            ep_max_q = 0
            ep_loss_1 = 0
            ep_loss_2 = 0
            # keeps sampling until done
            for j in range(0,self.config['training']['max_step'],self.config['training']['max_step_size']):
                start_time = time.time()

                rewards = 0
                TD_errors = 0

                # n-step TD-learning
                for n in range(self.config['training']['n_step']):
                    action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                        axis=0) + self.actor_noise()
                    if n == 0:
                        start_action = action
                        start_observation = previous_observation
                    if self.action_processor:
                        action_take = self.action_processor(action)
                    else:
                        action_take = action
                    # step forward
                    observation, reward, done, _ = self.env.step(action_take)

                    if self.obs_normalizer:
                        observation = self.obs_normalizer(observation)

                    ic(observation.shape)
                    target_q_single = self.td3critics.predict_target(np.expand_dims(observation, axis=0),
                                        self.actor.predict_target(np.expand_dims(observation, axis=0)) + np.expand_dims(self.actor_noise(),axis=0))


                    if done:
                        y = np.expand_dims(np.array([reward]),axis=0)
                        break
                    else:
                        y = reward + self.gamma * target_q_single

                    TD_error = self.td3critics.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                                                           np.expand_dims(action_take, axis=0),
                                                           y)[0][0]
                    previous_observation = observation
                    rewards += np.power(self.gamma,n)*reward
                    TD_errors += TD_error
                # prioritise experience replay
                self.buffer.store((start_observation, start_action, rewards, done, observation),TD_errors)

                # random experience replay
                # self.buffer_vanila.add(previous_observation, action, reward, done, observation)

                if self.buffer.size() >= self.batch_size:

                    (s_batch, a_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(self.batch_size)
                    # s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer_vanila.sample_batch(self.batch_size)

                    # Calculate targets
                    noise = np.vstack([self.actor_noise() for i in range(self.batch_size)])
                    target_q = self.td3critics.predict_target(s2_batch, self.actor.predict_target(s2_batch)+noise)
                    y_i = []
                    TD_errors = []

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
                        TD_errors.append(TD_error)

                    TD_errors = np.array(TD_errors)

                    self.buffer.update_priority(indices, TD_errors)

                    bias = TD_errors * np.array(is_weights)  # importance sampling

                    # bias depends on the temporal difference
                    predicted_q_value, q1_loss, q2_loss, _, _, = self.td3critics.train(
                                                    s_batch, a_batch,
                                                    np.reshape(y_i, (self.batch_size, 1)),
                                                    np.expand_dims(bias, axis=1))
                    # bias is one
                    # predicted_q_value, q1_loss, q2_loss, _, _, = self.td3critics.train(
                    #                                 s_batch, a_batch,
                    #                                 np.reshape(y_i, (self.batch_size, 1)),
                    #                                 np.ones((self.batch_size,1)))

                    ep_max_q += np.amax(predicted_q_value)

                    ep_loss_1 += np.mean(q1_loss)
                    ep_loss_2 += np.mean(q2_loss)

                    summaries = self.sess.run(self.summary_ops, feed_dict = {
                                            self.summary_vars[0] : np.mean(q1_loss),
                                            self.summary_vars[1] : np.mean(q2_loss),
                                            self.summary_vars[2] : np.amax(predicted_q_value),
                    })

                    [self.writer.add_summary(summary, self.config['training']['max_step']*i+self.config['training']['n_step']*j) for summary in summaries]
                    self.writer.flush()

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.td3critics.action_gradients(s_batch, a_outs)

                    if j % self.policy_delay == 0:
                        self.actor.train(s_batch, grads[0])

                        # Update target networks
                        self.actor.update_target_network()
                    self.critic.update_target_network()
                    self.critic_2.update_target_network()

                    end_time = time.time()
                    print("elapsed time {:.4f}s".format(end_time-start_time))

                ep_reward += reward
                previous_observation = observation

                if j == self.config['training']['max_step'] - self.config['training']['max_step_size'] or done:


                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss1: {:.8f}, loss2: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                             (ep_loss_1 / float(j+1)),
                            (ep_loss_2 / float(j+1))))
                    reward_summary = self.sess.run(self.summary_ops_r, feed_dict = {
                                                self.summary_vars_r : ep_reward
                    })
                    self.writer.add_summary(reward_summary,i)
                    self.writer.flush()
                if done:
                    break
            print("*"*12+'validaing'+"*"*12)
            self.validate(i)
        self.save_model(verbose=True)
        print('Finish.')
