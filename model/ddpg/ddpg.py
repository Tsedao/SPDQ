"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""

import os
import traceback
import json
import time
import numpy as np
import tensorflow as tf

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..base_model import BaseModel



class DDPG(BaseModel):
    def __init__(self, env, val_env, sess, actor, critic, actor_noise, config, obs_normalizer=None, action_processor=None,
                 model_save_path='weights/ddpg/ddpg.ckpt',
                 best_model_save_path='weights/best_ddpg/ddpg.ckpt',
                 summary_path='results/ddpg/'):
        # with open(config_file) as f:
        #     self.config = json.load(f)
        # assert self.config != None, "Can't load config file"

        self.config = config
        np.random.seed(self.config['training']['seed'])
        if env:
            env.seed(self.config['training']['seed'])

        self.model_save_path = model_save_path
        self.best_model_save_path = best_model_save_path
        self.summary_path = summary_path
        self.sess = sess
        # if env is None, then DDPG just predicts
        self.env = env
        self.val_env = val_env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.summary_ops, self.summary_vars = self.build_summaries('train')
        self.summary_ops_val, self.summary_vars_val = self.build_summaries('val')
        self.summary_vars_r, self.summary_vars_val_r = tf.Variable(0.,dtype=tf.float64), tf.Variable(0.,dtype=tf.float64)
        self.summary_ops_r = tf.summary.scalar("Episode_reward",self.summary_vars_r)
        self.summary_ops_val_r = tf.summary.scalar("Episode_reward_val",self.summary_vars_val_r)

        self.writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        self.num_episode = self.config['training']['episode']
        self.batch_size = self.config['training']['batch_size']
        self.n_step = self.config['training'].get('n_step',1)
        self.training_max_step = self.config['training']['max_step']
        self.training_max_step_size = self.config['training'].get('max_step_size',1)
        self.validating_max_step = self.config['training']['max_step_val']
        self.gamma = self.config['training']['gamma']
        #self.buffer = proportional.Experience(self.config['training']['buffer_size'])
        self.buffer = rank_based.Experience(self.config['training']['buffer_size'])
        self.best_val_reward = 0
        self.reward_summary_dict = {"epi":[],"reward_avg":[],"reward_std":[],"reward_sum":[]}
        np.random.seed(self.config['training']['seed'])

    def build_summaries(self,scope):
        with tf.variable_scope(scope):
            step_loss = tf.Variable(0.)
            a = tf.summary.scalar("step target loss", step_loss)
            step_qmax = tf.Variable(0.)
            b = tf.summary.scalar("step Q max", step_qmax)

        summary_vars = [step_loss, step_qmax]
        summary_ops = [a,b]

        return summary_ops, summary_vars



    def initialize(self, load_weights=True, verbose=True):
        """ Load training history from path. To be add feature to just load weights, not training states
        """
        if load_weights:
            try:
                variables = tf.global_variables()
                param_dict = {}
                saver = tf.train.Saver()
                saver.restore(self.sess, self.model_save_path)
                for var in variables:
                    var_name = var.name[:-2]
                    if verbose:
                        print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
                    param_dict[var_name] = var
            except:
                traceback.print_exc()
                print('Build model from scratch')
                self.sess.run(tf.global_variables_initializer())
        else:
            print('Build model from scratch')
            self.sess.run(tf.global_variables_initializer())

    def save_best_model(self, current_r, best_r):
        if current_r > best_r:
            if not os.path.exists(self.best_model_save_path):
                os.makedirs(self.best_model_save_path, exist_ok=True)
            best_r = current_r
            saver = tf.train.Saver()
            model_path = saver.save(self.sess, self.best_model_save_path)
            print("Best Model with reward %5f saved in %s" % (best_r, model_path))

        return best_r

    def validate_verbose(self, epi_counter, verbose=True):
        """
        Do validation on val env
        Args
            env: val
            buffer: simple replay buffer
        """


        val_ep_reward = 0
        val_ep_max_q = 0
        val_ep_ave_q = 0
        val_ep_loss = 0
        previous_observation, _ = self.val_env.reset()
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)
        val_ep_reward_list = []

        for j in range(self.validating_max_step):
            action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                axis=0)

            # step forward
            observation, reward, done, _ = self.step(self.val_env,previous_observation,action)

            val_ep_reward += reward
            val_ep_reward_list.append(val_ep_reward)
            # Calculate targets
            target_q = self.critic.predict_target(np.expand_dims(observation,axis=0),
                             self.actor.predict_target(np.expand_dims(observation,axis=0)))

            if done:
                y = reward
            else:
                y = reward + self.gamma * target_q[0]


            predicted_q_value, step_loss = self.critic.val(
                                        np.expand_dims(previous_observation,axis=0),
                                        np.expand_dims(action,axis=0),
                                        np.reshape(y, (1, 1)),
                                        np.ones((1,1)))
            previous_observation = observation                                  # update obs
            val_ep_max_q += np.amax(predicted_q_value)
            val_ep_ave_q += np.mean(predicted_q_value)
            val_ep_loss += np.mean(step_loss)

            summaries = self.sess.run(self.summary_ops_val, feed_dict = {
                                    self.summary_vars_val[0] : np.mean(step_loss),
                                    self.summary_vars_val[1] : np.amax(predicted_q_value)
            })

            [self.writer.add_summary(summary, self.validating_max_step*epi_counter+j) for summary in summaries]
            self.writer.flush()

            if done or j == self.validating_max_step - 1:

                print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, Qave: {:.4f}, target_predict_loss: {:.8f}'.format(
                        epi_counter, val_ep_reward, (val_ep_max_q / float(j+1)),
                        (val_ep_ave_q / float(j+1)) ,(val_ep_loss / float(j+1))))
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

    def validate(self, epi_counter, verbose=True):
        """
        Do validation on val env
        Args
            env: val
            buffer: simple replay buffer
        """


        ep_reward = 0
        val_ep_reward_list = []
        previous_observation, _ = self.val_env.reset()
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)

        for j in range(self.validating_max_step):

            action = self.predict_single(previous_observation)
            print('action:',action)
            obs, reward, done, _ = self.val_env.step(action)
            ep_reward += reward
            val_ep_reward_list.append(ep_reward)
            previous_observation = obs

            if  j == self.validating_max_step - 1:
                print("*"*12+'validaing'+"*"*12)
                print('Episode: {:d}, Reward: {:.2f}'.format(
                            epi_counter, ep_reward))
                reward_summary = self.sess.run(self.summary_ops_val_r, feed_dict = {
                                            self.summary_vars_val_r : ep_reward
                })
                self.writer.add_summary(reward_summary,epi_counter)
                self.writer.flush()
                val_ep_reward_avg = np.mean(val_ep_reward_list)
                val_ep_reward_std = np.std(val_ep_reward_list)
                self.reward_summary_dict['epi'].append(epi_counter)
                self.reward_summary_dict['reward_avg'].append(val_ep_reward_avg)
                self.reward_summary_dict['reward_std'].append(val_ep_reward_std)
                self.reward_summary_dict['reward_sum'].append(ep_reward)
                self.best_val_reward = self.save_best_model(ep_reward,self.best_val_reward)
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
        # main training loop
        for i in range(self.num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))
            print("*"*12+'training'+"*"*12)
            previous_observation, _ = self.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            ep_reward = 0
            ep_max_q = 0
            ep_ave_q = 0
            ep_loss = 0

            # keeps sampling until done
            for j in range(self.training_max_step):
                start_time = time.time()

                obs, start_action, rewards, done, TD_errors,ep_max_q, ep_ave_q,ep_loss = self.train_one_step(
                previous_observation,ep_max_q,ep_ave_q,ep_loss,i,j)

                end_time = time.time()

                print("elapsed time {:.4f}s".format(end_time-start_time))
                # add to buffer
                # self.buffer.add(previous_observation, action, reward, done, observation)
                self.buffer.store((previous_observation, start_action, rewards, done, obs),TD_errors)
                ep_reward += rewards
                previous_observation = obs

                if done or j == self.training_max_step - 1:

                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, Qave: {:.4f}, target_predict_loss: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                            (ep_ave_q / float(j+1)) ,(ep_loss / float(j+1))))
                    reward_summary = self.sess.run(self.summary_ops_r, feed_dict = {
                                                self.summary_vars_r : ep_reward
                    })
                    self.writer.add_summary(reward_summary,i)
                    self.writer.flush()
                    break
            print("*"*12+'validaing'+"*"*12)
            self.validate(i)

        self.save_model(verbose=True)
        print('Finish.')

    def train_one_step(self, previous_observation, ep_max_q, ep_ave_q, ep_loss,
                             epi_counter, step_counter):
        TD_errors = 0
        rewards = 0
        # n-step TD-learning
        for n in range(self.n_step):
            action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                axis=0) + self.actor_noise()

            if n == 0:
                start_action = action
                start_obs = previous_observation

            observation, reward, done, _  = self.step(self.env, previous_observation,action)
            previous_observation = observation
            rewards += np.power(self.gamma,n)*reward

        target_q_single = self.critic.predict_target(np.expand_dims(observation, axis=0),
                         self.actor.predict_target(np.expand_dims(observation, axis=0)))

        if done:
            y = np.array([[rewards]])
        else:
            y = rewards + np.power(self.gamma,self.n_step) * target_q_single

        TD_errors = self.critic.compute_TDerror(np.expand_dims(start_obs,axis=0),
                                               np.expand_dims(start_action, axis=0),
                                               y)[0][0][0]


        if self.buffer.size() >= self.batch_size:
            # batch update
            (s_batch, a_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(self.batch_size)
            # Calculate targets
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * target_q[k])

            # Update the critic given the targets

            TD_errors_batch = self.critic.compute_TDerror(s_batch, a_batch,
                                                np.reshape(y_i, (self.batch_size, 1)))[0]

            self.buffer.update_priority(indices, TD_errors_batch.squeeze(axis=1))

            bias = TD_errors_batch.squeeze(axis=1) * np.array(is_weights)  # importance sampling

            predicted_q_value, step_loss, _ = self.critic.train(
                                        s_batch, a_batch,
                                        np.reshape(y_i, (self.batch_size, 1)),
                                        np.expand_dims(bias, axis=1))

            ep_max_q += np.amax(predicted_q_value)
            ep_ave_q += np.mean(predicted_q_value)
            ep_loss += np.mean(step_loss)

            summaries = self.sess.run(self.summary_ops, feed_dict = {
                                    self.summary_vars[0] : np.mean(step_loss),
                                    self.summary_vars[1] : np.amax(predicted_q_value)
            })

            [self.writer.add_summary(summary, self.training_max_step*epi_counter+step_counter
              ) for summary in summaries]
            self.writer.flush()

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)[0]
            self.actor.train(s_batch, grads)

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

        return observation, start_action, rewards, done, TD_errors, ep_max_q, ep_ave_q, ep_loss

    def step(self, env, previous_observation, action):

        # step forward
        observation, reward, done, _ = env.step(action)

        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)

        return observation, reward, done, _

    def _predict_action(self,previous_observation):
        return self.actor.predict(previous_observation)

    def _predict_target_q(self, observation):
        return self.critic.predict_target((observation),
                 self.actor.predict_target(observation))

    def _compute_tderror(self, observation):
        pass

    def _train_actor(self, observation):
        pass

    def _train_critic(self,observation):
        pass

    def predict(self, observation):
        """ predict the next action using actor model, only used in deploy.
            Can be used in multiple environments.
        Args:
            observation: (batch_size, num_stocks + 1, window_length)
        Returns: action array with shape (batch_size, num_stocks + 1)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict_target(observation)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        """ Predict the action of a single observation
        Args:
            observation: (num_stocks + 1, window_length,num_features)
        Returns: a single action array with shape (num_stocks + 1,)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict_target(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_model(self, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)
