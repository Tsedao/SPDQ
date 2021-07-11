import os
import traceback
import json
import numpy as np
import time
import tensorflow as tf

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..ddpg.ddpg import DDPG


class D3PG(DDPG):
    """docstring for distributional deep deterministic policy gradient (d3pg)."""
    def __init__(self, env, val_env,sess, actor, critic, actor_noise, config,
                 policy_delay = 1, obs_normalizer=None, action_processor=None,
                 model_save_path='weights/d3pg/d3pg.ckpt',
                 best_model_save_path='weights/best_d3pg/d3pg.ckpt',
                 summary_path='results/d3pg/'):
        super().__init__(env, val_env, sess, actor, critic, actor_noise, config, obs_normalizer, action_processor,
                     model_save_path, best_model_save_path, summary_path)

        self.policy_delay = policy_delay


    def build_summaries(self,scope):
        with tf.variable_scope(scope):
            step_loss = tf.Variable(0.)
            a = tf.summary.scalar("step target1 loss", step_loss)
            step_qmax = tf.Variable(0.)
            b = tf.summary.scalar("step Q max", step_qmax)


        summary_vars = [step_loss, step_qmax]
        summary_ops = [a, b]

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
        val_ep_loss = 0
        self.best_val_reward = 0
        previous_observation, _ = self.val_env.reset()
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)

        for j in range(0, self.config['training']['max_step_val'],self.config['training']['max_step_val_size']):
            action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                   axis=0)

            if self.action_processor:
                action_take = self.action_processor(action)
            else:
                action_take = action
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
                target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

                y_i = []
                for k in range(self.batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + self.gamma * target_q[k,:])



                predicted_q_value, q_loss = self.critic.val(
                                                    s_batch, a_batch,
                                                    np.reshape(y_i, (self.batch_size, self.critic.num_quart)),
                                                    np.ones((self.batch_size,1)))

                val_ep_max_q += np.amax(np.mean(predicted_q_value,axis=1))
                val_ep_loss += np.mean(q_loss)

                summaries = self.sess.run(self.summary_ops_val, feed_dict = {
                                            self.summary_vars_val[0] : np.mean(q_loss),
                                            self.summary_vars_val[1] : np.amax(np.mean(predicted_q_value,axis=1))
                })

                [self.writer.add_summary(summary, self.config['training']['max_step_val']*episode_counter+j) for summary in summaries]
                self.writer.flush()

            if  j == self.config['training']['max_step_val'] - self.config['training']['max_step_val_size'] or done:
                print("*"*12+'validaing'+"*"*12)
                print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss: {:.8f}'.format(
                            episode_counter, val_ep_reward, (val_ep_max_q / float(j+1)),
                             (val_ep_loss / float(j+1))))
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


        np.random.seed(self.config['training']['seed'])
        num_episode = self.config['training']['episode']
        self.batch_size = self.config['training']['batch_size']
        self.gamma = self.config['training']['gamma']
        self.buffer_val = ReplayBuffer(self.config['training']['buffer_size_val'])
        #self.buffer = proportional.Experience(self.config['training']['buffer_size'])
        self.buffer = rank_based.Experience(self.config['training']['buffer_size'])

        # main training loop
        for i in range(num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation, _ = self.env.reset()
            # with open('test_before_norm.npy','wb') as f:
            #     np.save('test_before_norm.npy',np.expand_dims(previous_observation, axis=0))
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            ep_reward = 0
            ep_max_q = 0
            ep_loss = 0
            # keeps sampling until done
            for j in range(0,self.config['training']['max_step'],self.config['training']['max_step_size']):
                # with open('test_logdiff.npy','wb') as f:
                #     np.save('test_logdiff.npy',np.expand_dims(previous_observation, axis=0))

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


                    target_q_single = self.critic.predict_target(np.expand_dims(observation, axis=0),
                                        self.actor.predict_target(np.expand_dims(observation, axis=0)) +
                                        np.expand_dims(self.actor_noise(),axis=0))


                    # target_q_list = [target_q_single_1,target_q_single_2]
                    # min_q_ix = np.argmin(target_q_list)

                    if done:
                        y = reward*np.ones(shape=(1,self.config['training'].get(
                                                                'num_quart',1)))
                    else:
                        y = reward + self.gamma * target_q_single


                    TD_error = self.critic.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                                                           np.expand_dims(action_take, axis=0),
                                                           y)[0]
                    previous_observation = observation
                    rewards += np.power(self.gamma,n)*reward
                    TD_errors += TD_error
                # add to buffer
                # self.buffer.add(previous_observation, action, reward, done, observation)
                self.buffer.store((start_observation, start_action, rewards, done, observation),TD_errors)

                if self.buffer.size() >= self.batch_size:
                    # batch update
                    #s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    (s_batch, a_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(self.batch_size)

                    noise = np.vstack([self.actor_noise() for i in range(self.batch_size)])
                    target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch)+noise)
                    y_i = []
                    TD_errors = []

                    for k in range(self.batch_size):
                        if t_batch[k]:
                            y_tmp = r_batch[k]
                        else:
                            y_tmp = r_batch[k] + self.gamma * target_q[k,...]
                        # Update the critic given the targets


                        TD_error = self.critic.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                                                               np.expand_dims(a_batch[k],axis=0),
                                                               np.array([y_tmp]))[0]
                        y_i.append(y_tmp)
                        TD_errors.append(TD_error)

                    TD_errors = np.array(TD_errors)

                    self.buffer.update_priority(indices, TD_errors)

                    bias = TD_errors * np.expand_dims(np.array(is_weights),axis=1)  # importance sampling


                    predicted_q_value, q_loss,  _  = self.critic.train(
                                                    s_batch, a_batch,
                                                    np.reshape(y_i, (self.batch_size, self.critic.num_quart)),
                                                    bias)


                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs)
                    if j % self.policy_delay == 0:
                        self.actor.train(s_batch, grads)

                        # Update target networks
                        self.actor.update_target_network()
                    self.critic.update_target_network()


                    ep_max_q += np.amax(np.mean(predicted_q_value,axis=1))

                    ep_loss += np.mean(q_loss)


                    summaries = self.sess.run(self.summary_ops, feed_dict = {
                                            self.summary_vars[0] : np.mean(q_loss),
                                            self.summary_vars[1] : np.amax(np.mean(predicted_q_value,axis=1)),
                    })

                    [self.writer.add_summary(summary, self.config['training']['max_step']*i+j) for summary in summaries]
                    self.writer.flush()

                ep_reward += reward
                previous_observation = observation

                if done or self.config['training']['max_step'] - self.config['training']['max_step_size']:

                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                             (ep_loss / float(j+1))))
                    reward_summary = self.sess.run(self.summary_ops_r, feed_dict = {
                                                self.summary_vars_r : ep_reward
                    })
                    self.writer.add_summary(reward_summary,i)
                    self.writer.flush()
                    break
            self.validate(i)
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
        action = self.actor.predict(np.expand_dims(observation, axis=0))
        action = np.squeeze(action,axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action
