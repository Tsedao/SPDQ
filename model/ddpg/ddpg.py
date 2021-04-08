"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""

import os
import traceback
import json
import numpy as np
import tensorflow as tf

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..base_model import BaseModel



class DDPG(BaseModel):
    def __init__(self, env, sess, actor, critic, actor_noise, config, obs_normalizer=None, action_processor=None,
                 model_save_path='weights/ddpg/ddpg.ckpt', summary_path='results/ddpg/'):
        # with open(config_file) as f:
        #     self.config = json.load(f)
        # assert self.config != None, "Can't load config file"

        self.config = config
        np.random.seed(self.config['training']['seed'])
        if env:
            env.seed(self.config['training']['seed'])
        self.model_save_path = model_save_path
        self.summary_path = summary_path
        self.sess = sess
        # if env is None, then DDPG just predicts
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.summary_ops, self.summary_vars = self.build_summaries()

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax_Value", episode_ave_max_q)
        episode_loss = tf.Variable(0.)
        tf.summary.scalar("target loss", episode_loss)
        step_loss = tf.Variable(0.)
        tf.summary.scalar("step target loss", step_loss)
        step_qmax = tf.Variable(0.)
        tf.summary.scalar("step Q max", step_qmax)
        summary_vars = [episode_reward, episode_ave_max_q, episode_loss, step_loss, step_qmax]
        summary_ops = tf.summary.merge_all()

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

    def train(self, save_every_episode=1, verbose=True, debug=False):
        """ Must already call intialize
        Args:
            save_every_episode:
            print_every_step:
            verbose:
            debug:
        Returns:
        """
        writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

        self.actor.update_target_network()
        self.critic.update_target_network()

        np.random.seed(self.config['training']['seed'])
        num_episode = self.config['training']['episode']
        batch_size = self.config['training']['batch_size']
        gamma = self.config['training']['gamma']
        #self.buffer = ReplayBuffer(self.config['training']['buffer_size'])
        #self.buffer = proportional.Experience(self.config['training']['buffer_size'])
        self.buffer = rank_based.Experience(self.config['training']['buffer_size'])

        # main training loop
        for i in range(num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation, _ = self.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            ep_reward = 0
            ep_max_q = 0
            ep_ave_q = 0
            ep_loss = 0
            # keeps sampling until done
            for j in range(self.config['training']['max_step']):
                action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(
                    axis=0) + self.actor_noise()

                if self.action_processor:
                    action_take = self.action_processor(action)
                else:
                    action_take = action
                # step forward
                observation, reward, done, _ = self.env.step(action_take)

                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation)


                target_q_single = self.critic.predict_target(np.expand_dims(observation, axis=0),
                                    self.actor.predict_target(np.expand_dims(observation, axis=0)))

                if done:
                    y = reward
                else:
                    y = reward + target_q_single

                TD_error = self.critic.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                                                       np.expand_dims(action_take, axis=0),
                                                       y)[0][0][0]
                # add to buffer
                # self.buffer.add(previous_observation, action, reward, done, observation)
                self.buffer.store((previous_observation, action, reward, done, observation),TD_error)

                if self.buffer.size() >= batch_size:
                    # batch update
                    #s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    (s_batch, a_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(batch_size)
                    # Calculate targets
                    target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

                    y_i = []
                    for k in range(batch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + gamma * target_q[k])

                    # Update the critic given the targets

                    TD_errors = self.critic.compute_TDerror(s_batch, a_batch,
                                                        np.reshape(y_i, (batch_size, 1)))

                    self.buffer.update_priority(indices, TD_errors[0].squeeze(axis=1))

                    bias = TD_errors[0].squeeze(axis=1) * np.array(is_weights)  # importance sampling

                    predicted_q_value, step_loss, _ = self.critic.train(
                                                s_batch, a_batch,
                                                np.reshape(y_i, (batch_size, 1)),
                                                np.expand_dims(bias, axis=1))

                    ep_max_q += np.amax(predicted_q_value)
                    ep_ave_q += np.mean(predicted_q_value)
                    ep_loss += np.mean(step_loss)

                    summary_step_loss = self.sess.run(self.summary_ops, feed_dict = {
                                            self.summary_vars[3] : np.mean(step_loss),
                                            self.summary_vars[4] : np.amax(predicted_q_value)
                    })

                    writer.add_summary(summary_step_loss, self.config['training']['max_step']*i+j)
                    writer.flush()

                    # Update the actor policy using the sampled gradient
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs)
                    self.actor.train(s_batch, grads[0])

                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                ep_reward += reward
                previous_observation = observation

                if done or j == self.config['training']['max_step'] - 1:
                    # self.buffer.tree.print_tree()
                    # summary_str = self.sess.run(self.summary_ops, feed_dict={
                    #     self.summary_vars[0]: ep_reward,
                    #     self.summary_vars[1]: ep_max_q / float(j),
                    #     self.summary_vars[2]: ep_loss / float(j)
                    # })
                    #
                    # writer.add_summary(summary_str, i)
                    # writer.flush()

                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, Qave: {:.4f}, target_predict_loss: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                            (ep_ave_q / float(j+1)) ,(ep_loss / float(j+1))))
                    break

        self.save_model(verbose=True)
        print('Finish.')

    def predict(self, observation):
        """ predict the next action using actor model, only used in deploy.
            Can be used in multiple environments.
        Args:
            observation: (batch_size, num_stocks + 1, window_length)
        Returns: action array with shape (batch_size, num_stocks + 1)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(observation)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation):
        """ Predict the action of a single observation
        Args:
            observation: (num_stocks + 1, window_length)
        Returns: a single action array with shape (num_stocks + 1,)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_model(self, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)
