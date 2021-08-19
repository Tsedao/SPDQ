import os
import traceback
import json
import numpy as np
import tensorflow as tf

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..ddpg.ddpg import DDPG
from ..sac.stockcritic import SACCritics


class PPO(DDPG):
    """docstring for sac."""
    def __init__(self, env,
                       val_env,
                       sess,
                       actor,
                       critic,
                       actor_noise,
                       config,
                       lam,
                       obs_normalizer=None,
                       action_processor=None,
                       model_save_path='weights/ppo/ppo.ckpt',
                       best_model_save_path = 'weights/best_ppo/ppo.ckpt',
                       summary_path='results/ppo/'):
        super().__init__(env, val_env, sess, actor, critic, actor_noise, config, obs_normalizer, action_processor,
                     model_save_path, best_model_save_path,summary_path)

        self.lam = lam
        self.epochs = 200
        self.update_steps = self.training_max_step * 4



    def build_summaries(self, scope):

        with tf.variable_scope(scope):
            actor_loss = tf.Variable(0.)
            a = tf.summary.scalar("actor_loss", actor_loss)
            critic_loss = tf.Variable(0.)
            b = tf.summary.scalar("critic_loss", critic_loss)

        summary_vars = [actor_loss, critic_loss]
        summary_ops = [a, b]

        return summary_ops, summary_vars

    def validate(self, epi_counter, verbose=True):
        """
        Do validation on val env
        Args
            env: val
            buffer: simple replay buffer
        """


        ep_reward = 0
        previous_observation, _ = self.val_env.reset()
        if self.obs_normalizer:
            previous_observation = self.obs_normalizer(previous_observation)

        for j in range(self.validating_max_step):

            action = self.predict_single(previous_observation)
            print('action:',action)
            obs, reward, done, _ = self.val_env.step(action)
            ep_reward += reward
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
        np.random.seed(self.config['training']['seed'])
        self.buffer = ReplayBuffer(self.config['training']['buffer_size'])
        self.time_step = 0
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
            ep_actor_loss = 0
            ep_critic_loss = 0
            # keeps sampling until done
            for j in range(self.training_max_step):
                rewards_list = []
                values_list = []
                dones_list = []
                rewards = 0
                for n in range(self.n_step):
                    # with open('test_logdiff.npy','wb') as f:
                    #     np.save('test_logdiff.npy',np.expand_dims(previous_observation, axis=0))
                    # action, logprob = self.actor.predict(np.expand_dims(previous_observation, axis=0))
                    action, logprob, p_outs, d_outs, mu, sigma,x, mw = self.actor.test_predict(np.expand_dims(previous_observation, axis=0))


                    values = self.critic.predict(np.expand_dims(previous_observation, axis=0))

                    print(np.sum(action,axis=1))
                    print('isnan',np.isnan(previous_observation).any())
                    print('mu',mu)
                    print('sigma',sigma)
                    print('mw',mw)
                    print('logprob')
                    print(logprob)
                    print('prob')
                    print(p_outs)
                    print('det')
                    print(d_outs)
                    print('x',x)
                    action = np.squeeze(action,axis=0)
                    logprob = np.squeeze(logprob,axis=0)

                    if n == 0:
                        start_action = action
                        start_logprob = logprob
                        start_obs = previous_observation
                    if n == 1:
                        next_obs = previous_observation

                    # step forward
                    observation, reward, done, _ = self.env.step(action)

                    if self.obs_normalizer:
                        observation = self.obs_normalizer(observation)

                    previous_observation = observation
                    rewards += np.power(self.gamma,n)*reward

                    rewards_list.append(rewards)
                    values_list.append(values)
                    dones_list.append(done)

                values = self.critic.predict(np.expand_dims(observation, axis=0))
                values_list.append(values)

                target_values, adv = self.compute_gae(rewards_list,
                                                      dones_list,
                                                      values_list,
                                                      self.gamma,
                                                      self.lam,
                                                      self.n_step)
                self.time_step += 1
                self.buffer.add(start_obs, (start_action, start_logprob, target_values, adv),
                                            rewards, done, observation)

                previous_observation = next_obs
                ep_reward += rewards
                if self.time_step % self.update_steps == 0:
                    # batch update

                    for k in range(self.epochs):
                        s_batch, alta_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

                        a_batch, l_batch = np.vstack(alta_batch[:,0]), np.vstack(alta_batch[:,1])
                        target_batch, adv_batch = np.vstack(alta_batch[:,2]), np.vstack(alta_batch[:,3])

                        actor_loss, a_g, logprob, _ = self.actor.train(s_batch,adv_batch,l_batch)
                        print('adv_batch',adv_batch)
                        print('l_batch',l_batch)
                        print('new_log',logprob)
                        critic_out, critic_loss, _ = self.critic.train(s_batch,target_batch)
                        print('actor_loss',actor_loss)
                        print('a_g',a_g)
                        summaries = self.sess.run(self.summary_ops, feed_dict = {
                                                self.summary_vars[0] : np.mean(actor_loss),
                                                self.summary_vars[1] : np.mean(critic_loss)
                        })

                        [self.writer.add_summary(summary, ((self.time_step // self.update_steps)*self.epochs + k)
                          ) for summary in summaries]
                        self.writer.flush()

                    self.buffer.clear()

                    ep_actor_loss += np.mean(actor_loss)
                    ep_critic_loss += np.mean(critic_loss)
                    self.validate((self.time_step // self.update_steps),verbose=verbose)
                if done or j == self.training_max_step - 1:

                    print("*"*12+'training'+"*"*12)
                    print('Episode: {:d}, Reward: {:.2f}, actor loss: {:.8f}, critic loss: {:.8f}'.format(
                            i, ep_reward,
                            (ep_actor_loss / float(j+1)),
                            (ep_critic_loss / float(j+1))))
                    reward_summary = self.sess.run(self.summary_ops_r, feed_dict = {
                                                self.summary_vars_r : ep_reward
                    })

                    self.writer.add_summary(reward_summary,i)
                    self.writer.flush()
                    break
        self.save_model(verbose=True)
        print('Finish.')

    def compute_gae(self,rewards, dones, values, gamma, lam, num_step):
        discounted_return = np.empty([num_step])
        gae = 0
        for t in range(num_step-1,-1,-1):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae

            discounted_return[t] = gae + values[t]

        return discounted_return[0], gae

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
        return action
