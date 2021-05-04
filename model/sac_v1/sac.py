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
    def __init__(self, env,val_env sess, actor, critic1, critic2,actor_noise, config,
                 policy_delay = 2, obs_normalizer=None, action_processor=None,
                 model_save_path='weights/sac/sac.ckpt', summary_path='results/sac/'):
        super().__init__(env, val_env, sess, actor, critic1, actor_noise, config, obs_normalizer, action_processor,
                     model_save_path, summary_path)

        self.critic_2 = critic2
        self.policy_delay = policy_delay
        self.saccritics = SACCritics(sess, critic1, critic2)

    def build_summaries(self):

        step_loss_1 = tf.Variable(0.)
        tf.summary.scalar("step target1 loss", step_loss_1)
        step_loss_2 = tf.Variable(0.)
        tf.summary.scalar("step target2 loss", step_loss_2)
        step_qmax = tf.Variable(0.)
        tf.summary.scalar("step Q max", step_qmax)
        alpha_loss = tf.Variable(0.)
        tf.summary.scalar("step alpha loss", alpha_loss)

        summary_vars = [step_loss_1, step_loss_2, step_qmax, alpha_loss]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

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
        self.critic_2.update_target_network()

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
            for j in range(self.config['training']['max_step']):
                # with open('test_logdiff.npy','wb') as f:
                #     np.save('test_logdiff.npy',np.expand_dims(previous_observation, axis=0))
                action, logprob = self.actor.predict(np.expand_dims(previous_observation, axis=0))

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


                # target_q_single_1 = self.critic.predict_target(np.expand_dims(observation, axis=0),
                #                     self.actor.predict_target(np.expand_dims(observation, axis=0)))
                #
                # target_q_single_2 = self.critic_2.predict_target(np.expand_dims(observation, axis=0),
                #                     self.actor.predict_target(np.expand_dims(observation, axis=0)))

                target_q_single = self.saccritics.predict_target(np.expand_dims(observation, axis=0),
                                    *self.actor.predict_target(np.expand_dims(observation, axis=0)))


                # target_q_list = [target_q_single_1,target_q_single_2]
                # min_q_ix = np.argmin(target_q_list)

                if done:
                    y = reward
                else:
                    y = reward + gamma * target_q_single
                # if min_q_ix == 0:
                #     TD_error = self.critic.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                #                                            np.expand_dims(action_take, axis=0),
                #                                            y)[0][0][0]
                # else:
                #     TD_error = self.critic_2.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                #                                            np.expand_dims(action_take, axis=0),
                #                                            y)[0][0][0]

                TD_error = self.saccritics.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                                                       np.expand_dims(action_take, axis=0),
                                                       y)[0][0]
                # add to buffer
                # self.buffer.add(previous_observation, action, reward, done, observation)
                self.buffer.store((previous_observation, action,logprob, reward, done, observation),TD_error)

                if self.buffer.size() >= batch_size:
                    # batch update
                    #s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
                    (s_batch, a_batch,l_batch, r_batch, t_batch, s2_batch), is_weights, indices = self.buffer.select(batch_size)
                    # Calculate targets
                    # target_q_1 = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))
                    # target_q_2 = self.critic_2.predict_target(s2_batch, self.actor.predict_target(s2_batch))
                    # target_qs_list = np.concatenate([target_q_1,target_q_2],axis=1)
                    #
                    # min_qs_ix = np.argmin(target_qs_list, axis=1)

                    target_q = self.saccritics.predict_target(s2_batch, *self.actor.predict_target(s2_batch))
                    y_i = []
                    TD_errors = []

                    for k in range(batch_size):
                        if t_batch[k]:
                            y_tmp = r_batch[k]
                        else:
                            y_tmp = r_batch[k] + gamma * target_q[k,:]
                        # Update the critic given the targets
                        # if min_qs_ix[k] == 0:
                        #     TD_error = self.critic.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                        #                                            np.expand_dims(a_batch[k],axis=0),
                        #                                            np.array([[y_tmp]]))[0][0][0]
                        # else:
                        #     TD_error = self.critic_2.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                        #                                            np.expand_dims(a_batch[k],axis=0),
                        #                                            np.array([[y_tmp]]))[0][0][0]

                        TD_error = self.saccritics.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                                                               np.expand_dims(a_batch[k],axis=0),
                                                               np.array([y_tmp]))[0][0]
                        y_i.append(y_tmp)
                        TD_errors.append(TD_error)

                    TD_errors = np.array(TD_errors)

                    self.buffer.update_priority(indices, TD_errors)

                    bias = TD_errors * np.array(is_weights)  # importance sampling

                    # predicted_q_value_1, step_loss_1, _ = self.critic.train(
                    #                             s_batch, a_batch,
                    #                             np.reshape(y_i, (batch_size, 1)),
                    #                             np.expand_dims(bias, axis=1))
                    #
                    # predicted_q_value_2, step_loss_2, _ = self.critic_2.train(
                    #                             s_batch, a_batch,
                    #                             np.reshape(y_i, (batch_size, 1)),
                    #                             np.expand_dims(bias, axis=1))

                    predicted_q_value, q1_loss, q2_loss, _, _, = self.saccritics.train(
                                                    s_batch, a_batch,l_batch,
                                                    np.reshape(y_i, (batch_size, 1)),
                                                    np.expand_dims(bias, axis=1))


                    # Update the actor policy using the sampled gradient
                    a_outs, l_outs = self.actor.predict(s_batch)
                    grads = self.saccritics.action_logprob_gradients(s_batch, a_outs, l_outs)
                    # print(self.sess.run(self.saccritics.alpha))
                    if j % self.policy_delay == 0:
                        self.actor.train(s_batch, *grads)

                        # Update target networks
                        self.actor.update_target_network()
                    self.critic.update_target_network()
                    self.critic_2.update_target_network()

                    alpha_loss, _ = self.saccritics.train_alpha(l_outs)

                    ep_max_q += np.amax(predicted_q_value)

                    ep_loss_1 += np.mean(q1_loss)
                    ep_loss_2 += np.mean(q2_loss)
                    ep_alpha_loss += np.mean(alpha_loss)

                    summary_step_loss = self.sess.run(self.summary_ops, feed_dict = {
                                            self.summary_vars[0] : np.mean(q1_loss),
                                            self.summary_vars[1] : np.mean(q2_loss),
                                            self.summary_vars[2] : np.amax(predicted_q_value),
                                            self.summary_vars[3] : np.mean(alpha_loss)
                    })

                    writer.add_summary(summary_step_loss, self.config['training']['max_step']*i+j)
                    writer.flush()

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

                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}, loss1: {:.8f}, loss2: {:.8f}, alpha loss: {:.8f}'.format(
                            i, ep_reward, (ep_max_q / float(j+1)),
                             (ep_loss_1 / float(j+1)),
                            (ep_loss_2 / float(j+1)),
                            (ep_alpha_loss / float(j+1))))
                    break

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
        action, logprob = self.actor.predict(np.expand_dims(observation, axis=0))
        action = np.squeeze(action,axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action
