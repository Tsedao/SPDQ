from ..d3pg.d3pg import D3PG

from ..core.replay.replay_buffer import ReplayBuffer
from ..core.replay import proportional, rank_based
from ..base_model import BaseModel

import numpy as np
import tensorflow as tf
import os

class LowerController():
    def __init__(self, high_con, c):
        self.high_con = high_con
        self.c = c

    def step(self, env, previous_observation, subgoal, action,step):
        # step forward
        observation, reward_m, done, _ = env.step(action)

        pre_weights = np.squeeze(observation[:,-2:-1,-1],axis=1)
        curr_weights = np.squeeze(observation[:,-1:,-1],axis=1)
        # reward_w = -np.sqrt(np.sum((pre_weights+subgoal-curr_weights)**2))        # lower level reward
        # subgoal = np.exp(subgoal) / np.sum(np.exp(subgoal))
        print("subgoal",subgoal)
        print("curr_weights",curr_weights)
        # reward_w = -np.sqrt(np.sum((subgoal-curr_weights)**2))        # lower level reward
        reward_w = (np.dot(subgoal,curr_weights) / (np.linalg.norm(subgoal,ord=2)*np.linalg.norm(curr_weights,ord=2)))[0]
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)

        # next_subgoal = self._choose_subgoal(step,previous_observation, subgoal, observation)
        next_subgoal = subgoal
        # TODO: to check wether intrinsic reward equals to td error
        return observation, next_subgoal, reward_w, done, _

    def _choose_subgoal(self, step, previous_observation, s_g, obs):
        # if step % self.c == 0:
        #     s_g = np.expand_dims(self.high_con.predict_single(previous_observation),axis=0)
        # else:
        s_g = np.expand_dims(self._subgoal_transition(previous_observation, s_g, obs),axis=0)
        return s_g

    def _subgoal_transition(self, previous_observation, s_g, observation):
        return np.squeeze(observation[:,-2:-1,-1],axis=1) + np.squeeze(s_g,axis=0) - np.squeeze(observation[:,-1:,-1],axis=1)


class LowerControllerD3PG(LowerController,D3PG):
    def __init__(self,
                 env,
                 val_env,
                 sess,
                 high_con,
                 c,
                 actor,
                 critic,
                 actor_noise,
                 config,
                 policy_delay = 1,
                 obs_normalizer=False,
                 action_processor=None,
                 model_save_path='weights/hiro/lower_d3pg.ckpt',
                 best_model_save_path='weights/best_hiro/lower_d3pg.ckpt',
                 summary_path='results/hiro/'):
        LowerController.__init__(self, high_con=high_con,c=c)
        D3PG.__init__(self, env, val_env, sess, actor, critic, actor_noise, config,
                        policy_delay,obs_normalizer, action_processor,
                        model_save_path, best_model_save_path, summary_path)


    def train_one_step(self,previous_observation,s_g, ep_reward,ep_max_q, ep_loss,
                       epi_counter, step_counter):

        rewards = 0
        TD_errors = 0

        # n-step TD-learning
        for n in range(1):                                                       # no n-step TD-learing
            if len(s_g.shape) < 2:
                s_g = np.expand_dims(s_g, axis=0)
            action = self.actor.predict(np.expand_dims(previous_observation, axis=0),s_g).squeeze(
                axis=0) + self.actor_noise()

            if n == 0:
                start_action = action
                start_obs = previous_observation

            # step forward
            observation, n_sg, reward, done, _ = self.step(self.env,previous_observation, s_g, action,step_counter)


            target_q_single = self.critic.predict_target(np.expand_dims(observation, axis=0),
                                self.actor.predict_target(np.expand_dims(observation, axis=0),n_sg) +
                                np.expand_dims(self.actor_noise(),axis=0))



            if done:
                y = reward*np.ones(shape=(1,self.num_quart))
            else:
                y = reward + self.gamma * target_q_single


            TD_error = self.critic.compute_TDerror(np.expand_dims(previous_observation,axis=0),
                                                   np.expand_dims(action, axis=0),
                                                   y)[0]
            previous_observation = observation
            s_g = n_sg
            rewards += np.power(self.gamma,n)*reward
            print('n_sg:',n_sg)
            print('is_nan',np.isnan(observation).any())
            TD_errors += TD_error
            n_sg = np.squeeze(n_sg,axis=0)
        # add to buffer
        # self.buffer.add(previous_observation, action, reward, done, observation)
        if self.buffer.size() >= self.batch_size:
            # batch update
            #s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(batch_size)
            (s_batch, sg_batch, a_batch, r_batch, t_batch, s2_batch,sg2_batch), is_weights, indices = self.buffer.select(self.batch_size)

            noise = np.vstack([self.actor_noise() for i in range(self.batch_size)])
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch,sg2_batch)+noise)
            y_i = []
            TD_errors_list = []

            for k in range(self.batch_size):
                if t_batch[k]:
                    y_tmp = r_batch[k] * np.ones(self.num_quart)
                else:
                    y_tmp = r_batch[k] + self.gamma * target_q[k,...]
                # Update the critic given the targets


                TD_error = self.critic.compute_TDerror(np.expand_dims(s_batch[k],axis=0),
                                                       np.expand_dims(a_batch[k],axis=0),
                                                       np.array([y_tmp]))[0]
                y_i.append(y_tmp)
                TD_errors_list.append(TD_error)

            TD_errors_batch = np.array(TD_errors_list)

            self.buffer.update_priority(indices, TD_errors_batch)

            bias = TD_errors_batch * np.expand_dims(np.array(is_weights),axis=1)  # importance sampling


            predicted_q_value, q_loss,  _  = self.critic.train(
                                            s_batch, a_batch,
                                            np.reshape(y_i, (self.batch_size, self.critic.num_quart)),
                                            bias)


            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch,sg_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            if step_counter % self.policy_delay == 0:
                self.actor.train(s_batch,sg_batch, grads)

                # Update target networks
                self.actor.update_target_network()
            self.critic.update_target_network()


            ep_max_q += np.amax(np.mean(predicted_q_value,axis=1))

            ep_loss += np.mean(q_loss)


            summaries = self.sess.run(self.summary_ops, feed_dict = {
                                    self.summary_vars[0] : np.mean(q_loss),
                                    self.summary_vars[1] : np.amax(np.mean(predicted_q_value,axis=1)),
            })

            [self.writer.add_summary(summary, self.training_max_step*epi_counter+self.n_step*step_counter
                                    ) for summary in summaries]
            self.writer.flush()

        return start_obs, n_sg, start_action, rewards, observation, done, TD_errors, ep_reward, ep_max_q, ep_loss

    def predict_single(self, observation, subgoal):

        if self.obs_normalizer:
            observation = self.obs_normalizer(observation)
        action = self.actor.predict_target(np.expand_dims(observation, axis=0),
                                     np.expand_dims(subgoal,axis=0))
        action = np.squeeze(action,axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

class HIRO(BaseModel):
    def __init__(self,
                 low_con,
                 high_con,
                 obs_normalizer,
                 use_sac,
                 model_save_path='weights/hiro/hiro.ckpt',
                 best_model_save_path='weights/best_hiro/hiro.ckpt',
                 summary_path='results/hiro/'):

        self.low_con = low_con
        self.high_con = high_con
        self.obs_normalizer = obs_normalizer
        self.num_episode = self.low_con.num_episode
        self.batch_size = self.low_con.batch_size
        self.training_max_step = self.low_con.training_max_step
        self.training_max_step_size = self.low_con.training_max_step_size
        self.validating_max_step = self.low_con.validating_max_step
        self.use_sac = use_sac
        self.c = self.low_con.c
        self.sess = self.low_con.sess
        self.model_save_path = model_save_path
        self.best_model_save_path = best_model_save_path
        self.summary_path = summary_path
        self.writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        self.low_con.writer = self.writer
        self.high_con.writer = self.writer
        self.reward_summary_dict = {"epi":[],"reward_avg":[],"reward_std":[],"reward_sum":[]}


    def validate(self,epi_counter):
        previous_observation, _ = self.low_con.val_env.reset()
        ep_reward = 0
        val_ep_reward_list = []
        for j in range(self.validating_max_step):
            action = self.predict_single(previous_observation,j)
            print('action:',action)
            obs, reward, done, _ = self.low_con.val_env.step(action)
            ep_reward += reward
            previous_observation = obs
            if  j == self.validating_max_step - 1 or done:
                print("*"*12+'validaing'+"*"*12)
                print('Episode: {:d}, Reward: {:.2f}'.format(
                            epi_counter, ep_reward))
                reward_summary = self.sess.run(self.low_con.summary_ops_val_r, feed_dict = {
                                            self.low_con.summary_vars_val_r : ep_reward
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




    def train(self,verbose=True):
        self.low_con.actor.update_target_network()
        self.low_con.critic.update_target_network()

        self.high_con.actor.update_target_network()
        self.high_con.critic.update_target_network()
        self.best_val_reward = 0
        for i in range(self.num_episode):

            if verbose:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.low_con.buffer.count()))

            ep_reward_m = 0
            ep_max_q_m = 0
            ep_loss_m = 0
            ep_alpha_loss_m = 0
            self.high_con.ep_reward_list = []

            ep_reward_w = 0
            ep_max_q_w = 0
            ep_loss_w = 0
            self.low_con.ep_reward_list = []

            previous_observation, _ = self.low_con.env.reset()
            self.high_con.env.src.start_date = self.low_con.env.src._start_date
            _, _ = self.high_con.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation)

            for j in range(0,self.training_max_step,self.training_max_step_size):

                if j % self.c == 0:
                    print("***training high level controller***")
                    if not self.use_sac:
                        start_obs_m, start_goal, rewards_m, obs_m, done_m, TD_errors_m, ep_reward_m, ep_max_q_m ,ep_loss_m = self.high_con.train_one_step(
                                                                 previous_observation, ep_reward_m,ep_max_q_m, ep_loss_m,i, j)
                    else:
                        start_obs_m, start_goal, start_prob, rewards_m, obs_m, done_m, TD_errors_m, ep_reward_m, ep_max_q_m ,ep_loss_m, ep_alpha_loss_m = self.high_con.train_one_step(
                                                                    previous_observation,ep_reward_m,ep_max_q_m, ep_loss_m,ep_alpha_loss_m,i,j)
                    ep_reward_m += rewards_m
                # start_goal = self.subgoal_processor(start_goal)
                start_obs_w, n_sg, start_action, rewards_w,obs_w, done_w, TD_errors_w, ep_reward_w, ep_max_q_w ,ep_loss_w = self.low_con.train_one_step(
                previous_observation,start_goal, ep_reward_w,ep_max_q_w, ep_loss_w,i, j)


                ep_reward_w += rewards_w
                print('reward_m:',rewards_m)
                print('reward_w:',rewards_w)
                print('TD_error_w:',TD_errors_w)
                if j % self.c == 0:
                    if not self.use_sac:
                    # d3pg higher controller
                        self.high_con.buffer.store((start_obs_m, start_goal, rewards_m, done_m, obs_w),TD_errors_m)
                    else:
                    # sac higher controller
                        self.high_con.buffer.store((start_obs_m, start_goal, start_prob, rewards_m, done_m, obs_w),TD_errors_m)
                self.low_con.buffer.store((start_obs_w, start_goal, start_action, rewards_w,
                                           done_w, obs_w,n_sg),TD_errors_w)

                previous_observation = obs_w
                start_goal = n_sg

                if (done_w and done_m) or j == self.training_max_step - self.training_max_step_size:

                    print('Episode: {:d}, Reward_m: {:.2f}, Qmax_m: {:.4f}, loss_m: {:.8f}'.format(
                            i, ep_reward_m, (ep_max_q_m / float(j+1)),
                             (ep_loss_m / float(j+1))))
                    print('Episode: {:d}, Reward_w: {:.2f}, Qmax_w: {:.4f}, loss_w: {:.8f}'.format(
                            i, ep_reward_w, (ep_max_q_w / float(j+1)),
                             (ep_loss_w / float(j+1))))
                    reward_summary_w, reward_summary_m = self.sess.run([self.low_con.summary_ops_r,self.high_con.summary_ops_r],
                                                feed_dict = {self.low_con.summary_vars_r : ep_reward_w,
                                                            self.high_con.summary_vars_r: ep_reward_m

                    })
                    self.writer.add_summary(reward_summary_w,i)
                    self.writer.add_summary(reward_summary_m,i)
                    self.writer.flush()
                    break

            self.validate(i)

        self.save_model(verbose)

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

    def save_model(self, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)

    def predict_single(self,observation,step):
        if step % self.c == 0:
            self.val_subgoal = self.high_con.predict_single(observation)
        action = self.low_con.predict_single(observation,self.val_subgoal)
        return action

    def subgoal_processor(self, subgoal):
        sell_idx, buy_idx = np.where(subgoal<0)[0], np.where(subgoal>=0)[0]
        if len(sell_idx)==0 or len(buy_idx) == 0:
            subgoal = np.random.randn(*subgoal.shape)
            sell_idx, buy_idx = np.where(subgoal<0)[0], np.where(subgoal>=0)[0]
        buy_prop = (subgoal[buy_idx]) / np.sum((subgoal[buy_idx]))
        sell_prop = (-subgoal[sell_idx]) / np.sum((-subgoal[sell_idx]))
        subgoal[sell_idx] = -sell_prop
        subgoal[buy_idx] = buy_prop
        np.testing.assert_almost_equal(
            np.sum(subgoal),
            0,
            decimal=5
        )
        return subgoal

    def off_policy_correction(self, low_con, previous_observation, observation, subgoal,state_list, action_list):
        original_goal = np.expand_dims(subgoal,axis=0)
        start_weights = previous_observation[:,-1,-1]
        end_weights = observation[:,-1,-1]
        diff_goal = start_weights - end_weights
        diff_goal = np.expand_dims(diff_goal, axis=0)
        random_goals = np.random.normal(loc=diff_goal, scale=.5,
                                        size=(8, original_goal.shape[-1]))
        cand_goals = np.concatenate([original_goal, diff_goal, random_goals],axis=0)


    def _choose_action(self, s, sg):
        return self.low_con.policy(s, sg)

    def _choose_subgoal(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)
    def subgoal_transition(self, s, sg, n_s):
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        abs_s = s[:sg.shape[0]] + sg
        return -np.sqrt(np.sum((abs_s - n_s[:sg.shape[0]])**2))
