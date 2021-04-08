import sys
import math
import random
import numpy as np

from . import binary_heap

class BetaSchedule(object):
    def __init__(self, conf=None):
        self.batch_size = int(conf['batch_size'] if 'batch_size' in conf else 32)

        self.beta_zero = conf['beta_zero'] if 'beta_zero' in conf else 0.5
        self.learn_start = int(conf['learn_start'] if 'learn_start' in conf else 1000)
        # http://www.evernote.com/l/ACnDUVK3ShVEO7fDm38joUGNhDik3fFaB5o/
        self.total_steps = int(conf['total_steps'] if 'total_steps' in conf else 100000)
        self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)

    def get_beta(self, global_step):
        # beta, increase by global_step, max 1
        beta = min(self.beta_zero + (global_step - self.learn_start - 1) * self.beta_grad, 1)
        return beta, self.batch_size


class Experience(object):

    def __init__(self, buffer_size, alpha=0.7):
        print("Initializing rank_based.Experience()")

        # self.replace_flag = conf['replace_old'] if 'replace_old' in conf else True
        self.buffer_size = buffer_size
        self.alpha = alpha

        self.index = 0
        self.record_size = 0
        self.isFull = False

        self._experience = {}
        self.priority_queue = binary_heap.BinaryHeap(self.buffer_size)


    def size(self):
        return self.record_size

    def build_distribution(self):
        # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        pdf = list(
            map(lambda x: math.pow(x, -self.alpha), range(1, self.size() + 1))
        )
        pdf_sum = math.fsum(pdf)
        self.power_law_distribution = list(map(lambda x: x / pdf_sum, pdf))


    def save(self, filename):
        data = np.array([
            self.size,
            self.replace_flag,
            self.alpha,
            self.beta_sched.beta_zero,
            self.beta_sched.batch_size,
            self.beta_sched.learn_start,
            self.beta_sched.total_steps,
            self.index,
            self.record_size,
            self.isFull,
            self._experience,
            self.priority_queue.priority_queue,
            self.priority_queue.p2e,
            self.priority_queue.e2p,
            self.priority_queue.size
        ])
        np.save(filename, data)

    def load(self, filename):
        data = np.load(filename)
        self.size, \
        self.replace_flag, \
        self.alpha, \
        self.beta_sched.beta_zero, \
        self.beta_sched.batch_size, \
        self.beta_sched.learn_start, \
        self.beta_sched.total_steps, \
        self.index, \
        self.record_size, \
        self.isFull, \
        self._experience, \
        self.priority_queue.priority_queue, \
        self.priority_queue.p2e,\
        self.priority_queue.e2p, \
        self.priority_queue.size = data
        self.priority_queue.balance_tree()



    def store(self, experience, priority):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        insert_index = self.index
        if self.record_size < self.buffer_size:
            self.index += 1
            self.record_size += 1
        else:
            insert_index = self.priority_queue.get_e_id()[np.argmin(np.array(self.priority_queue.get_priority()))]
        if insert_index in self._experience:
            del self._experience[insert_index]
        self._experience[insert_index] = experience
        # add to priority queue
        # priority = self.priority_queue.get_max_priority()
        self.priority_queue.update(priority, insert_index)

        return True


    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        return [self._experience[v] for v in indices]

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        self.priority_queue.balance_tree()

    def update_priority(self, indices, delta):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """
        for i in range(0, len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    def sample(self, global_step, batch_size=None):
        beta, batch_size = self.beta_sched.get_beta(global_step)
        return self.select(beta, batch_size=batch_size)

    def select(self, batch_size, beta=0.7):
        """
        sample a mini batch from experience replay
        :param beta
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        self.build_distribution()
        distribution = self.power_law_distribution
        rank_list = []
        # sample from k segments
        for n in range(batch_size):
            index = random.randint(1, self.priority_queue.size)
            rank_list.append(index)

        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution[v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(alpha_pow) * self.size(), -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id
        rank_e_id = 0
        try:
            rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        except:
            print('a')
        # get experience id according rank_e_id
        experience = self.retrieve(rank_e_id)
        transition = (np.array([_[i] for _ in experience]) for i in range(len(experience[0]))) # s a r s' t

        return transition, w, rank_e_id
