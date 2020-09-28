"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

 
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001      # soft replacement

MEMORY_CAPACITY = 1000
BATCH_SIZE = 64


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]     # zip将对应元素打包成元组

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        ## 添加到tensorboard显示
        tf.summary.scalar('q_vlaue',-a_loss)
        tf.summary.scalar('td_error',td_error) 

        self.ep_reward = tf.placeholder(tf.float32, 1, 'ep_reward')
        tf.summary.scalar('ep_reward',tf.reduce_mean(self.ep_reward))

        self.sess.run(tf.global_variables_initializer())



    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def plot_(self,merged,ep_reward):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        return self.sess.run(merged,{self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.ep_reward: [ep_reward]})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

#In the low-dimensional case, we used batch
#normalization on the state input and all layers of the  network and all layers of the Q network prior
#to the action input

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            bn_layer = tf.layers.batch_normalization(s,name = 's_bn')

            net1 = tf.layers.dense(bn_layer, 400, activation=None, name='l1', trainable=trainable)
            bn_net1 = tf.layers.batch_normalization(net1, name = 'l1_bn')
            act_net1 =tf.nn.relu(bn_net1,name = 'l1_act')

            net2 = tf.layers.dense(act_net1, 300, activation=None, name='l2', trainable=trainable)
            bn_net2 = tf.layers.batch_normalization(net2, name = 'l2_bn')
            act_net2 =tf.nn.relu(bn_net2,name = 'l2_act')

            a = tf.layers.dense(act_net2, self.a_dim, activation=None, name='a', trainable=trainable)
            bn_a = tf.layers.batch_normalization(a, name = 'a_bn')
            act_a =tf.nn.relu(bn_a,name = 'a_act')

            act_a = tf.add(act_a,1)
            return tf.multiply(act_a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            bn_layer = tf.layers.batch_normalization(s,name = 's_bn')

            net1 = tf.layers.dense(bn_layer, 400, activation=None, name='l1', trainable=trainable)
            bn_net1 = tf.layers.batch_normalization(net1, name = 'l1_bn')
            act_net1 =tf.nn.relu(bn_net1,name = 'l1_act')

            n_l2 = 300
            w2_s = tf.get_variable('w2_s', [400, n_l2], trainable=trainable)
            w2_a = tf.get_variable('w2_a', [self.a_dim, n_l2], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l2], trainable=trainable)
            net = tf.nn.relu(tf.matmul(act_net1, w2_s) + tf.matmul(a, w2_a) + b2)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################
if __name__ == "__main__":
    
    # sess = tf.Session()


    env = gym.make(123)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    # writer = tf.summary.FileWriter("logs/", sess.graph)

    count = 0

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()
                count = count + 1
                print("学习   ",count)
                
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300:RENDER = True
                break
    print('Running time: ', time.time() - t1)