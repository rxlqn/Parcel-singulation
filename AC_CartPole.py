"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False             # 6006打不开要切换端口8008
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 5000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]        #4
N_A = env.action_space.n                    #2




class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")       # dim 1*n_features
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        self.acts_prob = []
        self.log_prob = []
        self.exp_v = []
        self.train_op = []

        for i in range(0,17):       #17个传送带分别建模 input state output action prob
            index = i
            self.actor_model(index, n_actions, lr=0.001)

    def actor_model(self,index, n_actions, lr=0.001):
        with tf.variable_scope('Actor'+str(index)):            # 拟合策略函数 input state,output act_prob
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'+str(index)
            )

            acts_prob = tf.layers.dense(   #全连接层
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'+str(index)
            )
            self.acts_prob.append(acts_prob)

        with tf.variable_scope('exp_v'+str(index)):                    # 目标最大化回报函数
            acts_prob_tmp = self.acts_prob[index]
            #  ??                                                                # 当前state take action对应的概率，增加正激励概率，减小负激励概率
            log_prob = tf.log(tf.clip_by_value(acts_prob_tmp[0,self.a[index]],1e-8,1.0))    # 防止出现nan
            self.log_prob.append(log_prob)
            # log_prob = tf.log(self.acts_prob)           # ? 输入action  
            exp_v = tf.reduce_mean(log_prob * self.td_error)
            self.exp_v.append(exp_v)  # advantage (TD_error) guided loss 步长

            ## 添加到tensorboard显示
            tf.summary.scalar('exp_v'+str(index),exp_v)

        with tf.variable_scope('train'+str(index)):
            self.train_op.append(tf.train.AdamOptimizer(lr).minimize(-self.exp_v[index]))  # minimize(-exp_v) = maximize(exp_v)  

    def learn(self, s, a, td):              # 反传？
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}

        for i in range(0,17):       #17个传送带分别建模 input state output action prob
            _, exp_v= self.sess.run([self.train_op[i], self.exp_v[i]], feed_dict)        #fetches feed_dict


    def choose_action(self, s):
        s = s[np.newaxis, :]                # 增加一个维度
        speed_lvl = []

        feed_dict = {self.s: s}
        for i in range(0,17):       #17个传送带分别建模 input state output action prob
            probs = self.sess.run(self.acts_prob[i], feed_dict)        #fetches feed_dict
            speed_lvl.append(np.random.choice(np.arange(5), p=probs[0]))

        return  speed_lvl           # return a int  按照概率P随机选择


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):  #input state output state_value
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v           # TDerror近似advantage function
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval      神经网络拟合advantage function，最小化loss
            self.loss = tf.reduce_mean(self.loss)
            tf.summary.scalar('loss',self.loss)

        
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _= self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error,v_

if __name__ == "__main__":

    sess = tf.Session()

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER: env.render()

            a = actor.choose_action(s)

            s_, r, done, info = env.step(a)

            if done: r = -20

            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break

