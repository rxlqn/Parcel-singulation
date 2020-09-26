# import AC_CartPole as ac 
import DDPG as ddpg
import simulation as sim 
import numpy as np
import tensorflow as tf
import gym
import time

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


# Superparameters
MAX_EPISODES = 300
MAX_EP_STEPS = 1000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = False

N_F = 6         # # of features
N_A = 17        # # of actions      17个传送带  10个速度

env = sim.Environment()

if __name__ == "__main__":

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    s_dim = N_F
    a_dim = N_A
    a_bound = 50

    ddpg = ddpg.DDPG(a_dim, s_dim, a_bound)
    merged = tf.summary.merge_all()


    if OUTPUT_GRAPH:
        writer = tf.summary.FileWriter("logs/", ddpg.sess.graph)
        # writer.add_summary


    var = 30  # control exploration

    s = env.reset()

    count = 0

    t1 = time.time()
    for i in range(MAX_EPISODES):
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            # if RENDER:
            #     env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.round(np.random.normal(a, var)), 5, 100)    # add randomness to action selection for exploration
            s_, r, done  = env.step(a)

            ddpg.store_transition(s, a, r , s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()
                count = count + 1
                # print("学习   ",count)

            s = s_
            ep_reward += r


            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, count)
                # if ep_reward > -300:RENDER = True
                # break

            if done:
                s = env.reset()
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, count)
                break   

    print('Running time: ', time.time() - t1)
    # 超过max_episode
    saver = tf.train.Saver()
    ckpt_path = './ckpt/test-model.ckpt'
    save_path = saver.save(ddpg.sess, ckpt_path)