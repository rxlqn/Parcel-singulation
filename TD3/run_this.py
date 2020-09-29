# import AC_CartPole as ac 
import TD3 as td3
import simulation as sim 
import numpy as np
import tensorflow as tf
import gym
import time

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


# Superparameters
MAX_EPISODES = 500
MAX_EP_STEPS = 1000

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

RENDER = False
OUTPUT_GRAPH = True

N_F = 6         # # of features
N_A = 17        # # of actions      17个传送带  10个速度

env = sim.Environment()

if __name__ == "__main__":

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    s_dim = N_F
    a_dim = N_A
    a_bound = 50

    td3 = td3.TD3(a_dim, s_dim, a_bound)
    merged = tf.summary.merge_all()


    if OUTPUT_GRAPH:
        writer = tf.summary.FileWriter("logs/", td3.sess.graph)
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
            a = td3.choose_action(s)
            a = np.clip(np.round(np.random.normal(a, var)), 5, 100)    # add randomness to action selection for exploration
            s_, r, done  = env.step(a)

            td3.store_transition(s, a, r , s_)

            if td3.pointer > MEMORY_CAPACITY:
                # var *= .9999    # decay the action randomness
                td3.learn()
                # print("学习   ",count)

            ep_reward += r


            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, count,"td3_pointer",td3.pointer)
                result = td3.plot_(merged,ep_reward/j)
                writer.add_summary(result, count)
                count = count + 1

            s = s_

            if done:
                s = env.reset()
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, count,"td3_pointer",td3.pointer)
                break   

    print('Running time: ', time.time() - t1)
    # 超过max_episode
    saver = tf.train.Saver()
    ckpt_path = './ckpt/test-model.ckpt'
    save_path = saver.save(td3.sess, ckpt_path)