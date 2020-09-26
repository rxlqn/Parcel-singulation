import AC_CartPole as ac 
import simulation as sim 
import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


# Superparameters
OUTPUT_GRAPH = True             # 6006打不开要切换端口8008
MAX_EPISODE = 500
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.002     # learning rate for critic


N_F = 6         # # of features
N_A = 10        # # of actions      17个传送带  10个速度

env = sim.Environment()

if __name__ == "__main__":

    sess = tf.Session()

    actor = ac.Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = ac.Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    merged = tf.summary.merge_all()


    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        # writer.add_summary

    s = env.reset()
    # init state 

    for i_episode in range(MAX_EPISODE):

        t = 0
        track_r = []
        while True:
            # if RENDER: env.render()

            a = actor.choose_action(s)

            s_, r, done = env.step(a)

            if done:            # 两个包裹都停下
                r = -5000
                s = env.reset()

            track_r.append(r)

            td_error,v_ = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

            if t%500 == 0:
                feed_dict = {actor.s: s[np.newaxis, :], actor.a: a, actor.td_error: td_error,critic.s: s[np.newaxis, :],critic.r: r,critic.v_: v_}
                result = sess.run(merged,feed_dict)

 

                writer.add_summary(result, t+i_episode*MAX_EP_STEPS)

            # update
            s = s_
            t += 1


            if done or t >= MAX_EP_STEPS:

                ep_rs_sum = sum(track_r)
                # print("episode:", i_episode, "  reward:", int(ep_rs_sum/MAX_EP_STEPS))
                

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break
    # 超过max_episode
    saver = tf.train.Saver()
    ckpt_path = './ckpt/test-model.ckpt'
    save_path = saver.save(sess, ckpt_path)