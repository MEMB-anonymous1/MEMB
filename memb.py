import numpy as np
import tensorflow as tf
import gym
from gym.envs import mujoco
import time
import core
from core import get_vars


class ReplayBuffer:
    """
    The replay buffer used to uniformly sample the data
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""
Model Embedding Model Based Algorithm
(with TD3 style Q value function update)
"""



def memb(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, model_lr=3e-4, value_lr=1e-3, pi_lr=3e-4, alpha=0.4,
        batch_size=100, start_steps=1000,max_ep_len=1000, save_freq=1,
        train_model_epoch=1, test_freq=10, exp_name='',env_name='',save_epoch=100):


    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]
    ac_kwargs['action_space'] = env.action_space

    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
        transition , r_rm, transition_pi ,r_rm_pi, v_prime = core.reward_dynamic_model(x_ph, a_ph, pi, **ac_kwargs)

    # Target value network for updates
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _,v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # TD3 style Q function updates

    min_q_pi = tf.minimum(q1_pi, q2_pi)

    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    r_backup = r_ph
    transition_backup = x2_ph

    r_loss = 0.5 * tf.reduce_mean((r_backup-r_rm)**2)
    transition_loss = 0.5 * tf.reduce_mean((transition_backup - transition)**2)
    model_loss = r_loss+transition_loss

    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss

    pi_loss = r_rm_pi - alpha*logp_pi + gamma*(1-d_ph)*v_prime


    # model train op
    model_optimizer = tf.train.AdamOptimizer(learning_rate=model_lr)
    model_params = get_vars('main/dm') + get_vars('main/rm')
    train_model_op = model_optimizer.minimize(model_loss, var_list=model_params)

    # policy train op
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    with tf.control_dependencies([train_model_op]):
        train_pi_op = pi_optimizer.minimize(-pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    value_optimizer = tf.train.AdamOptimizer(learning_rate=value_lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                train_pi_op, train_value_op, target_update]

    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    saver = tf.compat.v1.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)


    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent(epoch,n=1):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        total_reward = 0
        for j in range(n): # repeat n times
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            total_reward += ep_ret
        print('The '+str(epoch)+' epoch is finished!')
        print('The test reward is '+str(total_reward/n))
        return total_reward/n

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    reward_recorder = []


    for t in range(total_steps):
        """
        The algorithm would take total_steps totally in the training
        """

        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len==max_ep_len else d

        replay_buffer.store(o, a, r, o2, d)

        o = o2

        if t // steps_per_epoch > train_model_epoch:
            # train 5 steps of Q, V, and pi.
            # train 1 step of model
            for j in range(5):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                            x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']}
                _ = sess.run(step_ops, feed_dict)
            outs = sess.run(train_model_op, feed_dict)
        else:
            # pretrain the model
            batch = replay_buffer.sample_batch(batch_size)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done'],
                         }
            outs = sess.run(train_model_op, feed_dict)

        if d or (ep_len == max_ep_len):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            if epoch > train_model_epoch and epoch % test_freq == 0:
                # test the agent when we reach the test_freq, save the experiment result
                reward_test = test_agent(epoch)
                reward_recorder.append(reward_test)
                reward_nparray = np.asarray(reward_recorder)
                np.save(str(exp_name)+'_'+str(env_name)+'_'+str(save_freq)+'.npy',reward_nparray)

            if epoch % save_epoch == 0:
                # save the model
                saver.save(sess, str(exp_name)+'_'+str(env_name),global_step=epoch)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--exp_name', type=str, default='memb')
    parser.add_argument('--train_model_epoch', type=int, default=1)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--save_epoch', type=int, default=500)
    args = parser.parse_args()

    for i in range(0,5):
        # repeat 5 times of experiment
        tf.reset_default_graph()
        memb(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),gamma=args.gamma,
        seed=args.seed, epochs=args.epochs, save_freq=i,train_model_epoch=args.train_model_epoch,
        test_freq=args.test_freq, exp_name=args.exp_name, env_name=args.env_name,
        save_epoch=args.save_epoch)
