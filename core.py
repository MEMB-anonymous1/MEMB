import numpy as np
import tensorflow as tf

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


LOG_STD_MAX = 2
LOG_STD_MIN = -20

"""
The basic structure of stochastic policy agent
"""

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    # create value function
    value_function_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = value_function_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = value_function_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = value_function_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = value_function_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('v'):
        v = value_function_mlp(x)

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v


"""
Reward and transition dynamic model construction, the hidden_size for each model could
be adjusted in each subnetworks.
"""
def reward_dynamic_model(x, a, pi, hidden_sizes=(400,300), activation=tf.nn.relu, action_space=None):
    value_function_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('dm'):
        transition= dynamic_model(x,a)
    with tf.variable_scope('dm',reuse=True):
        transition_pi = dynamic_model(x,pi)
    with tf.variable_scope('rm'):
        r_rm = reward_model(x,a)
    with tf.variable_scope('rm',reuse=True):
        r_rm_pi = reward_model(x,pi)
    with tf.variable_scope('v', reuse=True):
        v_prime = value_function_mlp(transition_pi)
    return transition, r_rm, transition_pi ,r_rm_pi, v_prime


def dynamic_model(x, a, hidden_sizes=(256,128), activation=tf.nn.relu, output_activation=None):
    state_dim = x.shape.as_list()[-1]
    x = tf.concat([x,a],-1)

    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
    mu = tf.layers.dense(x, units=state_dim, activation=output_activation)
    log_std = tf.layers.dense(x, state_dim, activation=tf.tanh)

    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)

    transition = mu + tf.random_normal(tf.shape(mu)) * std
    return transition


def reward_model(x, a, hidden_sizes=(2*128,128), activation=tf.nn.relu, output_activation=None):
    x = tf.concat([x,a],-1)

    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)

    mu = tf.layers.dense(x, units=1, activation=output_activation)
    log_std = tf.layers.dense(x, 1, activation=tf.tanh)

    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)

    reward = mu + tf.random_normal(tf.shape(mu)) * std
    return reward
