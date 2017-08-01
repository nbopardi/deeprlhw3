# test_dqn_atari.py
import sys
import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *

# def atari_model(img_in, num_actions, scope, reuse=False):
#     # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
#     with tf.variable_scope(scope, reuse=reuse):
#         out = img_in
#         with tf.variable_scope("convnet"):
#             # original architecture
#             out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
#             out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
#             out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
#         out = layers.flatten(out)
#         with tf.variable_scope("action_value"):
#             out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
#             out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

#         return out

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def test_model(model_name, env, replay_buffer_size = 1000000, frame_history_len = 4):
    with tf.Session(graph = tf.Graph()) as sess:
        graph_name = model_name + '.meta'
        saver = tf.train.import_meta_graph(graph_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        meta_graph = tf.get_default_graph()

        # for op in meta_graph.get_operations():
        #     print str(op.name)
        if len(env.observation_space.shape) == 1:
            # This means we are running on low-dimensional observations (e.g. RAM)
            input_shape = env.observation_space.shape
        else:
            img_h, img_w, img_c = env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
        num_actions = env.action_space.n

        # set up placeholders
        # placeholder for current observation (or state)
        obs_t_ph              = meta_graph.get_tensor_by_name('obs_t_ph:0')    # placeholder for current action
        act_t_ph              = meta_graph.get_tensor_by_name('act_t_ph:0')
        # placeholder for current rewar
        rew_t_ph              = meta_graph.get_tensor_by_name('rew_t_ph:0')
        # placeholder for next observation (or state)
        obs_tp1_ph            = meta_graph.get_tensor_by_name('obs_tp1_ph:0')
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        done_mask_ph          = meta_graph.get_tensor_by_name('done_mask_ph:0')

        # casting to float on GPU ensures lower data transfer times.
        obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
        obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0
        
        current_qfunc = meta_graph.get_tensor_by_name('current_qfunc/current_q_func_op:0')
        # current_qfunc = meta_graph.get_tensor_by_name("current_qfunc:0")
        # current_qfunc = q_func(obs_t_float, num_actions, scope = "q_func", reuse = False)
        # action_predict = tf.argmax(current_qfunc, axis = 1)
        # action_predict = meta_graph.get_tensor_by_name('action_predict:0')
        action_predict = tf.argmax(current_qfunc, axis = 1, name = 'action_predict')
        # def predict_action(observation):

        #     return sess.run(action_predict, feed_dict = {obs_t_float: observation})[0]


        # tf.global_variables_initializer()
        last_obs = env.reset()
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

        done = False

        mean_episode_reward      = -float('nan')
        best_mean_episode_reward = -float('inf')
        iteration = 0
        while not done:

            obs_idx = replay_buffer.store_frame(last_obs)
            replay_obs = replay_buffer.encode_recent_observation()
            # print replay_obs.shape
            replay_obs = replay_obs.reshape(1,replay_obs.shape[0],replay_obs.shape[1],replay_obs.shape[2])
            # print replay_obs.shape
            # action = predict_action(replay_obs)
            action = sess.run([action_predict], feed_dict = {obs_t_ph: replay_obs})[0]
            last_obs, reward, done, info = env.step(action)
            
            env.render()
            replay_buffer.store_effect(obs_idx, action, reward, done)

            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
            if done:
                print("Iteration %d" % (iteration,))
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("best mean reward %f" % best_mean_episode_reward)
                print("episodes %d" % len(episode_rewards))
                # print("exploration %f" % exploration.value(t))
                # print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
                sys.stdout.flush()

            iteration += 1


def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = np.random.randint(50) # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    # session = get_session()
    # atari_learn(env, session, num_timesteps=task.max_timesteps)
    test_model('GAME_model-215', env)

if __name__ == "__main__":
    main()
