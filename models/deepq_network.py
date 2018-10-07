import tensorflow as tf
import numpy as np

from models import neural_network
from utils.nn_util import replay_buffer
from utils.tf_util import tf_op

class DQN():
    def __init__(self, sess, value_network, target_network, learning_rate,
            epsilon_init, epsilon_decay, epsilon_min, gamma, target_update_threshold):
        self.learning_rate = learning_rate
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.sess = sess
        self.state_dim = value_network.input_dimension
        self.action_dim = value_network.output_dimension
        self.step = 0
        self.target_update_threshold = target_update_threshold

        self.value_network = value_network
        self.target_network = target_network

        self.__buil_network()

    def __buil_network(self):
        self.state = self.value_network.input_tensor
        self.action = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.reward = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.next_state = self.target_network.input_tensor
        self.is_done = tf.placeholder(tf.float32, [None, ], name='s_')  # input Next State

        with tf.variable_scope('target_value'):
            target_tensor = self.reward + self.gamma * (1 - self.is_done) * tf.reduce_max(self.target_network.output_tensor, axis=1)
            target_tensor = tf.stop_gradient(target_tensor)
            #print(self.reward.shape.as_list())
        with tf.variable_scope('state_value'):
            #value_tensor = tf.reduce_sum(self.value_network.output_tensor * self.action)
            batch_size = tf.shape(self.action)[0]
            a_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), self.action], axis=1)
            value_tensor = tf.gather_nd(params=self.value_network.output_tensor, indices=a_indices)    # shape=(None, )
            tf_op.variable_summaries(value_tensor, 'select_state_q_value')

            assert_op = tf.assert_rank(self.value_network.output_tensor, 2)
            with tf.control_dependencies([assert_op]):
                for i in range(self.action_dim):
                    out = tf.slice(self.value_network.output_tensor, [0, i], [batch_size, 1])
                    tf_op.variable_summaries(out, 'state_{}_q_value'.format(i))
            #print(self.value_tensor.shape.as_list())
        with tf.variable_scope('square_loss'):
            diff = tf.squared_difference(target_tensor, value_tensor)
            self.loss = tf.reduce_mean(diff)
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = tf_op.truncated_grads(optimizer, self.loss, 1.0)

        # for tensorboard
        self.merged = tf.summary.merge_all()

        # for update target netwrok
        with tf.variable_scope('update_target_network'):
            target_params = self.target_network.params
            value_params = self.value_network.params
            self.hard_target_network_updater = tf.group(*[tf.assign(target_params[key], value_params[key]) for key in target_params])
            self.soft_target_network_updater = tf.group(*[tf.assign(target_params[key], 0.99 * target_params[key] + 0.01 * value_params[key]) for key in target_params])

    def choose_action(self, state):
        # to have batch dimension when feed into tf placeholder
        state = state[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            # forward feed the observation and get q value for every actions
            action_value = self.value_network.get_output_value(state)
            action = np.argmax(action_value)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return action

    def learn(self, state, action, reward, next_state, is_done):
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict={
            self.state: state,
            self.action: action,
            self.reward: reward,
            self.next_state: next_state,
            self.is_done: is_done,
        })
        if self.step % self.target_update_threshold == 0:
            #self.sess.run(self.soft_target_network_updater)
            self.sess.run(self.hard_target_network_updater)
        self.step += 1
        return summary

def test():
    import shutil; shutil.rmtree('gym_logs', True)
    import gym
    env = gym.make("CartPole-v1")
    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())
    state_dim = 4
    action_dim = 2
    #print(env.action_space.sample())
    with tf.Session() as sess:
        with tf.variable_scope('target_network'):
            ipt = tf.placeholder(tf.float32, shape=(None, state_dim))
            target_network = neural_network.MLP(sess, ipt, [20, action_dim], ['relu', 'none'])
        with tf.variable_scope('value_network'):
            ipt = tf.placeholder(tf.float32, shape=(None, state_dim))
            value_network = neural_network.MLP(sess, ipt, [20, action_dim], ['relu', 'none'])
        net = DQN(
                sess = sess,
                value_network=value_network,
                target_network=target_network,
                learning_rate=0.1,
                epsilon_init=1,
                epsilon_decay=0.999,
                epsilon_min=0.01,
                gamma=0.999,
                target_update_threshold=32,
                )
        buff = replay_buffer.ReplayBuffer(50000)
        batch_size = 128
        start_learn = batch_size
        render_time = 1
        max_step_time = 500
        writer = tf.summary.FileWriter('gym_logs', sess.graph)

        sess.run(tf.global_variables_initializer())

        for i_episode in range(100000):
            observation = env.reset()
            reward_sum = 0
            for t in range(max_step_time):
                if len(buff) > start_learn and i_episode % render_time == 0:
                    env.render()

                action = net.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                reward_sum += reward
                buff.add(observation, action, reward, next_observation, done)
                if len(buff) > start_learn:
                    #net.learn(observation, action, reward, next_observation, done)
                    summary = net.learn(*buff.sample(batch_size))
                    writer.add_summary(summary, i_episode)
                observation = next_observation
                if done or t == max_step_time - 1:
                    if 'avg' not in locals():
                        avg = reward_sum
                    else:
                        avg = 0.99 * avg + 0.01 * reward_sum
                    print('i_episode:', i_episode, 'avg:', avg, 'reward_sum:', reward_sum)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='epsilon', simple_value=net.epsilon)])
                    writer.add_summary(summary, i_episode)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='finished timesteps', simple_value=t + 1)])
                    writer.add_summary(summary, i_episode)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='reward_sum', simple_value=reward_sum)])
                    writer.add_summary(summary, i_episode)
                    break
