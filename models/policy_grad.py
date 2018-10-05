import tensorflow as tf
import numpy as np

from models import neural_network
from utils.tf_util import tf_op

class PolicyGradient():
    def __init__(self, sess, network, learning_rate):
        self.learning_rate = learning_rate
        self.sess = sess
        self.state_dim = network.input_dimension
        self.action_dim = network.output_dimension

        self.network = network

        self.__buil_network()

    def __buil_network(self):
        self.state = self.network.input_tensor # input State
        self.action = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.vt = tf.placeholder(tf.float32, [None, ], name='r')

        with tf.variable_scope('network_output'):
            out_logits = self.network.output_tensor
            self.out_prob = tf.nn.softmax(out_logits, name='act_prob')

        with tf.variable_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_logits, labels=self.action)
            self.loss = tf.reduce_mean(neg_log_prob * self.vt)
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = tf_op.truncated_grads(optimizer, self.loss, 1.0)

        # for tensorboard
        self.merged = tf.summary.merge_all()

    def choose_action(self, state):
        prob_weights = self.sess.run(self.out_prob, feed_dict={
            self.state: state[np.newaxis, :],
        })
        action = np.random.choice(range(self.action_dim), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self, state, action, vt):
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict={
            self.state: state,
            self.action: action,
            self.vt: vt,
        })
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
        with tf.variable_scope('network'):
            ipt = tf.placeholder(tf.float32, shape=(None, state_dim))
            network = neural_network.MLP(sess, ipt, [20, action_dim], ['tanh', 'none'])
        net = PolicyGradient(
                sess = sess,
                network=network,
                learning_rate=0.1,
                )
        gamma = 0.999
        batch_size = 128
        start_learn = batch_size
        render_time = 1
        max_step_time = 500
        writer = tf.summary.FileWriter('gym_logs', sess.graph)

        sess.run(tf.global_variables_initializer())

        for i_episode in range(100000):
            observation = env.reset()
            states, actions, vt = [], [], []
            reward_sum = 0
            for i in range(max_step_time):
                if i_episode > start_learn and i_episode % render_time == 0:
                    env.render()

                action = net.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                reward_sum += reward
                states.append(observation)
                actions.append(action)
                vt.append(reward)
                observation = next_observation
                if done or i == max_step_time - 1:
                    # calc vt
                    t = 0
                    for i in reversed(range(len(vt))):
                        t = t * gamma + vt[i]
                        vt[i] = t

                    # learn
                    summary = net.learn(states, actions, vt)
                    writer.add_summary(summary, i_episode)

                    # plot
                    summary = tf.Summary(value=[tf.Summary.Value(tag='finished timesteps', simple_value=t + 1)])
                    writer.add_summary(summary, i_episode)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='reward_sum', simple_value=reward_sum)])
                    writer.add_summary(summary, i_episode)
                    break
