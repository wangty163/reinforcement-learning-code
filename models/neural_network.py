import tensorflow as tf
import numpy as np

from utils.tf_util import tf_op

def activate(weights, func_name, name=None):
    """ function """
    func_name = func_name.lower()
    if func_name == 'sigmoid':
        return tf.nn.sigmoid(weights, name=name)
    elif func_name == 'softmax':
        return tf.nn.softmax(weights, name=name)
    elif func_name == 'relu':
        return tf.nn.relu(weights, name=name)
    elif func_name == 'tanh':
        return tf.nn.tanh(weights, name=name)
    elif func_name == 'elu':
        return tf.nn.elu(weights, name=name)
    elif func_name == 'none':
        return weights
    else:
        return tf.nn.relu(weights, name=name)

class MLP:
    """
    生成MLP网络, in_placeholder 为输入的inplaceholder, layer_sizes为每层的维数，activation_names为对应每层的activation
    in_placeholder=tf.placeholder(tf.float32, [None, 42])
    layer_sizes=[10, 1], activation_names=['relu', 'none']
    """
    def __init__(self, sess, in_placeholder, layer_sizes, activation_names, initial_val_dict=None):
        tf.assert_equal(tf.rank(in_placeholder), 2, message='input rank should be 2')

        self.input_dimension = in_placeholder.shape.as_list()[1]
        self.output_dimension = layer_sizes[-1]
        self.input_tensor = in_placeholder
        self.sess = sess
        self.params = {}

        # build network
        layer_sizes.insert(0, self.input_dimension)
        activation_names.insert(0, 'none')

        layers_n = len(layer_sizes)
        cur_in = in_placeholder #当前层的输入
        for ind in range(layers_n - 1):
            in_size = layer_sizes[ind]
            out_size = layer_sizes[ind + 1]
            w_name = "W%d" % (ind + 1)
            b_name = "b%d" % (ind + 1)

            if initial_val_dict is None or w_name not in initial_val_dict:
                w_initializer = tf.initializers.truncated_normal(stddev=0.1 / np.sqrt(float(in_size)))
                #w_initial_val = tf.truncated_normal([in_size, out_size], stddev=0.1 / np.sqrt(float(in_size)))
            else:
                w_initializer = initial_val_dict[w_name]

            if initial_val_dict is None or b_name not in initial_val_dict:
                b_initializer = tf.initializers.truncated_normal(stddev=0.1 / np.sqrt(float(in_size)))
                #b_initial_val = tf.truncated_normal([1, out_size], stddev=0.1 / np.sqrt(float(in_size)))
            else:
                b_initializer = initial_val_dict[b_name]

            w = tf.get_variable(w_name, shape=[in_size, out_size], dtype=tf.float32, initializer=w_initializer)
            b = tf.get_variable(b_name, shape=[1, out_size], dtype=tf.float32, initializer=b_initializer)

            for i in range(in_size):
                tensor = tf.gather(w, i, axis=0)
                tf_op.variable_summaries(tensor, '{}_{}'.format(w_name, i))
            tf_op.variable_summaries(w, '{}'.format(w_name))
            tf_op.variable_summaries(b, '{}'.format(b_name))

            #w = tf.Variable(w_initial_val, name=w_name, dtype=tf.float32)
            #b = tf.Variable(b_initial_val, name=b_name, dtype=tf.float32)
            z = tf.add(tf.matmul(cur_in, w), b)
            cur_out = activate(z, func_name=activation_names[ind + 1])
            self.params[w_name] = w
            self.params[b_name] = b
            cur_in = cur_out
        self.output_tensor = cur_out

    @property
    def param_tensor(self):
        return self.__params.values()

    def get_output_value(self, input_value):
        feed_dict = {
            self.input_tensor: input_value,
        }
        output = self.sess.run(self.output_tensor, feed_dict)
        return output

    def get_gradients(self, loss):
        trainable_variables = list(self.params.values())
        grads = tf.gradients(loss, trainable_variables)
        grads = list(zip(grads, trainable_variables))
        return grads

def test():
    with tf.Session() as sess:
        ipt = tf.placeholder(tf.float32, shape=(None, 2))
        mlp = MLP(sess, ipt, [1], ['none'])

        sess.run(tf.global_variables_initializer())

        output = mlp.get_output_value(np.random.rand(1, 2))
        print(output)

if __name__ == '__main__':
    test()
