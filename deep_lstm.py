import numpy as np
import tensorflow as tf

class DLSTM:
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters
        self.E = tf.Variable(tf.random.uniform([word_dim, hidden_dim], -np.sqrt(1. / word_dim), np.sqrt(1. / word_dim)))
        self.U = tf.Variable(tf.random.uniform([8, hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim)))
        self.W = tf.Variable(tf.random.uniform([8, hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim)))
        self.V = tf.Variable(tf.random.uniform([hidden_dim, word_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim)))
        self.b = tf.Variable(tf.zeros([8, hidden_dim]))
        self.c = tf.Variable(tf.zeros([word_dim]))

        self.optimizer = tf.keras.optimizers.RMSprop()

    def forward_prop_step(self, x_t, h_t1_prev, h_t2_prev, c_t1_prev, c_t2_prev):
        x_e = tf.nn.embedding_lookup(self.E, x_t)

        i_t1 = tf.nn.hard_sigmoid(tf.matmul(x_e, self.U[0]) + tf.matmul(h_t1_prev, self.W[0]) + self.b[0])
        f_t1 = tf.nn.hard_sigmoid(tf.matmul(x_e, self.U[1]) + tf.matmul(h_t1_prev, self.W[1]) + self.b[1])
        o_t1 = tf.nn.hard_sigmoid(tf.matmul(x_e, self.U[2]) + tf.matmul(h_t1_prev, self.W[2]) + self.b[2])
        g_t1 = tf.nn.tanh(tf.matmul(x_e, self.U[3]) + tf.matmul(h_t1_prev, self.W[3]) + self.b[3])
        c_t1 = c_t1_prev * f_t1 + g_t1 * i_t1
        h_t1 = tf.nn.tanh(c_t1) * o_t1

        i_t2 = tf.nn.hard_sigmoid(tf.matmul(h_t1, self.U[4]) + tf.matmul(h_t2_prev, self.W[4]) + self.b[4])
        f_t2 = tf.nn.hard_sigmoid(tf.matmul(h_t1, self.U[5]) + tf.matmul(h_t2_prev, self.W[5]) + self.b[5])
        o_t2 = tf.nn.hard_sigmoid(tf.matmul(h_t1, self.U[6]) + tf.matmul(h_t2_prev, self.W[6]) + self.b[6])
        g_t2 = tf.nn.tanh(tf.matmul(h_t1, self.U[7]) + tf.matmul(h_t2_prev, self.W[7]) + self.b[7])
        c_t2 = c_t2_prev * f_t2 + g_t2 * i_t2
        h_t2 = tf.nn.tanh(c_t2) * o_t2

        output_t = tf.nn.softmax(tf.matmul(h_t2, self.V) + self.c)

        return output_t, h_t1, h_t2, c_t1, c_t2

    def train_step(self, x, y):
        loss, gradients = self.calculate_loss(x, y)
        self.optimizer.apply_gradients(zip(gradients, [self.E, self.U, self.W, self.V, self.b, self.c]))
        return loss

    def forward_propagation(self, x):
        initial_hidden_state1 = tf.zeros([tf.shape(x)[0], self.hidden_dim])
        initial_hidden_state2 = tf.zeros([tf.shape(x)[0], self.hidden_dim])
        initial_cell_state1 = tf.zeros([tf.shape(x)[0], self.hidden_dim])
        initial_cell_state2 = tf.zeros([tf.shape(x)[0], self.hidden_dim])

        outputs, _, _, _, _ = tf.scan(
            self.forward_prop_step,
            x,
            initializer=(tf.zeros([tf.shape(x)[0], self.word_dim]), initial_hidden_state1, initial_hidden_state2, initial_cell_state1, initial_cell_state2)
        )

        return outputs
    
    def calculate_loss(self, x, y):
        with tf.GradientTape() as tape:
            output, _, _, _, _ = self.forward_propagation(x)
            output_error = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
            loss = output_error

        gradients = tape.gradient(loss, [self.E, self.U, self.W, self.V, self.b, self.c])
        return loss, gradients