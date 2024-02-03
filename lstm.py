import numpy as np
import tensorflow as tf

class LSTM:
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (4, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (4, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((4, hidden_dim))
        c = np.zeros(word_dim)

        # TensorFlow variables
        self.E = tf.Variable(E.astype(np.float32))
        self.U = tf.Variable(U.astype(np.float32))
        self.W = tf.Variable(W.astype(np.float32))
        self.V = tf.Variable(V.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.c = tf.Variable(c.astype(np.float32))

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop()

    def forward_prop_step(self, x_t, h_t_prev, c_t_prev):
        x_e = tf.nn.embedding_lookup(self.E, x_t)

        i_t = tf.nn.hard_sigmoid(tf.matmul(self.U[0], x_e) + tf.matmul(self.W[0], h_t_prev) + self.b[0])
        f_t = tf.nn.hard_sigmoid(tf.matmul(self.U[1], x_e) + tf.matmul(self.W[1], h_t_prev) + self.b[1])
        o_t = tf.nn.hard_sigmoid(tf.matmul(self.U[2], x_e) + tf.matmul(self.W[2], h_t_prev) + self.b[2])
        g_t = tf.nn.tanh(tf.matmul(self.U[3], x_e) + tf.matmul(self.W[3], h_t_prev) + self.b[3])
        c_t = c_t_prev * f_t + g_t * i_t
        h_t = tf.nn.tanh(c_t) * o_t

        output_t = tf.nn.softmax(tf.matmul(self.V, h_t) + self.c)

        return output_t, h_t, c_t

    def calculate_loss(self, x, y):
        with tf.GradientTape() as tape:
            output, _, _ = self.forward_propagation(x)
            output_error = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
            loss = output_error

        gradients = tape.gradient(loss, [self.E, self.U, self.W, self.V, self.b, self.c])
        return loss, gradients

    def train_step(self, x, y):
        loss, gradients = self.calculate_loss(x, y)
        self.optimizer.apply_gradients(zip(gradients, [self.E, self.U, self.W, self.V, self.b, self.c]))
        return loss

    def forward_propagation(self, x):
        initial_hidden_state = tf.zeros([tf.shape(x)[0], self.hidden_dim])
        initial_cell_state = tf.zeros([tf.shape(x)[0], self.hidden_dim])

        outputs, _, _ = tf.scan(
            self.forward_prop_step,
            x,
            initializer=(tf.zeros([tf.shape(x)[0], self.word_dim]), initial_hidden_state, initial_cell_state)
        )

        return outputs
