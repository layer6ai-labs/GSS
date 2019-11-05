import tensorflow as tf
import numpy as np

class ResidualGraphConvolutionalNetwork():
    def __init__(self, train_batch_size, val_batch_size, num_layers=2,
                 hidden_units=2048, init_weights=1e-5, layer_decay=0.4):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.init_weights = init_weights
        self.layer_decay = layer_decay

    def gcn_layer(self, x, adj, i_layer, regularizer):

        with tf.variable_scope('GCN_{}'.format(i_layer)):
            init_w = (np.random.randn(self.hidden_units, self.hidden_units) * self.init_weights)
            init_w[np.where(np.eye(self.hidden_units) != 0)] = 1
            constant_init = tf.convert_to_tensor(init_w, dtype=tf.float32)
            W = tf.get_variable(name="w", dtype=tf.float32,
                                 initializer=constant_init,
                                 regularizer=regularizer)
            B = tf.get_variable(name='b', shape=[self.hidden_units], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            #Ax
            Ax = tf.sparse_tensor_dense_matmul(adj, x)
            pre_nonlinearity = tf.nn.bias_add(tf.matmul(Ax, W), B)
            output = tf.nn.elu(pre_nonlinearity)

            return pre_nonlinearity, output

    def decoder(self, x):

        with tf.variable_scope('Decoder'):
            self.hidden_emb = tf.nn.l2_normalize(x, axis=1)
            adj_preds = tf.matmul(self.hidden_emb, tf.transpose(self.hidden_emb))
            adj_preds = tf.nn.relu(adj_preds)

            return adj_preds

    def network(self, x, adj, regularizer):
        residual = None
        for i in range(1, self.num_layers + 1):
            pre_nonlinearity, x = self.gcn_layer(x, adj, i, regularizer)
            if residual is not None:
                x = residual + self.layer_decay * x
            residual = pre_nonlinearity
        output = self.decoder(x)

        return output

class GSS_loss():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def gss_loss(self, logits):
        losses = -0.5 * self.alpha * (logits - self.beta) ** 2
        return tf.reduce_mean(losses)


