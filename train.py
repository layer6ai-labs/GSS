from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import numpy as np
from model import *

from utils.helper import *

parser = argparse.ArgumentParser()
parser.add_argument('--kq', type=int, default=5, help='Top k number for the query graph.')
parser.add_argument('--k', type=int, default=5, help='Top k number for the index graph.')
parser.add_argument('--alpha', type=float, default=1, help='Parameter alpha for gss loss.')
parser.add_argument('--beta', type=float, default=None, help='Parameter beta for gss loss.')
parser.add_argument('--beta-percentile', type=float, default=None, help='Automatically select beta by the percentile of similarity matrix''s distribution.')
parser.add_argument('--seed', type=int, default=None, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden-units', type=int, default=2048, help='Number of units in hidden layer')
parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--init-weights', type=float, default=1e-5, help='The std of the off-diagonal elements of the weights in GCN referred as epsilon.')
parser.add_argument('--regularizer-scale', type=float, default=1e-5, help='The scale of l2 regularization')
parser.add_argument('--layer-decay', type=float, default=0.3, help='Residual GCN layer decay.')
parser.add_argument('--dataset', type=str, default='roxford5k', choices=['roxford5k', 'rparis6k', 'instre'],
                    help='Dataset.')
parser.add_argument('--data-path', type=str, default=None, help='Dataset files location.')
parser.add_argument('--gpu-id', type=int, default=None, help='Which GPU to use. By default None means training on CPU.')
parser.add_argument("--report-hard", help="If evaluate on Hard setup or Medium. It doesn't matter to INSTRE",
                    action="store_true")
parser.add_argument("--graph-mode", type=str, default='descriptor',
                    choices=['descriptor', 'ransac', 'approx_ransac'],
                    help="Choose the way to construct kNN graph. Descriptor mode uses the "
                         "inner product of dense descriptors referred as GSS in the GeM+GSS. Ransac "
                         "mode applies spatial verification on both query and index graphs referred as "
                         "GeM+GSS_V-SV. Approx_ransac mode is a fast inference method where spatial "
                         "verification is only applied on index graph during offline training phase "
                         "referred as GeM+GSS_V.")
args = parser.parse_args()
for key in vars(args):
    print(key + ":" + str(vars(args)[key]))

if args.beta is not None and args.beta_percentile is not None:
    raise Exception('beta and beta_percentile can not be used at the same time!')


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):

    if args.seed:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)

    Q, X = load_data(args.dataset, args.data_path)

    if args.graph_mode == 'ransac':
        q_RANSAC_graph, x_RANSAC_graph = load_ransac_graph(args.dataset, args.data_path)
    elif args.graph_mode == 'approx_ransac':
        _, x_RANSAC_graph = load_ransac_graph(args.dataset, args.data_path)
        q_RANSAC_graph = None
    else:
        q_RANSAC_graph, x_RANSAC_graph = None, None

    q_adj, q_features, x_adj, x_features = gen_graph(Q, X, args.kq, args.k, q_RANSAC_graph, x_RANSAC_graph)

    all_features = np.concatenate([q_features, x_features])
    all_adj = combine_graph(q_adj, x_adj)

    all_adj_normed = preprocess_graph(all_adj)
    x_adj_normed = preprocess_graph(x_adj)

    revop_map = get_roc_score_matrix(all_features, Q.shape[1], args.report_hard, args.dataset)
    print('GEM descriptors performance: {}'.format(revop_map))

    x_adj_normed_sparse_tensor = convert_sparse_matrix_to_sparse_tensor(x_adj_normed)
    all_adj_normed_sparse_tensor = convert_sparse_matrix_to_sparse_tensor(all_adj_normed)

    features_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, args.hidden_units])
    adj_placeholder = tf.sparse_placeholder(dtype=tf.float32, shape=[None, None])

    regularizer = tf.contrib.layers.l2_regularizer(scale=args.regularizer_scale)

    model = ResidualGraphConvolutionalNetwork(train_batch_size=x_adj_normed.shape[0],
                                              val_batch_size=all_adj_normed.shape[0],
                                              num_layers=args.num_layers,
                                              hidden_units=args.hidden_units,
                                              init_weights=args.init_weights,
                                              layer_decay=args.layer_decay)

    training_dataset = tf.data.Dataset.from_tensor_slices(features_placeholder).batch(model.train_batch_size).repeat()
    validation_dataset = tf.data.Dataset.from_tensor_slices(features_placeholder).batch(model.val_batch_size)

    iterator = tf.data.Iterator.from_structure(training_dataset.output_types)
    itr_train_init_op = iterator.make_initializer(training_dataset)
    itr_valid_init_op = iterator.make_initializer(validation_dataset)
    next_element = iterator.get_next()

    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

    logits = model.network(next_element, adj_placeholder, regularizer)

    beta_placeholder = tf.placeholder(dtype=tf.float32)

    loss = GSS_loss(args.alpha, beta_placeholder).gss_loss(logits)

    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss += reg_term
    grads_and_vars = optimizer.compute_gradients(loss)

    update_op = optimizer.apply_gradients(grads_and_vars)
    init_op = tf.global_variables_initializer()

    best_map = 0.0

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
        sess.run(init_op)
        itr = 0

        while itr < args.epochs:
            # training step
            sess.run(itr_train_init_op, feed_dict={features_placeholder: x_features})
            start_time = time.time()
            if itr == 0:
                hidden_emb = sess.run(model.hidden_emb, feed_dict={adj_placeholder: x_adj_normed_sparse_tensor})
                if args.beta_percentile is not None:
                    beta_score = np.percentile(np.dot(hidden_emb, hidden_emb.T).flatten(), args.beta_percentile)
                elif args.beta is not None:
                    beta_score = args.beta
                else:
                    raise Exception('At least one of beta and beta_percentile should be set!')
            _, loss_out = sess.run([update_op, loss], feed_dict={adj_placeholder: x_adj_normed_sparse_tensor,
                                                                 beta_placeholder: beta_score})
            train_time = time.time() - start_time
            itr += 1
            # eval step
            sess.run(itr_valid_init_op, feed_dict={features_placeholder: all_features})
            start_time = time.time()
            hidden_emb_np = sess.run(model.hidden_emb, feed_dict={adj_placeholder: all_adj_normed_sparse_tensor})
            eval_time = time.time() - start_time
            revop_map = get_roc_score_matrix(hidden_emb_np, Q.shape[1], args.report_hard, args.dataset)
            if revop_map > best_map:
                best_map = revop_map
            print("train time: {}, eval time: {}, revop:{}, best revop:{}".format(str(train_time), str(eval_time),
                                                                                 revop_map, best_map))

if __name__ == '__main__':
    main(args)

