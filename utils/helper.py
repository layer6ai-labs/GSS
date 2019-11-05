from utils.revop import *
import scipy.sparse as sp
import tensorflow as tf
import time

def load_data(dataset, data_path):
    if dataset == 'instre':
        cfg, data = init_intre(dataset, data_path)
    else:
        cfg,data = init_revop(dataset, data_path)
    Q = data['Q']
    X = data['X']
    return Q, X

def load_ransac_graph(dataset, data_path):
    q_RANSAC_graph = np.load(os.path.join(data_path, 'graphs', '{}_query_ransac_graph.npy'.format(dataset)))
    x_RANSAC_graph = np.load(os.path.join(data_path, 'graphs', '{}_index_ransac_graph.npy'.format(dataset)))
    return q_RANSAC_graph, x_RANSAC_graph

def gen_graph(Q, X, kq=5, k=5, q_RANSAC_graph=None, x_RANSAC_graph=None):
    t = time.time()
    q_sim = np.matmul(Q.T, X)

    if q_RANSAC_graph is not None:
        q_sim_top = q_RANSAC_graph[:, 0:kq]
    else:
        q_sim_top = np.argpartition(q_sim, -kq, 1)[:, -kq:]

    q_adj = np.zeros(q_sim.shape)
    for i in range(q_adj.shape[0]):
        q_adj[i,q_sim_top[i]] = q_sim[i,q_sim_top[i]]
    q_adj = sp.csr_matrix(q_adj)

    x_sim = np.matmul(X.T, X)

    if x_RANSAC_graph is not None:
        x_sim_top = x_RANSAC_graph[:, 0:k]
    else:
        x_sim_top = np.argpartition(x_sim, -k, 1)[:, -k:]

    x_adj = np.zeros(x_sim.shape)

    for i in range(x_adj.shape[0]):
        x_adj[i, x_sim_top[i]] = x_sim[i, x_sim_top[i]]
        x_adj[x_sim_top[i], i] = x_sim[i, x_sim_top[i]]
        x_adj[i, i] = 0

    x_adj = sp.csr_matrix(x_adj)

    print('Created G_q with [kq={}] [shape=[{}, {}]] , G with [k={}] [shape=[{}, {}]] in {:.2f}s'.
          format(kq, q_adj.shape[0], q_adj.shape[1], k, x_adj.shape[0], x_adj.shape[1], time.time() - t))

    return q_adj, Q.T, x_adj, X.T

def get_roc_score_matrix(emb, Q_end=70, report_hard=True, dataset='roxford5k'):
    embQ = emb[:Q_end, :].T
    embX = emb[Q_end:, :].T
    revop_inner_prod = np.matmul(embX.T, embQ)
    revop_preds = np.argsort(-revop_inner_prod, axis=0)
    if dataset == 'instre':
        map = eval_instre(revop_preds, silent=True)
    else:
        map = eval_revop(revop_preds, silent=True, report_hard=report_hard)
    return map

def combine_graph(q_adj, x_adj):
    adj_all = sp.vstack((q_adj, x_adj))
    zeros = sp.csr_matrix((adj_all.shape[0], q_adj.shape[0]))
    adj_all = sp.hstack((zeros, adj_all))
    adj_all = sp.csr_matrix(adj_all)

    return adj_all

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).transpose()

    return sp.csr_matrix(adj_normalized)

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data.astype(np.float32), coo.shape)