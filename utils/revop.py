import os
from scipy.io import loadmat
import numpy as np

from utils.evaluate import compute_map

def init_revop(dataset,data_root):
    global test_dataset
    global cfg
    global features
    test_dataset = dataset
    # config file for the dataset
    # separates query image list from database image list, when revisited protocol used
    cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))
    # load query and database features
    print('>> {}: Loading features...'.format(test_dataset))
    features = loadmat(os.path.join(data_root, 'features', '{}_resnet_rsfm120k_gem.mat'.format(test_dataset)))
    return cfg, features

def init_intre(dataset,data_root):
    global test_dataset
    global cfg
    global features
    test_dataset = dataset
    # config file for the dataset
    # separates query image list from database image list, when revisited protocol used
    print('>> {}: Evaluating test dataset...'.format(test_dataset))
    cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))
    # load query and database features
    print('>> {}: Loading features...'.format(test_dataset))
    features = {}
    features['Q'] = np.load(os.path.join(data_root, 'features', 'instre_gem_query_ms_lw.npy'))
    features['X'] = np.load(os.path.join(data_root, 'features', 'instre_gem_index_ms_lw.npy'))
    return cfg, features

def eval_revop(p,silent=False, report_hard=True):
    # revisited evaluation
    gnd = cfg['gnd']

    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(p, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(p, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(p, gnd_t, ks)
    if not silent:
        print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

    if report_hard:
        return np.around(mapH*100, decimals=2)
    else:
        return np.around(mapM * 100, decimals=2)

def eval_instre(p,silent = False):
    # revisited evaluation
    gnd = cfg['gnd']

    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['ok']])
        #g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(p, gnd_t, ks)
    if not silent:
        print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapH*100, decimals=2), np.around(mapH*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprH*100, decimals=2), np.around(mprH*100, decimals=2), np.around(mprH*100, decimals=2)))
    return np.around(mapH*100, decimals=2)

def configdataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset == 'roxford5k' or dataset == 'rparis6k' or dataset =='instre':
        cfg = {'ext' : '.jpg', 'qext' : '.jpg', 'dir_data' : os.path.join(dir_main, dataset)}
        cfg['gnd_fname'] = os.path.join(cfg['dir_data'], 'gnd_' + dataset + '.mat')
        print(cfg['gnd_fname'])
        gt = loadmat(cfg['gnd_fname'])
        cfg['imlist'] = [str(''.join(im)) for iml in np.squeeze(gt['imlist']) for im in iml]
        cfg['qimlist'] = [str(''.join(im)) for iml in np.squeeze(gt['qimlist']) for im in iml]
        cfg['gnd'] = gnd_mat2py(gt['gnd'])
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])

    elif dataset == 'revisitop1m':
        cfg = {'ext' : '.jpg', 'dir_data' : os.path.join(dir_main, dataset)}
        cfg['imlist_fname'] = os.path.join(cfg['dir_data'], '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['n'] = len(cfg['imlist'])

    else:
        raise ValueError('Unknown dataset: %s!' % dataset)

    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg


def config_imname(cfg, i):
    _, ext = os.path.splitext(cfg['imlist'][i])
    if ext:
        return os.path.join(cfg['dir_images'], cfg['imlist'][i])
    else:
        return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    _, ext = os.path.splitext(cfg['qimlist'][i])
    if ext:
        return os.path.join(cfg['dir_images'], cfg['qimlist'][i])
    else:
        return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


def gnd_mat2py(gnd):
    gnd = np.squeeze(gnd);
    gndpy = []
    for i in np.arange(len(gnd)):
        gndi = gnd[i]
        gndpyi = {}
        try:
            gndpyi['ok'] = gnd[i]['ok']-1
            gndpyi['ok'] = gndpyi['ok'].reshape(gndpyi['ok'].shape[1])
        except:
            pass
        try:
            gndpyi['easy'] = gnd[i]['easy']-1
            gndpyi['easy'] = gndpyi['easy'].reshape(gndpyi['easy'].shape[1])
        except:
            pass
        try:
            gndpyi['hard'] = gnd[i]['hard']-1
            gndpyi['hard'] = gndpyi['hard'].reshape(gndpyi['hard'].shape[1])
        except:
            pass
        try:
            gndpyi['junk'] = gnd[i]['junk']-1
            gndpyi['junk'] = gndpyi['junk'].reshape(gndpyi['junk'].shape[1])
        except:
            pass
        try:
            gndpyi['bbx'] = np.squeeze(gnd[i]['bbx'])
        except:
            pass
        gndpy.append(gndpyi)

    return gndpy


def read_imlist(imlist_fn):
    file = open(imlist_fn, 'r')
    imlist = file.read().splitlines();
    file.close()
    return imlist
