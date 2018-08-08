# Finetune a network in tensorflow on the CUB dataset
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import ntpath
import json
import pdb
import random
import torchfile
import importlib
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.ops import array_ops
from tensorflow.python import pywrap_tensorflow
from pprint import pprint
import pickle
from dotmap import DotMap
import glob
import itertools
from itertools import groupby
from random import shuffle
from tqdm import tqdm
sys.path.insert(0, '/nethome/rrs6/models/research/slim')

SEED = 1111
tf.set_random_seed(SEED)

random.seed(SEED)


"""
Some classes in Scott Reed's captions
are named differently than the
original CUB_200_2011 folder
"""
CUB_FNAME_FIX = {'093.Clark_Nutcracker': '093.Clarks_Nutcracker',
                 '124.Le_Conte_Sparrow': '124.Le_Contes_Sparrow',
                 '180.Wilson_Warbler': '180.Wilsons_Warbler',
                 '125.Lincoln_Sparrow': '125.Lincolns_Sparrow',
                 '023.Brandt_Cormorant': '023.Brandts_Cormorant',
                 '178.Swainson_Warbler': '178.Swainsons_Warbler',
                 '122.Harris_Sparrow': '122.Harriss_Sparrow',
                 '113.Baird_Sparrow': '113.Bairds_Sparrow',
                 '123.Henslow_Sparrow': '123.Henslows_Sparrow',
                 '098.Scott_Oriole': '098.Scotts_Oriole',
                 '061.Heermann_Gull': '061.Heermanns_Gull',
                 '022.Chuck_will_Widow': '022.Chuck_wills_Widow',
                 '193.Bewick_Wren': '193.Bewicks_Wren',
                 '067.Anna_Hummingbird': '067.Annas_Hummingbird',
                 '126.Nelson_Sharp_tailed_Sparrow': '126.Nelsons_Sparrow',
                 '115.Brewer_Sparrow': '115.Brewers_Sparrow',
                 '009.Brewer_Blackbird': '009.Brewers_Blackbird'}

parser = argparse.ArgumentParser()
parser.add_argument('--config_json', default='./arg_configs/vgg16_config_AWA_full.json')

VGG_MEAN = [123.68, 116.78, 103.94]


def uhead_plotter(l1, l2, l3, l4, l5, l6, directory, mode):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(l1)), l1, label='seen_seen')
    ax1.plot(range(len(l2)), l2, label='unseen_unseen')
    ax1.plot(range(len(l3)), l3, label='seen_unseen_seen_unseen')
    ax1.plot(range(len(l4)), l4, label='seen_seen_unseen')
    ax1.plot(range(len(l5)), l5, label='unseen_seen_unseen')
    ax1.plot(range(len(l6)), l6, label='harmonic')
    plt.legend()
    plt.title(mode)
    fig.savefig(directory + mode + '.png')
    plt.close(fig)


def uhead_plotter_loss(l1,directory, mode):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(l1)), l1, label=' val set')
    plt.legend()
    plt.title(mode)
    fig.savefig(directory + mode + '.png')
    plt.close(fig)


def parse_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def norm1(a):
    return np.sum(np.abs(a))


def entropy(ls):
    print(ls)
    probs = {x:ls.count(x)/len(ls) for x in ls}
    p = np.array(list(probs.values()))
    return -p.dot(np.log2(p))

def encode_attributes_class(config, imlabelist):
    im_attr = {}
    cls_attr = {}
    # Use class level supervision
    if config.supervision=='class':
        class_att_labels = []
        with open(config.classattrdir) as f:
            for n, line in enumerate(f):
                l = [x for x in line.rstrip().split(" ") ]
                l = [x for x in l if x]
                #l = l.remove('')
                l = [float(x) if float(x)!=-1.00 else 0 for x in l ]
                cls_attr[n]=l
                class_att_labels.append(l)
        class_att_labels = np.array(class_att_labels)
        #class_att_avg = np.mean(class_att_labels, axis = 0)

        for c in range(int(config.n_class)):
            imids = [k for k,v in im_attr.items() if v['cls'] == c]
            for id in imids:
                im_attr[id] = {}
                im_attr[id]['att']= class_att_labels[c]/np.max(class_att_labels)
    return im_attr, cls_attr


def encode_tfidf(tf_file, imlabelist, config):
    # Function to encode the TF-IDF features from wikipedia articles
    # Make this compatible with Ram's attribute encoding function
    attrdir = './data/CUB/11083D_TFIDF.mat'
    tf_idf = scio.loadmat(attrdir)['PredicateMatrix']
    im_attr = {}
    print('Encoding TF-IDF....')
    for i in tqdm(range(len(imlabelist))):
        #print(tf_idf[imlabelist[i]-1].tolist())
        im_attr[str(i+1)] = {}
        im_attr[str(i+1)]['att'] = tf_idf[imlabelist[i]-1].tolist()
    return im_attr


def encode_captions(cap_dir, imlist_new, imlabelist, config):
    # config.attrdir has to be 2 directories joined as strings default argument to use
    # In interest of time, we're only doing w2v captions
    # Get caption text dir and feature dir
    attrdir = './data/CUB/text_c10,./data/CUB/w2v_c10'
    cap_dir = attrdir.split(',')[0]
    feat_dir = attrdir.split(',')[1]
    # Load appropriate mapping
    all_f = glob.glob(cap_dir + '/*/*.txt')
    all_f = sorted(all_f)
    all_f = [x.replace('./','').replace('.txt', '.jpg').replace(cap_dir + '/', '') for x in all_f]
    # Load all class t7 files
    t7_dict = {}
    class_names = list(set([x.split('/')[0] for x in imlist_new]))
    print('Loading caption feature files..')
    for i in class_names:
        fname = i
        if fname in list(CUB_FNAME_FIX.keys()):
            fname = CUB_FNAME_FIX[fname]
        t7_dict[i] = torchfile.load(feat_dir + '/' + fname + '.t7')
    im_attr = {}
    cls_attr = {}
    # Do this iteratively
    print('Encoding captions...')
    for i in tqdm(range(len(imlist_new))):
        imname = imlist_new[i]
        class_id = int(imlist_new[i].split('.')[0])-1
                # Image name to class-t7 file
        class_name = imname.split('/')[0]
        data = t7_dict[class_name]
        imind = all_f.index(imname)
        indlist = sorted([all_f.index(x) for x in all_f if class_name in x])
        pos = indlist.index(imind)
        feat = data[pos].T
        #im_attr[str(i+1)] = {}
        im_attr[str(i+1)] = np.mean(feat, axis=0).tolist()

        if class_id in cls_attr:
            cls_attr[class_id].append(np.mean(feat, axis=0))
        else:
            cls_attr[class_id] = []
            cls_attr[class_id].append(np.mean(feat, axis=0))
    for id in cls_attr:
        cls_attr[id] = np.array(cls_attr[id]).mean(axis=0)
    return im_attr, cls_attr


def im_imid_map(imagelist):
    im_imid = {}
    imid_im = {}
    with open(imagelist) as f:
        for line in f:
            l = line.rstrip('\n').split(" ")
            im_imid[ntpath.basename(l[1])] = l[0]
            imid_im[l[0]] = ntpath.basename(l[1])
    return im_imid, imid_im


def get_alphas(config, imattrlist, checkpoint_path):
    graph = tf.Graph()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    num_input = len(imattrlist[0])
    
    # load dom2alpha model 
    if config.dom2alpha_model =='linear':
        n_alphas = int(config.n_alphas)
        weights = {'out': tf.Variable(tf.random_normal([num_input, n_alphas]))}
        biases = {'out': tf.Variable(tf.random_normal([n_alphas]))}
        adam_vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'beta' in x.name]

        def neural_net(x):
            out_layer = tf.add(tf.matmul(x, weights['out']), biases['out'])
            return out_layer

        text = tf.placeholder(tf.float32, [None, int(config.n_attr)])
        out = neural_net(text)

    elif config.dom2alpha_model=='multilayer':
        n_alphas = int(config.n_alphas)
        def multilayer_perceptron(x):
            layer_1 = tf.contrib.layers.fully_connected(x, num_outputs=int(config.n_hidden_1), activation_fn=tf.nn.relu, weights_regularizer=regularizer)
            layer_2 = tf.contrib.layers.fully_connected(layer_1, num_outputs=int(config.n_hidden_2), activation_fn=tf.nn.relu, weights_regularizer=regularizer)
            out_layer = tf.contrib.layers.fully_connected(layer_2, num_outputs=n_alphas, activation_fn=None, weights_regularizer=regularizer)
            return out_layer

        # Construct model

        text = tf.placeholder(tf.float32, [None, int(config.n_attr)])
        out = multilayer_perceptron(text)

    elif config.dom2alpha_model=='2layer':
        n_alphas = int(config.n_alphas)
        def multilayer_perceptron(x):
            layer_1 = tf.contrib.layers.fully_connected(x, num_outputs=int(config.n_hidden), activation_fn=tf.nn.relu, weights_regularizer=regularizer)
            #layer_2 = tf.contrib.layers.fully_connected(layer_1, num_outputs=int(config.n_hidden_2), activation_fn=tf.nn.relu, weights_regularizer=regularizer)
            out_layer = tf.contrib.layers.fully_connected(layer_1, num_outputs=n_alphas, activation_fn=None, weights_regularizer=regularizer)
            return out_layer

        # Construct model

        text = tf.placeholder(tf.float32, [None, int(config.n_attr)])
        out = multilayer_perceptron(text)


    saver = tf.train.Saver()
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    saver.restore(sess, checkpoint_path)
    alpha_val = sess.run(out, feed_dict={text:imattrlist})

    return alpha_val



def load_class_splits(split_file, class_listf):
    # Create a mapping for the reduced classes
    # Load class splits from split_file
    class_split = scio.loadmat(split_file)
    # Get train class-IDs
    train_cid = class_split['train_cid'][0].tolist()
    test_cid = class_split['test_cid'][0].tolist()
    # Load all classes and ignore classes that are not in the seen set
    test_unseen_class = []
    for line in open(class_listf, 'r').readlines():
        classID = int(line.strip('\n').split(' ')[0])
        class_name = line.strip('\n').split(' ')[1]
        if classID in test_cid:
            test_unseen_class.append((classID-1, class_name))

    # Create mapping
    ids = sorted([x[0] for x in test_unseen_class])

    idmaps = {}
    idmaps_inv = {}
    for i in range(len(ids)):
        idmaps[ids[i]] = i
        idmaps_inv[i] = ids[i]

    idmaps_seen = {}
    idmaps_seen_inv = {}
    for i in range(len(train_cid)):
        idmaps_seen[train_cid[i] - 1] = i
        idmaps_seen_inv[i] = train_cid[i] - 1


    idmaps_all = {}
    idmaps_all_inv = {}
    for i in range(len(train_cid)+len(test_cid)):
        if i < len(train_cid):
            idmaps_all[train_cid[i] - 1] = i
            idmaps_all_inv[i] = train_cid[i] - 1
            #print(train_cid[i] -1)
        else:
            idmaps_all[ids[i-len(train_cid)]] = i
            idmaps_all_inv[i] = ids[i-len(train_cid)]
    return test_unseen_class, idmaps, idmaps_all, idmaps_seen, idmaps_inv, idmaps_all_inv, idmaps_seen_inv


def load_data(config, fname,test_unseen_class, idmaps, idmaps_all, idmaps_seen, idmaps_inv, idmaps_all_inv, idmaps_seen_inv, imagedir, imagelist, imagelabellist, train_test_split_list):


    # Check if files exist or not
    im_imid, imid_im = im_imid_map(config.imagelist)
    # Create proper train, val and test splits from the CUB dataset
    imlist = [x.strip('\n').split(' ')[1] for x in open(imagelist, 'r').readlines()]
    imidlist = [x.strip('\n').split(' ')[0] for x in open(imagelist, 'r').readlines()]
    imlabelist = [int(y.strip('\n').split(' ')[1])-1 for y in open(imagelabellist, 'r').readlines()]

    #############################################################################################################################
    # Load New imagelist
    if config.modality == 'attributes':
        im_attr, cls_attr = encode_attributes_class(config,imlabelist)
    elif config.modality == 'wikipedia':
        im_attr = encode_tfidf(config.attrdir, imlabelist, config)
    elif config.modality == 'captions':
        imlist_new = [x.strip('\n').split(' ')[1] for x in open(config.imagelist, 'r').readlines()]
        im_attr, cls_attr = encode_captions(config.attrdir, imlist_new, imlabelist, config)
    else:
        print("Modality not supported")
    # print(im_attr.keys())
    #############################################################################################################################
    imattrlist = [cls_attr[x] for x in imlabelist]
    clsattralphas = {}
    clsattralphas_list = get_alphas(config, list(cls_attr.values()), config.text2alpha_ckpt).tolist()

    for h in cls_attr:
        clsattralphas[h] = clsattralphas_list[h]


    # Remove all instances of unseen classes from the dataset
    req_im, req_im_seen, req_imclass, req_imattr, req_attralpha, req_split = [], [], [], [], [], []

    with open (config.sampling_images, 'rb') as fp:
        files = pickle.load(fp)


    shuffle(files)
    seen_im_list = files[:3000]

    unseen_test_split_file = config.unseen_test_split_file
    seen_test_split_file = config.seen_test_split_file

    with open(unseen_test_split_file,'r') as fj:
        data  = json.load(fj)
    for d in data:
        req_im.append(d[0])
        req_imclass.append(int(d[1]))
        req_imattr.append(cls_attr[int(d[1])])

        req_attralpha.append(clsattralphas[idmaps_inv[int(d[1])]])
        req_im_seen.append(seen_im_list[random.randint(0,len(seen_im_list)-1)])
    req_dataset = list(zip(req_im, req_im_seen, req_imclass, req_imattr, req_attralpha))

    shuffle(req_dataset)
    train_split = req_dataset
    print("size of train dataset", len(train_split))

    req_dataset = list(zip(req_im,req_im_seen, req_imclass, req_imattr, req_attralpha))
    val_split = req_dataset

    train_split = [(x[0], x[1], x[2], x[3], x[4]) for x in train_split]
    val_split = [(x[0], x[1], x[2], x[3], x[4]) for x in val_split]

    print("size of val dataset", len(val_split))

    # Create test split seen classes --> for debugging
    req_im, req_imclass, req_imattr, req_attralpha, req_split = [], [], [], [], []

    with open(seen_test_split_file,'r') as fj:
        data  = json.load(fj)
    for d in data:
        req_im.append(d[0])
        req_imclass.append(int(d[1]))
        req_imattr.append(cls_attr[int(d[1])])
        req_attralpha.append(clsattralphas[idmaps_seen_inv[int(d[1])]])
        #req_im_seen.append(seen_im_list[random.randint(0,len(seen_im_list)-1)])
        req_split.append(0)

    whole_dataset = list(zip(req_im, req_imclass, req_attralpha, req_split))

    train_split_seen = [(x[0], x[1], x[2]) for x in whole_dataset if x[3]==0]



    # Create test split for all classes --> generalized zsl

    req_im, req_imclass, req_imattr, req_attralpha, req_split = [], [], [], [], []

    with open(seen_test_split_file,'r') as fj:
        data  = json.load(fj)
    for d in data:
        req_im.append(d[0])
        req_imclass.append(int(d[1]))
        req_imattr.append(cls_attr[int(d[1])])
        req_attralpha.append(clsattralphas[idmaps_seen_inv[int(d[1])]])
        #req_im_seen.append(seen_im_list[random.randint(0,len(seen_im_list)-1)])
        req_split.append(0)


    with open(unseen_test_split_file,'r') as fj:
        data  = json.load(fj)

    for d in data:
        req_im.append(d[0])
        req_imclass.append(int(d[1])+int(config.n_seen))
        req_imattr.append(cls_attr[int(d[1])])
        req_attralpha.append(clsattralphas[idmaps_inv[int(d[1])]])
        #req_im_seen.append(seen_im_list[random.randint(0,len(seen_im_list)-1)])
        req_split.append(0)

    print("test dataset: number of images from both seen and unseen classes: ", len(req_im))

    whole_dataset = list(zip(req_im, req_imclass, req_attralpha, req_split))

    test_split = [(x[0], x[1], x[2]) for x in whole_dataset if x[3]==0]


    req_im, req_imclass, req_imattr, req_attralpha, req_split = [], [], [], [], []

    with open(seen_test_split_file,'r') as fj:
        data  = json.load(fj)
    for d in data:
        req_im.append(d[0])
        req_imclass.append(int(d[1]))
        req_attralpha.append(clsattralphas[idmaps_seen_inv[int(d[1])]])
        #req_im_seen.append(seen_im_list[random.randint(0,len(seen_im_list)-1)])
        req_split.append(0)

    whole_dataset = list(zip(req_im, req_imclass, req_attralpha, req_split))

    test_split_seen = [(x[0], x[1], x[2]) for x in whole_dataset if x[3]==0]

    req_im, req_imclass, req_imattr, req_attralpha, req_split = [], [], [], [], []
    with open(unseen_test_split_file,'r') as fj:
        data  = json.load(fj)
    for d in data:
        req_im.append(d[0])
        req_imclass.append(int(d[1])+int(config.n_seen))
        req_attralpha.append(clsattralphas[idmaps_inv[int(d[1])]])
        #req_im_seen.append(seen_im_list[random.randint(0,len(seen_im_list)-1)])
        req_split.append(0)

    whole_dataset = list(zip(req_im, req_imclass, req_attralpha, req_split))

    test_split_unseen = [(x[0], x[1], x[2]) for x in whole_dataset if x[3]==0]

    print("length of whole test dataset: ", len(test_split))
    print("length of whole test_seen dataset: ", len(test_split_seen))
    print("length of whole test_unseen dataset: ", len(test_split_unseen))


    return train_split, val_split, test_split, test_split_seen, test_split_unseen, train_split_seen


def check_accuracy(sess, prediction, imclass, accuracy, is_training, dataset_init_op, verbose=False):
    # Check accuracy on train or val
    # Initialize the dataset
    sess.run(dataset_init_op)
    #num_correct, num_samples = 0, 0
    acc_list = []
    while True:
        try:
            acc, prediction_val, imclass_val= sess.run([accuracy, prediction, imclass], {is_training: False})
            if verbose:
                print("pred:  ", prediction_val)
                print("gt cls:", imclass_val)
            acc_list.append(acc)
        except tf.errors.OutOfRangeError:
            break
    final_acc = np.mean(np.array(acc_list))
    #return float(num_correct)/num_samples
    return final_acc


def check_accuracy_normalized(sess, prediction, imclass, accuracy, is_training, dataset_init_op, verbose=False):
    # Check accuracy on train or val
    # Initialize the dataset
    sess.run(dataset_init_op)
    acc_list = []
    whole_pred_list = []
    whole_label_list = []
    while True:
        try:
            acc, prediction_val, imclass_val= sess.run([accuracy, prediction, imclass], {is_training: False})
            if verbose:
                print("pred:  ", prediction_val)
                print("gt cls:", imclass_val)
            acc_list.append(acc)
            # Get of unique predictions
            if isinstance(prediction_val,int):
                prediction_val = np.array([prediction_val])
            if isinstance(imclass_val,int):
                imclass_val = np.array([imclass_val])
            try:
                whole_pred_list += prediction_val.tolist()
                whole_label_list += imclass_val.tolist()
            except TypeError:
                break
        except tf.errors.OutOfRangeError:
            break

    final_acc = np.mean(np.array(acc_list))
    # Get unique classes
    unique_cls = list(set(whole_label_list))
    # Find incices corresponding to a class
    all_cls_acc = []
    for y in unique_cls:
        gt_indices = [i for i,x in enumerate(whole_label_list) if x == y]
        acc_clas = float([whole_pred_list[i] for i in gt_indices].count(y))/len(gt_indices)
        all_cls_acc.append(acc_clas)
    # print("pred  list:", whole_pred_list)
    # print("label list:", whole_label_list)
    return np.mean(all_cls_acc)
    # return final_acc

def main(args):

    # Load config JSON and use the arguments
    config = parse_json(args.config_json)
    pprint(config)
    config = DotMap(config)
    print("loading class splits ..")
    test_unseen_class, idmaps, idmaps_all, idmaps_seen, idmaps_inv, idmaps_all_inv, idmaps_seen_inv = load_class_splits(config.split_file, config.class_listf)
    #test_unseen_class, idmaps, idmaps_all, idmaps_seen = load_class_splits(config.split_file, config.class_listf)

    # Load the dataset splits
    print('Loading data...')
    # train_split, val_split, test_split, test_split_seen, test_split_unseen, train_split_seen = load_data(config, config.save_path.split('/')[0], test_unseen_class, idmaps, idmaps_all, idmaps_seen, config.imagedir, config.imagelist, config.imagelabellist, config.train_test_split_list, float(config.train_prop), float(config.val_prop), float(config.test_prop))

    train_split, val_split, test_split, test_split_seen, test_split_unseen, train_split_seen = load_data(config, config.save_path.split('/')[0], test_unseen_class, idmaps, idmaps_all, idmaps_seen, idmaps_inv, idmaps_all_inv, idmaps_seen_inv, config.imagedir, config.imagelist, config.imagelabellist, config.train_test_split_list)


    train_files,seen_train_files, train_imclass, train_imattr, train_attralpha= map(list, zip(*train_split))
    val_files,seen_val_files, val_imclass, val_imattr, val_attralpha = map(list, zip(*val_split))
    test_files, test_labels, test_attralpha = map(list, zip(*test_split))
    test_files_seen, test_labels_seen, test_attralpha_seen = map(list, zip(*test_split_seen))
    train_files_seen, train_labels_seen, train_attralpha_seen = map(list, zip(*train_split_seen))
    test_files_unseen, test_labels_unseen, test_attralpha_unseen = map(list, zip(*test_split_unseen))

    train_imclass = np.array(train_imclass).astype('int32')
    val_imclass = np.array(val_imclass).astype('int32')
    test_labels = np.array(test_labels).astype('int32')
    test_labels_seen = np.array(test_labels_seen).astype('int32')
    test_labels_unseen = np.array(test_labels_unseen).astype('int32')

    # Write graph definition based on model name
    # Define the computation graph with necessary functions
    graph = tf.Graph()
    with graph.as_default():
        # Preprocessing function and module import
        preprocess_module_name = 'preprocessing.' + config.preprocess_fn
        preprocess_module = importlib.import_module(preprocess_module_name)

        # Get image size
        mc = getattr(nets, config.model_class, None)
        m = getattr(mc, config.model_name, None)
        #im_size = getattr(m, 'default_image_size', None)
        im_size = int(config.image_size)

        # Parsing an pre-processing function
        def _parse_function_seen(seen_filename, imclass, attralpha):
            image_f = tf.read_file(seen_filename)
            image_dec = tf.image.decode_jpeg(image_f, channels=3)
            image = tf.cast(image_dec, tf.float32)
            # Resize image
            res_img = tf.image.resize_images(image, [im_size, im_size])
            # attralpha_noise = tf.random_normal([int(config.n_alphas)])
            return res_img, imclass, attralpha

        # Substitute for parse_function seen+ unseen uncluding preprocessing
        def parse_fn_train(filename, imclass, attralpha):
            image_file = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_file, channels=3)
            processed_image = preprocess_module.preprocess_image(image, im_size, im_size, is_training=False)
            return processed_image, imclass, attralpha

        def _parse_function_noise(seen_filename, imclass, attralpha):
            image_f = tf.read_file(seen_filename)
            image_dec = tf.image.decode_jpeg(image_f, channels=3)
            image = tf.cast(image_dec, tf.float32)
            # Resize image
            res_img = tf.image.resize_images(image, [im_size, im_size])
            img = tf.random_normal([im_size, im_size, 3], mean=0, stddev=1.0)
            img = tf.div(tf.subtract(img, tf.reduce_min(img)), tf.subtract(tf.reduce_max(img), tf.reduce_min(img)))
            img = tf.cast(img*255.0, tf.int32)
            res_img_noise = tf.cast(img, tf.float32)
            res_img = res_img_noise
            return res_img, imclass, attralpha


        # Parsing an pre-processing function
        def _parse_function_val(filename, imclass, attralpha):
            image_f = tf.read_file(filename)
            image_dec = tf.image.decode_jpeg(image_f, channels=3)
            image = tf.cast(image_dec, tf.float32)
            # Resize image
            res_img = tf.image.resize_images(image, [im_size, im_size])
            return res_img, imclass, attralpha

        # preprocessing function
        def prepro(image, imclass, attralpha):
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            proc_img = image - means
            return proc_img, imclass, attralpha


        # Dataset creation
        # Training dataset
        print("Creating datasets..")
        train_files = tf.constant(train_files)
        seen_train_files = tf.constant(seen_train_files)
        train_imclass = tf.constant(train_imclass)
        train_attralpha = tf.constant(train_attralpha)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((seen_train_files, train_imclass, train_attralpha))

        if config.prepro == 'unified':
            train_dataset = train_dataset.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size = int(config.batch_size))
        else:
            if config.sampling_mode =='noise':
                train_dataset = train_dataset.map(_parse_function_noise, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
                train_dataset = train_dataset.map(prepro, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
            else:
                train_dataset = train_dataset.map(_parse_function_seen, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
                train_dataset = train_dataset.map(prepro, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_split))
        batched_train_dataset = train_dataset.batch(int(config.batch_size))

        # Validation dataset
        val_files = tf.constant(val_files)
        val_imclass = tf.constant(val_imclass)
        val_attralpha = tf.constant(val_attralpha)

        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_files, val_imclass, val_attralpha))

        if config.prepro == 'unified':
            val_dataset = val_dataset.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        else:
            val_dataset = val_dataset.map(_parse_function_val, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
            val_dataset = val_dataset.map(prepro, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        batched_val_dataset = val_dataset.batch(int(config.batch_size))

        # Test dataset
        test_files = tf.constant(test_files)
        test_labels = tf.constant(test_labels)
        test_attralpha = tf.constant(test_attralpha)
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_files, test_labels, test_attralpha))
        if config.prepro == 'unified':
            test_dataset = test_dataset.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        else:
            test_dataset = test_dataset.map(_parse_function_val, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
            test_dataset = test_dataset.map(prepro, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        batched_test_dataset = test_dataset.batch(int(config.batch_size))

        test_files_seen = tf.constant(test_files_seen)
        test_labels_seen = tf.constant(test_labels_seen)
        test_attralpha_seen = tf.constant(test_attralpha_seen)
        test_dataset_seen = tf.contrib.data.Dataset.from_tensor_slices((test_files_seen, test_labels_seen, test_attralpha_seen))
        if config.prepro =='unified':
            test_dataset_seen = test_dataset_seen.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        else:
            test_dataset_seen = test_dataset_seen.map(_parse_function_val, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
            test_dataset_seen = test_dataset_seen.map(prepro, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        batched_test_dataset_seen = test_dataset_seen.batch(int(config.batch_size))

        # Test dataset unseen
        test_files_unseen = tf.constant(test_files_unseen)
        test_labels_unseen = tf.constant(test_labels_unseen)
        test_attralpha_unseen = tf.constant(test_attralpha_unseen)
        test_dataset_unseen = tf.contrib.data.Dataset.from_tensor_slices((test_files_unseen, test_labels_unseen, test_attralpha_unseen))
        # test_dataset_unseen = test_dataset_unseen.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        if config.prepro =='unified':
            test_dataset_unseen = test_dataset_unseen.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        else:
            test_dataset_unseen = test_dataset_unseen.map(_parse_function_val, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
            test_dataset_unseen = test_dataset_unseen.map(prepro, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        batched_test_dataset_unseen = test_dataset_unseen.batch(int(config.batch_size))

        # Train dataset seen
        train_files_seen = tf.constant(train_files_seen)
        train_labels_seen = tf.constant(train_labels_seen)
        train_attralpha_seen = tf.constant(train_attralpha_seen)
        train_dataset_seen = tf.contrib.data.Dataset.from_tensor_slices((train_files_seen, train_labels_seen, train_attralpha_seen))
        #train_dataset_seen = train_dataset_seen.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        if config.prepro =='unified':
            train_dataset_seen = train_dataset_seen.map(parse_fn_train, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        else:
            train_dataset_seen = train_dataset_seen.map(_parse_function_val, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
            train_dataset_seen = train_dataset_seen.map(prepro, num_threads=int(config.batch_size), output_buffer_size=int(config.batch_size))
        batched_train_dataset_seen = train_dataset_seen.batch(int(config.batch_size))


        # Define iterator that operates on either of the splits
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
        images, imclass, attralpha = iterator.get_next()
        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)
        test_init_op = iterator.make_initializer(batched_test_dataset)
        test_seen_init_op = iterator.make_initializer(batched_test_dataset_seen)
        train_seen_init_op = iterator.make_initializer(batched_train_dataset_seen)
        test_unseen_init_op = iterator.make_initializer(batched_test_dataset_unseen)

        # Boolean variable for train-vs-test
        is_training = tf.placeholder(tf.bool)

        # Define the global step to be some tf.Variable
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        model_c = getattr(nets, config.model_class, None)
        model = getattr(model_c, config.model_name, None)

        arg_scope = getattr(model_c, config.scope, None)

        # Get number of classes in train and test
        n_seen = int(config.n_seen)
        n_unseen = int(config.n_unseen)
        print("Defining model from ..  ", model_c )
        with slim.arg_scope(arg_scope(weight_decay=float(0))):
            print("--------------------------Using original network---------------------------------------------------------------")
    # with slim.arg_scope(arg_scope()):
            if config.base_model == 'resnet':
                logits, endpoints = model(images, num_classes=n_seen, is_training=is_training)
            else:
                logits, endpoints = model(images, num_classes=n_seen, is_training=is_training, dropout_keep_prob=bool(config.dropout))

        if config.base_model == 'resnet':
            fc8_seen_weights = tf.contrib.framework.get_variables(config.penultimate_seen_weights)
            fc8_seen_biases = tf.contrib.framework.get_variables('resnet_v1_101/logits/biases:0')
        else:
            fc8_seen_weights = tf.contrib.framework.get_variables('vgg_16/fc8/weights:0')
            fc8_seen_biases = tf.contrib.framework.get_variables('vgg_16/fc8/biases:0')

        # Check for model path
        # assert(os.path.isfile(config.ckpt_path))

        if config.base_model =='resnet':
            if config.ckpt == 'old':
                orig_ckpt = config.orig_ckpt_path
                orig_ckpt_reader = pywrap_tensorflow.NewCheckpointReader(orig_ckpt)
                new_ckpt_reader = pywrap_tensorflow.NewCheckpointReader(config.ckpt_path)
                new_var_to_shape_map = new_ckpt_reader.get_variable_to_shape_map()
                orig_var_to_shape_map = orig_ckpt_reader.get_variable_to_shape_map()
                vars_in_orig_ckpt = [key for key in sorted(orig_var_to_shape_map)]
                vars_in_new_ckpt = [key for key in sorted(new_var_to_shape_map)]
                vars_in_graph = [x.name.split(':')[0] for x in tf.contrib.framework.get_variables()]
                
                # Variables to borrow from old ckpt
                vars_to_borrow = list(set(list(set(vars_in_graph) - set(vars_in_new_ckpt))) & set(vars_in_orig_ckpt))
                
                # Variables to initialize
                # vars_to_init = list(set(vars_in_graph) - set(vars_to_borrow))
                vars_to_init = list(set(vars_in_graph) - set(vars_to_borrow + vars_in_new_ckpt))
                
                # Old ckpt init function
                old_ckpt_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(orig_ckpt, [x for x in tf.contrib.framework.get_variables() if (x.name.split(':')[0] in vars_to_borrow) and ('global_step' not in x.name)])
                # New ckpt init function
                new_ckpt_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.ckpt_path, [x for x in tf.contrib.framework.get_variables() if x.name.split(':')[0] in vars_in_new_ckpt])
            else:
                new_ckpt_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.ckpt_path, tf.contrib.framework.get_variables_to_restore(exclude=['global_step']))

            var_init = tf.variables_initializer([global_step_tensor])
            # get seen weights and initialize new layer with mean of them

            sess1 = tf.Session()
            new_ckpt_init_fn(sess1)
            fc8_seen_weights_value = sess1.run(fc8_seen_weights)[0]
            fc8_seen_biases_value = sess1.run(fc8_seen_biases)[0]

            fc8_seen_weights_mean = fc8_seen_weights_value.mean(axis=3)
            fc8_seen_biases_mean = fc8_seen_biases_value.mean(axis=0)

            fc8_seen_weights_init = np.repeat(fc8_seen_weights_mean, n_unseen, axis=2)
            fc8_seen_biases_init = np.repeat(fc8_seen_biases_mean, n_unseen)

            logits = tf.squeeze(logits)
            # Add a new head
            if config.unseen_w_init == 'seen_centered':
                # Initialize by a gaussian centered on the seen class weights
                mean_seen_wt = tf.reduce_mean(tf.squeeze(fc8_seen_weights), axis=1, keep_dims=True)
                std_seen_wt = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(fc8_seen_weights) - mean_seen_wt), axis=1))
                mean_wt = tf.tile(mean_seen_wt, [1, n_unseen])
                std_wt = tf.tile(tf.expand_dims(std_seen_wt, axis=1), [1, n_unseen])
                w_init = tf.random_normal_initializer(mean_wt, std_wt)
                logits_unseen = slim.conv2d(endpoints['global_pool'], n_unseen, [1,1], activation_fn = None, normalizer_fn = None, scope='logits_unseen', weights_initializer = w_init)
            else:
            	logits_unseen = slim.conv2d(endpoints['global_pool'], n_unseen, [1,1], activation_fn = None, normalizer_fn = None, scope='logits_unseen', weights_initializer = tf.constant_initializer(fc8_seen_weights_init))

            logits_seen = slim.conv2d(endpoints['global_pool'], n_seen, [1,1], activation_fn = None, normalizer_fn = None, scope='logits_seen', weights_initializer = tf.constant_initializer(fc8_seen_weights_value))
            logits = array_ops.squeeze(logits_seen, [1,2])
            logits_unseen = array_ops.squeeze(logits_unseen, [1,2])

        else:
            var_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['global_step'])
            print("Using base model checkpoint from: ", config.ckpt_path)
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.ckpt_path, var_to_restore)

            if config.unseen_w_init == 'seen_centered':
                print("Seen centered initializaton of unseen weights")
                # Initialize by a gaussian centered on the seen class weights
                mean_seen_wt = tf.reduce_mean(tf.squeeze(fc8_seen_weights), axis=1, keep_dims=True)
                std_seen_wt = tf.sqrt(tf.reduce_mean(tf.square(tf.squeeze(fc8_seen_weights) - mean_seen_wt), axis=1))
                mean_wt = tf.tile(mean_seen_wt, [1, n_unseen])
                std_wt = tf.tile(tf.expand_dims(std_seen_wt, axis=1), [1, n_unseen])
                w_init = tf.random_normal_initializer(mean_wt, std_wt)
                logits_unseen = array_ops.squeeze(tf.contrib.layers.fully_connected(inputs=endpoints['vgg_16/fc7'], num_outputs=n_unseen, activation_fn=None, weights_initializer = w_init), [1,2], name = 'fc8_unseen')
            else:
                logits_unseen = array_ops.squeeze(tf.contrib.layers.fully_connected(inputs=endpoints['vgg_16/fc7'], num_outputs=n_unseen, activation_fn=None), [1,2], name = 'fc8_unseen')

        # Evaluation Metrics for seen classes
        prediction_seen = tf.to_int32(tf.argmax(logits, -1))
        prediction_seen  = tf.squeeze(prediction_seen )
        imclass = tf.squeeze(imclass)
        correct_prediction_seen  = tf.equal(prediction_seen , imclass)
        accuracy_seen  = tf.reduce_mean(tf.cast(correct_prediction_seen , tf.float32))
        logits_seen_unseen = tf.concat([logits, logits_unseen],1)

        sys.stdout.flush()
        
        
        # Evaluation Metrics
        prediction = tf.to_int32(tf.argmax(logits_unseen, -1))
        prediction = tf.squeeze(prediction)
        imclass = tf.squeeze(imclass)
        correct_prediction = tf.equal(prediction, imclass)
        accuracy_unseen = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        # Generalized ZSL performance
        prediction_seen_unseen = tf.to_int32(tf.argmax(logits_seen_unseen,-1))
        correct_prediction_seen_unseen = tf.equal(prediction_seen_unseen, imclass)
        accuracy_seen_unseen = tf.reduce_mean(tf.cast(correct_prediction_seen_unseen, tf.float32))

        # ----------------------------------Optimization starts here---------------------------------------

        # Define one-hot for seen class
        one_hot_seen = tf.one_hot(imclass, n_seen, 1.0)
        signal_seen = tf.multiply(logits, one_hot_seen)

        # Define how to get alphas
        layer_name = config.model_name + '/' + config.layer_name
        grads_seen = tf.gradients(signal_seen, endpoints[layer_name])

        # Get alphas
        alphas_seen = tf.reduce_sum(grads_seen[0], [1,2])

        # Define one-hot for unseen class
        one_hot_unseen = tf.one_hot(imclass, n_unseen, 1.0)
        signal_unseen = tf.multiply(logits_unseen, one_hot_unseen)

        # Define how to get alphas
        layer_name = config.model_name + '/' + config.layer_name
        grads_unseen = tf.gradients(signal_unseen, endpoints[layer_name])[0]

        # Get alphas
        alphas_unseen = tf.reduce_sum(grads_unseen, [1,2])

        # Regularization coefficient
        lambda_loss = float(config.reg_lambda)

        attr_alpha_normalized = tf.nn.l2_normalize(attralpha, 1)
        alphas_unseen_normalized = tf.nn.l2_normalize(alphas_unseen, 1)

        # Cosine distance loss, assumes that both inputs are normalized
        def binary_activation(x):
            cond = tf.less(x, tf.zeros(tf.shape(x)))
            out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
            return out

        # Loss between network alphas and the predicted importances from domain expert --> alpha model
        if config.alpha_loss_type =="cd":
            zsl_alpha_loss = tf.reduce_mean(tf.losses.cosine_distance(attr_alpha_normalized, alphas_unseen_normalized, dim=1, reduction=tf.losses.Reduction.NONE))

        # Define the optimizers
        if config.optimizer =='adam':
            optimizer = tf.train.AdamOptimizer(float(config.learning_rate))
        if config.optimizer =='sgd':
            optimizer = tf.train.GradientDescentOptimizer(float(config.learning_rate))
        if config.optimizer =='sgd_momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate = float(config.learning_rate) , momentum = 0.9)

        # get the newly initialized vars (for the unseen head)
        if config.base_model =='resnet':
            new_var = [v for v in tf.trainable_variables() if v.name == 'logits_unseen/weights:0' or v.name =='logits_unseen/biases:0' or v.name == 'logits_seen/weights:0' or v.name =='logits_seen/biases:0']
        else:
            new_var = [v for v in tf.trainable_variables() if v.name == 'fully_connected/weights:0' or v.name =='fully_connected/biases:0']#tf.contrib.framework.get_variables('logits_unseen')

        # Regularizer term
        if config.reg_loss == 'dimn_wise_l2':
            zsl_reg_loss = tf.nn.l2_loss(tf.squeeze(new_var[0]) - tf.expand_dims(tf.reduce_mean(tf.squeeze(fc8_seen_weights), axis=1), axis=1))
        
        
        # Total loss is sum of loss and lambda times reg term
        zsl_loss = zsl_alpha_loss + lambda_loss * zsl_reg_loss
        weights_unseen_grad = tf.gradients(alphas_unseen, new_var[0])[0]

        # define training op with the parameters to be optimized (unseen head params and global step)
        new_train_op = tf.contrib.slim.learning.create_train_op(zsl_loss, optimizer, variables_to_train=new_var)
        new_vars_with_adam = new_var + [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'beta' in x.name or 'Adam' in x.name or 'global_step' in x.name]
        new_vars_with_adam_momentum = new_var + [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Momentum' in x.name ]

        # Define the global step to be some tf.Variable
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
        new_init = tf.variables_initializer(new_vars_with_adam + new_vars_with_adam_momentum + [global_step_tensor])

        print("Graph finalized")
        
        # save accuracies as trainig proceeds 
        seen_seen_head_ls, unseen_unseen_head_ls, seen_seen_unseen_head_ls, seen_unseen_seen_unseen_head_ls, unseen_seen_unseen_head_ls, hm_ls = [],[],[],[],[], []
        vLoss = []
        print("Running simple forward pass for getting initial performance ..")
        with tf.Session(graph=graph) as sess:
            if config.base_model=='resnet':
                if config.ckpt=='old':
                    new_ckpt_init_fn(sess)
                    old_ckpt_init_fn(sess)
                else:
                    new_ckpt_init_fn(sess)
                sess.run(var_init)
            else:
                init_fn(sess)
            sess.run(new_init)

            val_loss_best = 1e20
            iteration = 1

            # Calculate seen head accuracy on the test set
            seen_seen_head = check_accuracy_normalized(sess, prediction_seen,imclass, accuracy_seen, is_training, train_seen_init_op, verbose=False)
            print("seen head test accuracy (argmax {}):   {}".format(config.n_seen, seen_seen_head))

            test_accuracy = check_accuracy_normalized(sess, prediction_seen_unseen,imclass, accuracy_seen_unseen, is_training, test_init_op, verbose=False)
            print("Normalized Initial seen + unseen head full test accuracy: {}".format(test_accuracy))
           
            # Define criterion for early stopping (loss doesn't improve by 1% in 40 iterations)
            zsl_loss_monitor = 1e20
            zsl_loss_ctr = 1
            zsl_loss_window = int(config.zsl_loss_estop_window)
            
            
            # Start training
            print("Starting Optimization .....")
            epoch_flag = False
            for epoch in range(1, int(config.num_epochs)):
                if epoch_flag:
                    break
                m = 0
                sys.stdout.flush()
                sess.run(train_init_op)
                loss_list = []
                loss_alpha_list = []
                loss_reg_list = []
                while True:
                    try:
                        l, zsl_alpha, zsl_reg, zsl_total = sess.run([new_train_op, zsl_alpha_loss, zsl_reg_loss, zsl_loss], {is_training:False})
                        if zsl_loss_ctr >= zsl_loss_window:
                            epoch_flag = True
                            print('Breaking out of optimization split\n\n')
                            break
                        if iteration == 1:
                            zsl_loss_monitor = zsl_total
                        else:
                            if (1 - (zsl_total/zsl_loss_monitor)) > float(config.eps_perc):
                                zsl_loss_ctr = 0
                                zsl_loss_monitor = zsl_total
                            else:
                                zsl_loss_ctr += 1
                        loss_list.append(l)
                        loss_alpha_list.append(zsl_alpha)
                        loss_reg_list.append(zsl_reg)
                        iteration +=1
                    except tf.errors.OutOfRangeError:
                        break
                valLoss = np.mean(np.array(loss_list))
                print("Epoch {}, average_training_loss_alpha: {}".format(epoch, np.mean(np.array(loss_alpha_list))))
                print("Epoch {}, average_training_loss_reg  : {}".format(epoch, np.mean(np.array(loss_reg_list))))
                print("Epoch {}, average_training_loss: {}".format(epoch, valLoss))

                # Compute accuracy
                seen_seen_head_ls.append(seen_seen_head)

                unseen_unseen_head = check_accuracy_normalized(sess, prediction,imclass, accuracy_unseen, is_training, val_init_op)
                print("unseen head test accuracy (argmax {}):   {}".format(config.n_unseen, unseen_unseen_head))
                unseen_unseen_head_ls.append(unseen_unseen_head)

                seen_seen_unseen_head = check_accuracy_normalized(sess, prediction_seen_unseen, imclass, accuracy_seen_unseen, is_training, test_seen_init_op)
                print("seen head full test accuracy: (argmax {}): {}".format(config.n_class, seen_seen_unseen_head))
                seen_seen_unseen_head_ls.append(seen_seen_unseen_head)
               
                seen_unseen_seen_unseen_head = check_accuracy_normalized(sess, prediction_seen_unseen, imclass, accuracy_seen_unseen, is_training, test_init_op)
                print("seen + unseen head full test accuracy: (argmax {}): {}".format(config.n_class, seen_unseen_seen_unseen_head))
                seen_unseen_seen_unseen_head_ls.append(seen_unseen_seen_unseen_head)

                unseen_seen_unseen_head = check_accuracy_normalized(sess, prediction_seen_unseen, imclass, accuracy_seen_unseen, is_training, test_unseen_init_op, verbose=False)
                print("unseen head full test accuracy: (argmax {}):  {}".format(config.n_class, unseen_seen_unseen_head))
                unseen_seen_unseen_head_ls.append(unseen_seen_unseen_head)

                # Compute Harmonic Mean of seen accuracies and unseen accuracies.
                H = 2*seen_seen_unseen_head * unseen_seen_unseen_head/(seen_seen_unseen_head+ unseen_seen_unseen_head)
                print("Harmonic mean", H)

                hm_ls.append(H)

                if valLoss <= val_loss_best:
                    val_loss_best = valLoss
                    checkpoint_dir = config.ckpt_dir + '{}_{}_{}_cnn_seen_val{}_alpha_loss_{}_d2a_model_{}_bs_{}_lr_{}_lambda_{}_epoch_{}_ssu_{:0.2f}_usu_{:0.3f}_h_{:0.3f}.ckpt'.format(config.dataset, config.model_name, config.modality, config.n_unseen, config.alpha_loss_type, config.dom2alpha_model, config.batch_size, config.learning_rate, config.reg_lambda, epoch, seen_seen_unseen_head, unseen_seen_unseen_head, H)
                    saver.save(sess, checkpoint_dir)
                    print("saved_checkpoint to {}".format(checkpoint_dir))

                uhead_plotter(seen_seen_head_ls, unseen_unseen_head_ls, seen_unseen_seen_unseen_head_ls, seen_seen_unseen_head_ls, unseen_seen_unseen_head_ls, hm_ls, config.ckpt_dir, 'Normalized_Accuracy_logs')

            sys.stdout.flush()
        print("Optimization Done")
        print("Best Checkpoint: ", checkpoint_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
