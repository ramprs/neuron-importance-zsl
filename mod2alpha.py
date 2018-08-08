# Code to map from any modality to alphas.
# Train using class_info and alphas from a trained network
import argparse
import numpy as np
import random
random.seed(1234)
from random import shuffle
import pickle
from pprint import pprint
from dotmap import DotMap
import pdb
import csv
import os
import json
import tensorflow as tf
from scipy.io import loadmat
import ntpath
from scipy.stats import spearmanr
import glob
from tqdm import tqdm
import torchfile
import scipy.io as scio

# Fix CUB names due to mismatch in Scott Reed's caption dataset
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
parser.add_argument('--config_json', default='')

def encode_attributes_class(attrdir, imlabelist, config):
    im_attr = {}
    cls_attr = {}
    # Use class level supervision
    if config.supervision=='class':
        class_att_labels = []
        with open(attrdir) as f:
            for n, line in enumerate(f):
                l = [x for x in line.rstrip().split(" ") ]
                l = [x for x in l if x]
                #l = l.remove('')
                if config.a2t=="True":
                    l = [int(x) if float(x)!=-1.00 else 0 for x in l ]
                else:
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


def parse_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


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
    print('Loading feature files..')
    for i in class_names:
        fname = i
        if fname in list(CUB_FNAME_FIX.keys()):
            fname = CUB_FNAME_FIX[fname]
        t7_dict[i] = torchfile.load(feat_dir + '/' + fname + '.t7')

    im_attr = {}
    # Do this iteratively
    print('Encoding captions...')
    for i in tqdm(range(len(imlist_new))):
        imname = imlist_new[i]
        # Image name to class-t7 file
        class_name = imname.split('/')[0]
        data = t7_dict[class_name]
        imind = all_f.index(imname)
        indlist = sorted([all_f.index(x) for x in all_f if class_name in x])
        pos = indlist.index(imind)
        feat = data[pos].T
        im_attr[str(i+1)] = {}
        im_attr[str(i+1)]['att'] = np.mean(feat, axis=0).tolist()
    return im_attr


def im_imid_map(imgmap):
    im_imid = {}
    with open(imgmap) as f:
        for line in f:
            l = line.rstrip('\n').split(" ")
            im_imid[ntpath.basename(l[1])] = l[0]
    return im_imid


def load_class_splits(config):
    split_file = config.split_file
    class_listf= config.class_listf
    # Create a mapping for the reduced classes
    # Load class splits from split_file
    class_split = loadmat(split_file)

    # Get train class-IDs
    train_cid = class_split['train_cid'][0].tolist()
    test_cid = class_split['test_cid'][0].tolist()
    # Load all classes and ignore classes that are not in the seen set
    train_seen_class = []
    val_seen_class = []
    for line in open(class_listf, 'r').readlines():
        classID = int(line.strip('\n').split(' ')[0])
        class_name = line.strip('\n').split(' ')[1]
        if classID in train_cid:
            train_seen_class.append((classID-1, class_name))

    # Split train classes into train and val
    random.shuffle(train_seen_class)
    train_seen_class_split = train_seen_class[:int(config.n_train_class)]
    val_seen_class_split = train_seen_class[int(config.n_train_class):]

    return train_seen_class_split, val_seen_class_split


def create_splits(config):
    # Create proper train,val and test splits from CUB alphas dataset
    im_imid = im_imid_map(config.imagelist)
    train_seen_class, val_seen_class= load_class_splits(config)

    imlist_new = [x.strip('\n').split(' ')[1] for x in open('./data/CUB/images.txt', 'r').readlines()]
    imlabelist = [int(y.strip('\n').split(' ')[1])-1 for y in open(config.imagelabellist, 'r').readlines()]

    # modality specific data loader
    if config.modality == 'attributes':
        if config.a2t == 'True':
            im_attr,cls_attr = encode_attributes_class(config.classattrdir_binary,imlabelist, config)
        else:
            im_attr,cls_attr = encode_attributes_class(config.classattrdir,imlabelist, config)
    elif config.modality == 'wikipedia':
        im_attr = encode_tfidf(config.attrdir, imlabelist, config)
    elif config.modality == 'captions':
        cls_attr = encode_captions(config.attrdir, imlist_new, imlabelist, config)
    else:
        print("Modality not supported")

    imlist = []
    imlabellist = []
    imattrlist = []
    imalphaslist = []
    train_val_split = []
    trainval_alphadir = config.alphadir.format(config.dataset, config.dataset, config.alpha_model_name, int(config.n_seen_train), config.alpha_layer_name)
    print('alphadir: ', trainval_alphadir)
    for filename in tqdm(os.listdir(trainval_alphadir)):
        if filename.endswith(".json") :
            with open(trainval_alphadir+filename,'r') as fj:
                data  = json.load(fj)
            image = ntpath.basename(data['image'])
            image_id = im_imid[str(image)]
            imlist.append(image_id)
            gt_class = int(data['gt_cid'])
            if config.modality =='attributes':
                attr = cls_attr[gt_class]
            elif config.modality == 'captions':
                attr = cls_attr[str(gt_class+1)]['att']
            imattrlist.append(attr)
            # Train on all train and val attributes and test on test attributes
            # train = 1
            if gt_class in [x for (x,_) in train_seen_class]: 
                train = 1
            else:
                train = 0
            imlabellist.append(gt_class)
            gt_class_alpha = data['gt_alpha']
            imalphaslist.append(gt_class_alpha)
            train_val_split.append(train)


    whole_dataset = list(zip(imlist, imlabellist, imattrlist, imalphaslist, train_val_split))
    train_split = [x for x in whole_dataset if x[4] == 1]
    val_split = [x for x in whole_dataset if x[4] == 0]

    shuffle(train_split)

    # split into train and validation set
    train_split = [(x[0], x[1],x[2], x[3]) for x in train_split]
    val_split = [(x[0], x[1],x[2], x[3]) for x in val_split]
    print('#train_instances: %d', len(train_split))
    print('#val_instances: %d', len(val_split))


    return train_split, val_split


def main(args):
    # Load config JSON and use the arguments
    config = parse_json(args.config_json)
    pprint(config)
    config = DotMap(config)

    train_split, val_split = create_splits(config)
    train_im, train_class, train_attr, train_alphas = map(list, zip(*train_split))
    val_im, val_class, val_attr, val_alphas = map(list, zip(*val_split))
    graph = tf.Graph()

    with graph.as_default():

        # Training dataset
        train_attr = tf.constant(np.array(train_attr).astype(np.float32))
        train_alphas = tf.constant(train_alphas)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_attr, train_alphas))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_split))
        batched_train_dataset = train_dataset.batch(int(config.dom2alpha_batch_size))

        # Val dataset
        val_attr = tf.constant(np.array(val_attr).astype(np.float32))
        val_alphas = tf.constant(val_alphas)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_attr, val_alphas))
        val_dataset = val_dataset.shuffle(buffer_size=len(val_split))
        batched_val_dataset = val_dataset.batch(int(config.dom2alpha_batch_size))

        # Define iterator that operates on either of the splits
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)

        text, alphas = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Define the global step to be some tf.Variable
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
        step_initializer = tf.variables_initializer([global_step_tensor])



        if config.dom2alpha_model == "linear":
            print("----------------------------------------------------------------Creating a linear model, att to alpha--------------------------------------------------------------------")
            num_input = int(config.n_attr)
            num_output = int(config.n_alphas)

            weights = {'out': tf.Variable(tf.random_normal([num_input, num_output]))}
            biases = {'out': tf.Variable(tf.random_normal([num_output]))}
            adam_vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'beta' in x.name]

            def neural_net(x):
                out_layer = tf.add(tf.matmul(x, weights['out']), biases['out'])
                return out_layer

            out = neural_net(text)

        elif config.dom2alpha_model == "multilayer":
            print("------------------------------------------------------------Creating a multilayer (3 layer) model, att to alpha-----------------------------------------------------------")
            n_input = int(config.n_attr)
            n_hidden_1 = int(config.n_hidden_1)# 400
            n_hidden_2 = int(config.n_hidden_2)# 450
            # n_hidden_2 = 450
            n_output = int(config.n_alphas)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

            # Create model
            def multilayer_perceptron(x):
                # Hidden fully connected layer with 256 neurons
                layer_1 = tf.contrib.layers.fully_connected(x, num_outputs=n_hidden_1, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

                # Hidden fully connected layer with 256 neurons
                layer_2 = tf.contrib.layers.fully_connected(layer_1, num_outputs=n_hidden_2, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

                # Output fully connected layer with a neuron for each class
                out_layer = tf.contrib.layers.fully_connected(layer_2, num_outputs=n_output, activation_fn=None, weights_regularizer=regularizer)

                return out_layer

            # Construct model
            out = multilayer_perceptron(text)


        elif config.dom2alpha_model == "2layer":
            print("------------------------------------------------------------Creating a multilayer (3 layer) model, att to alpha-----------------------------------------------------------")
            n_input = int(config.n_attr)
            n_hidden = int(config.n_hidden)# 400
            # n_hidden_2 = 450
            n_output = int(config.n_alphas)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

            # Create model
            def multilayer_perceptron(x):
                # Hidden fully connected layer with 256 neurons
                layer_1 = tf.contrib.layers.fully_connected(x, num_outputs=n_hidden, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

                # Hidden fully connected layer with 256 neurons
                #layer_2 = tf.contrib.layers.fully_connected(layer_1, num_outputs=n_hidden_2, activation_fn=tf.nn.relu, weights_regularizer=regularizer)

                # Output fully connected layer with a neuron for each class
                out_layer = tf.contrib.layers.fully_connected(layer_1, num_outputs=n_output, activation_fn=None, weights_regularizer=regularizer)

                return out_layer

            # Construct model
            out = multilayer_perceptron(text)


        #define loss

        # Normalize the gt and predicted alphas (required for before feeding to cosine distance loss function)
        out_normalized = tf.nn.l2_normalize(out, 1)
        alphas_normalized = tf.nn.l2_normalize(alphas, 1)

        if config.alpha_loss_type == "cd":
            alpha_loss = tf.reduce_mean(tf.losses.cosine_distance(alphas_normalized, out_normalized, dim=1, reduction = tf.losses.Reduction.NONE))
        elif config.alpha_loss_type == "l1":
            alpha_loss = tf.reduce_mean(tf.abs(alphas - out))
        elif config.alpha_loss_type == "cdandl1":
            alpha_loss = tf.reduce_mean(tf.abs(alphas - out)) + float(config.cdl1_reg)* tf.reduce_mean(tf.losses.cosine_distance(alphas_normalized, out_normalized, dim=1, reduction = tf.losses.Reduction.NONE))

        # regularization term: Not sure if this is necessary. It doesn't matter if the alphas scale is matched. Only the weights for the final classifier need to be of the right scale.
        reg_loss = float(config.dom2alpha_lambda_reg) * tf.abs(tf.nn.l2_loss(out) - tf.nn.l2_loss(alphas))
        loss = alpha_loss + reg_loss

        # Training Op
        optimizer = tf.train.AdamOptimizer(learning_rate=float(config.learning_rate))
        train_op = optimizer.minimize(loss)

        adam_vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'beta' in x.name]
        adam_initializer = tf.variables_initializer(adam_vars)
        init_op = tf.initialize_all_variables()

        # Define saver
        saver = tf.train.Saver()
        tf.get_default_graph().finalize()

    # Start a session to learn the modality to alpha mapping
    with tf.Session(graph=graph) as sess:
        val_loss_best = 1e20
        val_corr_best = -1
        sess.run(adam_initializer)
        sess.run(step_initializer)
        sess.run(init_op)
        tf.train.global_step(sess, global_step_tensor)

        # Start by evaluating on val class data
        sess.run(val_init_op)
        val_loss = []
        while True:
            try:
                l = sess.run(loss)
                val_loss.append(l)
            except tf.errors.OutOfRangeError:
                break
        Initial_valLoss = np.array(val_loss).mean()

        perf = []
        print('Initial Val Loss: {} '.format(Initial_valLoss))
        iteration = 1
        for epoch in range(int(config.num_epochs)):
            print('Epoch {}/{}'.format(epoch+1, int(config.num_epochs)))

            sess.run(train_init_op)
            while True:
                try:
                    sess.run(train_op)
                    iteration = iteration + 1
                    if (iteration-2)%100==0:
                        print('Iteration: {} Training Loss: {} '.format(iteration, l))
                except tf.errors.OutOfRangeError:
                    break

            print("Validating on the val set (images of val classes)")

            # Load val class info
            sess.run(val_init_op)

            val_loss = []
            val_alpha_loss = []
            val_reg_loss = []
            val_rank_corr = []

            while True:
                try:
                    l, out_val  = sess.run([loss, out])
                    val_loss.append(l)
                except tf.errors.OutOfRangeError:
                    break
            valLoss = np.array(val_loss).mean()
            print("Epoch {}, average_val_loss: {}".format(epoch, valLoss))

            if valLoss < val_loss_best:
                val_loss_best = valLoss
                checkpoint_dir = config.dom2alpha_ckpt_dir + 'mod_{}_2alpha_dset_{}_baseNcls_{}_basemodel_{}_layername_{}_d2a_model_{}_n_train_{}_alphaloss_{}_epoch_{}_loss_{:0.2f}.ckpt'.format(config.modality, config.dataset, config.n_seen_train, config.base_model, config.alpha_layer_name, config.dom2alpha_model, config.n_train_class, config.alpha_loss_type, epoch, valLoss)
                print("Saving model parameters to: ", checkpoint_dir)
                saver.save(sess, checkpoint_dir)
            else:
                print("Val loss went up ")

            iteration += 1

        print("Optimization Finished! ")
        print("Best Checkpoint dir: ", checkpoint_dir)
        print("Initial Validation loss was: {}".format(Initial_valLoss))
        print("Best Validation loss achieved: {}".format(val_loss_best))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
