"""
Code to finetune a given CNN on the seen-images for the seen-classes concerned datasets
(Have to add special checks to handle images with multiple frames)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import pdb
import json
import random
import importlib
import itertools

# Add slim folder path to syspath
sys.path.insert(0, '/nethome/rrs6/models/research/slim')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.ops import array_ops

from tqdm import tqdm
from pprint import pprint
from dotmap import DotMap
from random import shuffle

# Load JSON
def parse_json(json_file):
        with open(json_file, 'r') as f:
                data = json.load(f)
        return data

# Save JSON
def save_json(data, json_file):
        with open(json_file, 'w') as f:
                json.dump(data, f)

# Plot losses/accuracy on train-val splits
def plot_metrics(train_metric, val_metric, title, save_fname):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(train_metric)), train_metric, label='Train Split')
        ax.plot(range(len(val_metric)), val_metric, label='Validation Split')
        plt.legend()
        plt.title(title)
        fig.savefig(save_fname)
        plt.close(fig)

# Generate hyper-params for grid sweep
# (Do not mess with dropout)
def generate_hyperparameters(config):
        # Get hyper-parameter ranges from the config and generate values
        return {
                "lr_fc": [10**x for x in range(int(config.lr_fc.split(',')[0]), int(config.lr_fc.split(',')[1])+1)],
                "lr_full": [10**x for x in range(int(config.lr_full.split(',')[0]), int(config.lr_full.split(',')[1])+1)],
                "wt_dec": [10**x for x in range(int(config.wt_dec.split(',')[0]), int(config.wt_dec.split(',')[1])+1)]
                }

        # Return trial-based accuracy (for both loss and accuracy)
def check_metrics(sess, prediction, labels, loss, is_training, dataset_init_op, trials=10, verbose=False):
        sess.run(dataset_init_op)
        whole_loss_list, whole_pred_list, whole_label_list = [], [], []
        while True:
                try:
                        preds, labl, los = sess.run([prediction, labels, loss], {is_training: False})
                        if verbose:
                                print(preds.shape)
                        whole_loss_list.append(los)
                        whole_pred_list += preds.tolist()
                        whole_label_list += labl.tolist()
                except tf.errors.OutOfRangeError:
                        break

        # Sample-based on number of trials
        # Assume default split 0.8
        all_indices = list(range(len(whole_label_list)))
        num_samples = int(0.8*len(all_indices))
        trial_loss_res = []
        trial_acc_res = []
        for i in range(trials):
                shuffle(all_indices)
                sample_indices = all_indices[:num_samples]
                pred_sample = [x for i,x in enumerate(whole_pred_list) if i in sample_indices]
                label_sample = [x for i,x in enumerate(whole_label_list) if i in sample_indices]
                loss_sample = [x for i,x in enumerate(whole_loss_list) if i in sample_indices]
                # Get unique labels
                unique_labl = list(set(label_sample))
                per_cls_acc = []
                for y in unique_labl:
                        gt_ind = [i for i,x in enumerate(label_sample) if x == y]
                        cls_acc = float([pred_sample[i] for i in gt_ind].count(y)/len(gt_ind))
                        per_cls_acc.append(cls_acc)

                trial_loss_res.append(np.mean(loss_sample))
                trial_acc_res.append(np.mean(per_cls_acc))

        # Return results averaged over trials
        return np.mean(trial_loss_res), np.std(trial_loss_res), np.mean(trial_acc_res), np.std(trial_acc_res)

# Function to load data-splits
def load_data(train_split_json, val_split_json, test_split_json):
        # Load data from corresponding splits
        if os.path.isfile(train_split_json):
                print('Train split file exists..')
                train_split = parse_json(train_split_json)
        else:
                print('Train split file does not exist..')

        if os.path.isfile(val_split_json):
                print('Val split file exists..')
                val_split = parse_json(val_split_json)
        else:
                print('Val split file does not exist..')

        if os.path.isfile(test_split_json):
                print('Test split file exists..')
                test_split = parse_json(test_split_json)
        else:
                print('Test split file does not exist..')

        return train_split, val_split, test_split

# Load arguments from config and run training
# Have multiple runs over hyper-parameters to make sure correct ones are selected
def run_training(config, learning_rate_fc, learning_rate_full, weight_decay, batch_size):
        assert(config.finetune_whole_cnn!=None)
        # Suffix for ckpt-path
        ckpt_suffix = 'setting_lrfc_' + str(learning_rate_fc) + '_lrfull_' + str(learning_rate_full) + '_wd_' + str(weight_decay) + '_bsz_' + str(batch_size)

        # Initialize random seed
        random.seed(int(config.random_seed))

        # Number of classes to fine-tune CNN on
        n_classes = int(config.num_classes)

        # Load dataset splits
        train_split, val_split, test_split = load_data(config.train_split_json, config.val_split_json, config.test_split_json)

        # Extract filenames and labels
        train_files, train_labels = map(list, zip(*train_split))
        val_files, val_labels = map(list, zip(*val_split))
        test_files, test_labels = map(list, zip(*test_split))
        print('#train-instances: %d' % len(train_split))
        print('#val-instances: %d' % len(val_split))
        print('#test-instances: %d' % len(test_split))

        # Type-cast labels
        train_labels = np.array(train_labels).astype('int32')
        val_labels = np.array(val_labels).astype('int32')
        test_labels = np.array(test_labels).astype('int32')

        save_dir = config.save_path + ckpt_suffix + '/'
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        # Write model graph definition
        print('Creating Graph..')
        graph = tf.Graph()
        with graph.as_default():
                # Load the preprocessing function based on argument
                preprocess_module_name = 'preprocessing.' + config.preprocess_fn
                preprocess_module = importlib.import_module(preprocess_module_name)

                model_class = getattr(nets, config.model_class, None)
                model_name = getattr(model_class, config.model_name, None)
                im_size = int(config.image_size)

                def train_preprocess(filename, label):
                        image_file = tf.read_file(filename)
                        image = tf.image.decode_jpeg(image_file, channels=3)
                        processed_image = preprocess_module.preprocess_image(image, im_size, im_size, is_training=True)
                        return processed_image, label

                def test_preprocess(filename, label):
                        image_file = tf.read_file(filename)
                        image = tf.image.decode_image(image_file, channels=3)
                        processed_image = preprocess_module.preprocess_image(image, im_size, im_size, is_training=False)
                        return processed_image, label

                # Contrib dataset creation

                # Training Split
                train_files = tf.constant(train_files)
                train_labels = tf.constant(train_labels)
                train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_files, train_labels))
                train_dataset = train_dataset.map(train_preprocess, num_threads=int(config.num_workers), output_buffer_size=batch_size)
                train_dataset = train_dataset.shuffle(buffer_size=10000)
                batched_train_dataset = train_dataset.batch(batch_size)

                # Validation Split
                val_files = tf.constant(val_files)
                val_labels = tf.constant(val_labels)
                val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_files, val_labels))
                val_dataset = val_dataset.map(test_preprocess, num_threads=int(config.num_workers), output_buffer_size=batch_size)
                batched_val_dataset = val_dataset.batch(batch_size)

                # Test Split
                test_files = tf.constant(test_files)
                test_labels = tf.constant(test_labels)
                test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_files, test_labels))
                test_dataset = test_dataset.map(test_preprocess, num_threads=int(config.num_workers), output_buffer_size=batch_size)
                batched_test_dataset = test_dataset.batch(batch_size)

                # Define iterator
                iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
                images, labels = iterator.get_next()

                # Dataset init ops
                train_init_op = iterator.make_initializer(batched_train_dataset)
                val_init_op = iterator.make_initializer(batched_val_dataset)
                test_init_op = iterator.make_initializer(batched_test_dataset)

                # Boolean variable for train-vs-test
                is_training = tf.placeholder(tf.bool)

                # Define the global step to be some tf.Variable
                global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

                arg_scope = getattr(model_class, config.scope, None)
                # Get model arg-scope
                if 'resnet' in config.model_class:
                    with slim.arg_scope(arg_scope(weight_decay=float(weight_decay))):
                        logits, endpoints = model_name(images, num_classes=n_classes, is_training=is_training)
                        logits = array_ops.squeeze(logits, [1,2])
                else:
                    with slim.arg_scope(arg_scope(weight_decay=float(weight_decay))):
                        logits, endpoints = model_name(images, num_classes=n_classes, is_training=is_training, dropout_keep_prob=float(config.dropout))

                # Squeeze logits
                #logits = array_ops.squeeze(logits, [1,2])
                #logits = tf.Squeeze(logits)
                # Check for checkpoint-path
                assert(os.path.isfile(config.ckpt_path))

                # Define variables to train and restore from the checkpoint
                vars_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[config.model_name + '/' + config.layer_name, 'global_step'])
                ckpt_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.ckpt_path, vars_to_restore)

                # Evaluation metrics
                prediction = tf.to_int32(tf.argmax(logits, 1))
                correct_prediction = tf.equal(prediction, labels)
                # Accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                # Define loss-criterion
                loss_ce = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                # Add regularizers
                loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])*weight_decay
                loss = loss_ce + loss_reg

                # Define 2 optimization ops:
                #	1. To train the last layer for certain #epochs
                last_optimizer = tf.train.AdamOptimizer(learning_rate_fc)
                last_vars = tf.contrib.framework.get_variables(config.model_name + '/' + config.layer_name)
                last_train_op = tf.contrib.slim.learning.create_train_op(loss, last_optimizer, variables_to_train=last_vars)
                # (Call tf.contrib.framework.get_variables() again after declaring optimizer)
                last_init = tf.variables_initializer(tf.contrib.framework.get_variables(config.model_name + '/' + config.layer_name) + \
                        [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'beta' in x.name] +\
                        [global_step_tensor])

                #	2. To train the whole network for certain (follow-up) #epochs
                full_optimizer = tf.train.AdamOptimizer(learning_rate_full)
                full_train_op = tf.contrib.slim.learning.create_train_op(loss, full_optimizer)
                full_vars = tf.contrib.framework.get_variables()
                full_init = tf.variables_initializer([x for x in full_vars if ('Adam' in x.name)] + \
                        [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if ('beta' in x.name)])

                # Log summaries
                for var in last_vars:
                        tf.summary.histogram(var.name, var)
                tf.summary.scalar('Loss', loss)
                tf.summary.scalar('Accuracy', accuracy)
                merged_summary = tf.summary.merge_all()

                # Initialize the filewriter
                writer = tf.summary.FileWriter(save_dir + '_filewriter')

                # Define saver to save checkpoints
                saver = tf.train.Saver(tf.trainable_variables() + [x for x in tf.contrib.framework.get_variables() if 'moving_mean' in x.name or 'moving_var' in x.name])
                tf.get_default_graph().finalize()

        # Define variables to plot and store summaries
        tr_loss = []
        vl_loss = []
        tr_acc = []
        vl_acc = []
        iteration = 1
        ckpt_tracker = 0
        es_fc = int(config.estop_fc)
        es_full = int(config.estop_full)
        val_monitor_fc = 9999
        val_monitor_full = 9999
        ctr_fc = 0
        ctr_full = 0
        inter_epoch = 0
        with tf.Session(graph=graph) as sess:
                ckpt_init_fn(sess)

                # Add the model graph to tensorboard
                writer.add_graph(sess.graph)

                # Train last fc-layer
                sess.run(last_init)
                tf.train.global_step(sess, global_step_tensor)
                print('Training the last layer now..')
                for epoch in range(int(config.num_epochs_fc)):
                        if ctr_fc >= es_fc:
                            break
                        # Run an epoch over the training data
                        print('Epoch %d/%d' % (epoch+1, int(config.num_epochs_fc)))
                        sess.run(train_init_op)
                        while True:
                                try:
                                        loss_fc, s, logits_val = sess.run([last_train_op, merged_summary, logits], {is_training: True})
                                        if iteration%100==0:
                                            print('Iteration %d Training Loss: %f' % (iteration, loss_fc))
                                        iteration += 1
                                        writer.add_summary(s, iteration)
                                except tf.errors.OutOfRangeError:
                                    break

                        # Check metrics on train and validation sets
                        train_loss, train_loss_std, train_acc, train_acc_std = check_metrics(sess, prediction, labels, loss, is_training, train_init_op)
                        val_loss, val_loss_std, val_acc, val_acc_std = check_metrics(sess, prediction, labels, loss, is_training, val_init_op)
                        print('Epoch %d Training Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, train_loss, train_loss_std, train_acc, train_acc_std))
                        print('Epoch %d Validation Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, val_loss, val_loss_std, val_acc, val_acc_std))
                        # Test-split performance
                        test_loss, test_loss_std, test_acc, test_acc_std = check_metrics(sess, prediction, labels, loss, is_training, test_init_op)
                        print('Epoch %d Test Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, test_loss, test_loss_std, test_acc, test_acc_std))
                        test_set_perf = { 'Loss': str(test_loss) + '+/-' + str(test_loss_std), 'Accuracy': str(test_acc) + '+/-' + str(test_acc_std)}
                        save_json(test_set_perf, save_dir + 'test_perf.json')
                        tr_loss.append(train_loss)
                        vl_loss.append(val_loss)
                        tr_acc.append(train_acc)
                        vl_acc.append(val_acc)
                        if val_loss < val_monitor_fc:
                                val_monitor_fc = val_loss
                                ctr_fc = 0
                                print('Saving checkpoint...', save_dir+'last_layer_ckpt-'+str(iteration))
                                saver.save(sess=sess, save_path=save_dir + 'last_layer_ckpt', global_step=iteration)
                                ckpt_tracker = iteration
                        else:
                                ctr_fc += 1
                        plot_metrics(tr_loss, vl_loss, 'Cross_Entropy_Loss', save_dir + 'loss_log.png')
                        plot_metrics(tr_acc, vl_acc, 'Accuracy', save_dir + 'accuracy_log.png')
                        inter_epoch = epoch
                print("Best checkpoint", ckpt_tracker)
                if int(config.finetune_whole_cnn) == 1:

                        # Restore previously (best) checkpoint
                        saver.restore(sess, save_dir + 'last_layer_ckpt-' + str(ckpt_tracker))

                        # Train the entire CNN
                        sess.run(full_init)
                        print('Fine-tuning the whole network')
                        for epoch in range(int(config.num_epochs_full)):
                                if ctr_full >= es_full:
                                        break
                            # Run an epoch over the training data
                                print('Epoch %d/%d' % (epoch+inter_epoch, int(config.num_epochs_fc)))
                                sess.run(train_init_op)
                                while True:
                                        try:
                                                loss_full, s = sess.run([full_train_op, merged_summary], {is_training: True})
                                                if iteration%100==0:

                                                    print('Iteration %d Training Loss: %f' % (iteration, loss_full))
                                                iteration += 1
                                                writer.add_summary(s, iteration)
                                        except tf.errors.OutOfRangeError:
                                                break
                                # Check metrics on train and validation sets
                                train_loss, train_loss_std, train_acc, train_acc_std = check_metrics(sess, prediction, labels, loss, is_training, train_init_op)
                                val_loss, val_loss_std, val_acc, val_acc_std = check_metrics(sess, prediction, labels, loss, is_training, val_init_op)
                                print('Epoch %d Training Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, train_loss, train_loss_std, train_acc, train_acc_std))
                                print('Epoch %d Validation Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, val_loss, val_loss_std, val_acc, val_acc_std))
                                # Test-split performance
                                test_loss, test_loss_std, test_acc, test_acc_std = check_metrics(sess, prediction, labels, loss, is_training, test_init_op)
                                print('Epoch %d Test Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, test_loss, test_loss_std, test_acc, test_acc_std))
                                tr_loss.append(train_loss)
                                vl_loss.append(val_loss)
                                tr_acc.append(train_acc)
                                vl_acc.append(val_acc)
                                # Test-split performance
                                if val_loss < val_monitor_full:
                                        val_monitor_full = val_loss
                                        ctr_full = 0
                                        print('Saving checkpoint...', save_dir +'ckpt-'+str(iteration))
                                        saver.save(sess=sess, save_path=save_dir + 'ckpt', global_step=iteration)
                                        test_set_perf = { 'Loss': str(test_loss) + '+/-' + str(test_loss_std), 'Accuracy': str(test_acc) + '+/-' + str(test_acc_std)}
                                        save_json(test_set_perf, save_dir + 'test_perf.json')
                                else:
                                        ctr_full += 1
                                plot_metrics(tr_loss, vl_loss, 'Cross_Entropy_Loss', save_dir + 'loss_log.png')
                                plot_metrics(tr_acc, vl_acc, 'Accuracy', save_dir + 'accuracy_log.png')


# Load arguments from config and run training
def run_training_no_search(config_json):

        config = parse_json(config_json)
        #assert(config.finetune_whole_cnn!=None)
        print('Training Script Arguments..')
        pprint(config)
        config = DotMap(config)
        # Initialize random seed
        random.seed(int(config.random_seed))
        # Number of classes to fine-tune CNN on
        n_classes = int(config.num_classes)
        # Load dataset splits
        train_split, val_split, test_split = load_data(config.train_split_json, config.val_split_json, config.test_split_json)
        # Extract filenames and labels
        train_files, train_labels = map(list, zip(*train_split))
        val_files, val_labels = map(list, zip(*val_split))
        test_files, test_labels = map(list, zip(*test_split))
        print('#train-instances: %d' % len(train_split))
        print('#val-instances: %d' % len(val_split))
        print('#test-instances: %d' % len(test_split))
        # Type-cast labels
        train_labels = np.array(train_labels).astype('int32')
        val_labels = np.array(val_labels).astype('int32')
        test_labels = np.array(test_labels).astype('int32')
        save_dir = config.save_path + '/'
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)



        # Write model graph definition
        print('Creating Graph..')
        graph = tf.Graph()
        with graph.as_default():
                model_class = getattr(nets, config.model_class, None)
                model_name = getattr(model_class, config.model_name, None)
                #im_size = getattr(model_name, 'default_image_size', None)
                im_size = int(config.image_size)
                MEAN = [float(config.c1_mean), float(config.c2_mean), float(config.c3_mean)]
                def _parse_function(filename, label):
                        image_file = tf.read_file(filename)
                        image_decoded = tf.image.decode_jpeg(image_file, channels=3)
                        image = tf.cast(image_decoded, tf.float32)
                        image_resized = tf.image.resize_images(image, [im_size, im_size])
                        return image_resized, label

                def preprocess(image, label):
                        means = tf.reshape(tf.constant(MEAN), [1, 1, 3])
                        processed_image = image - means
                        return processed_image, label

                # Contrib dataset creation

                # Training Split
                train_files = tf.constant(train_files)
                train_labels = tf.constant(train_labels)
                train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_files, train_labels))
                # pdb.set_trace()
                train_dataset = train_dataset.map(_parse_function, num_threads=int(config.num_workers), output_buffer_size=int(config.batch_size))
                train_dataset = train_dataset.map(preprocess, num_threads=int(config.num_workers), output_buffer_size=int(config.batch_size))
                train_dataset = train_dataset.shuffle(buffer_size=10000)
                batched_train_dataset = train_dataset.batch(int(config.batch_size))

                # Validation Split
                val_files = tf.constant(val_files)
                val_labels = tf.constant(val_labels)
                val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_files, val_labels))
                val_dataset = val_dataset.map(_parse_function, num_threads=int(config.num_workers), output_buffer_size=int(config.batch_size))
                val_dataset = val_dataset.map(preprocess, num_threads=int(config.num_workers), output_buffer_size=int(config.batch_size))
                batched_val_dataset = val_dataset.batch(int(config.batch_size))

                # Test Split
                test_files = tf.constant(test_files)
                test_labels = tf.constant(test_labels)
                test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_files, test_labels))
                test_dataset = test_dataset.map(_parse_function, num_threads=int(config.num_workers), output_buffer_size=int(config.batch_size))
                test_dataset = test_dataset.map(preprocess, num_threads=int(config.num_workers), output_buffer_size=int(config.batch_size))
                batched_test_dataset = test_dataset.batch(int(config.batch_size))

                # Define iterator
                iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
                images, labels = iterator.get_next()

                # Dataset init ops
                train_init_op = iterator.make_initializer(batched_train_dataset)
                val_init_op = iterator.make_initializer(batched_val_dataset)
                test_init_op = iterator.make_initializer(batched_test_dataset)

                # Boolean variable for train-vs-test
                is_training = tf.placeholder(tf.bool)

                # Define the global step to be some tf.Variable
                global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

                # Get model arg-scope
                arg_scope = getattr(model_class, config.scope, None)
                print(arg_scope)
                if 'resnet' in config.model_class:
                    with slim.arg_scope(arg_scope(weight_decay=float(config.wt_dec))):
                        logits, endpoints = model_name(images, num_classes=n_classes, is_training=is_training)
                        logits = array_ops.squeeze(logits, [1,2])
                else:
                    with slim.arg_scope(arg_scope(weight_decay=float(config.wt_dec))):
                        logits, endpoints = model_name(images, num_classes=n_classes, is_training=is_training, dropout_keep_prob=float(config.dropout))


                # Check for checkpoint-path
                # assert(os.path.isfile(config.ckpt_path))

                # Define variables to train and restore from the checkpoint
                vars_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[config.model_name + '/' + config.layer_name, 'global_step'])
                ckpt_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.ckpt_path, vars_to_restore)

                # Define loss-criterion
                loss_ce = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                # Add regularizers
                loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])*float(config.wt_dec)
                loss = loss_ce + loss_reg

                # Define 2 optimization ops:
                #	1. To train the last layer for certain #epochs
                last_optimizer = tf.train.AdamOptimizer(float(config.lr_fc))
                last_vars = tf.contrib.framework.get_variables(config.model_name + '/' + config.layer_name)
                last_train_op = tf.contrib.slim.learning.create_train_op(loss, last_optimizer, variables_to_train=last_vars, clip_gradient_norm=4.0)
                # (Call tf.contrib.framework.get_variables() again after declaring optimizer)
                last_init = tf.variables_initializer(tf.contrib.framework.get_variables(config.model_name + '/' + config.layer_name) + \
                        [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'beta' in x.name] +\
                        [global_step_tensor])

                #	2. To train the whole network for certain (follow-up) #epochs
                full_optimizer = tf.train.AdamOptimizer(float(config.lr_full))
                full_train_op = tf.contrib.slim.learning.create_train_op(loss, full_optimizer, clip_gradient_norm=4.0)
                full_vars = tf.contrib.framework.get_variables()
                full_init = tf.variables_initializer([x for x in full_vars if ('Adam' in x.name)] + \
                        [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if ('beta' in x.name)])

                # Evaluation metrics
                logits_argmax = tf.argmax(logits, 1)
                prediction = tf.to_int32(logits_argmax)
                correct_prediction = tf.equal(prediction, labels)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                # Log summaries
                for var in last_vars:
                        tf.summary.histogram(var.name, var)
                tf.summary.scalar('Loss', loss)
                tf.summary.scalar('Accuracy', accuracy)
                merged_summary = tf.summary.merge_all()

                # Initialize the filewriter
                writer = tf.summary.FileWriter(config.save_path + 'filewriter')

                # Define saver to save checkpoints
                # pdb.set_trace()
                saver = tf.train.Saver(tf.trainable_variables() + [x for x in tf.contrib.framework.get_variables() if 'moving_mean' in x.name or 'moving_var' in x.name])
                # saver = tf.train.Saver(tf.trainable_variables())
                tf.get_default_graph().finalize()

        if not os.path.exists(config.save_path):
                os.mkdir(config.save_path)

        # Define variables to plot and store summaries
        tr_loss = []
        vl_loss = []
        tr_acc = []
        vl_acc = []
        iteration = 1
        es_fc = int(config.estop_fc)
        es_full = int(config.estop_full)
        val_monitor_fc = 9999
        val_monitor_full = 9999
        ctr_fc = 0
        ctr_full = 0
        inter_epoch = 0

        with tf.Session(graph=graph) as sess:
                ckpt_init_fn(sess)

                # Add the model graph to tensorboard
                writer.add_graph(sess.graph)

                # Train last fc-layer
                sess.run(last_init)
                tf.train.global_step(sess, global_step_tensor)
                print('Training the last layer (no search)..')
                for epoch in range(int(config.num_epochs_fc)):
                        if ctr_fc >= es_fc:
                            break
                        # Run an epoch over the training data
                        print('Epoch %d/%d' % (epoch+1, int(config.num_epochs_fc)))
                        sess.run(train_init_op)
                        while True:
                                try:
                                        loss_fc, s, logits_val, prediction_val = sess.run([last_train_op, merged_summary, logits_argmax, prediction], {is_training: True})
                                        if iteration%50==0:
                                            print('Iteration %d Training Loss: %f' % (iteration, loss_fc))
                                        # print('prediction', prediction_val.shape)
                                        iteration += 1
                                        writer.add_summary(s, iteration)
                                except tf.errors.OutOfRangeError:
                                        break

                        # Check metrics on train and validation sets
                        train_loss, train_loss_std, train_acc, train_acc_std = check_metrics(sess, prediction, labels, loss, is_training, train_init_op)
                        val_loss, val_loss_std, val_acc, val_acc_std = check_metrics(sess, prediction, labels, loss, is_training, val_init_op)
                        print('Epoch %d Training Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, train_loss, train_loss_std, train_acc, train_acc_std))
                        print('Epoch %d Validation Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, val_loss, val_loss_std, val_acc, val_acc_std))
                        # Test-split performance
                        #test_loss, test_loss_std, test_acc, test_acc_std = check_metrics(sess, prediction, labels, loss, is_training, test_init_op, trials=10, verbose=True)
                        # print('Epoch %d Test Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, test_loss, test_loss_std, test_acc, test_acc_std))
                        #test_set_perf = { 'Loss': str(test_loss) + '+/-' + str(test_loss_std), 'Accuracy': str(test_acc) + '+/-' + str(test_acc_std)}
                        # save_json(test_set_perf, save_dir + 'test_perf.json')

                        tr_loss.append(train_loss)
                        vl_loss.append(val_loss)
                        tr_acc.append(train_acc)
                        vl_acc.append(val_acc)
                        if val_loss < val_monitor_fc:
                                val_monitor_fc = val_loss
                                ctr_fc = 0
                                print('Saving checkpoint...', config.save_path+'ckpt-'+str(iteration))
                                saver.save(sess=sess, save_path=config.save_path + 'ckpt', global_step=iteration)
                                ckpt_tracker = iteration
                        else:
                                ctr_fc += 1
                        plot_metrics(tr_loss, vl_loss, 'Cross_Entropy_Loss', config.save_path + 'loss_log.png')
                        plot_metrics(tr_acc, vl_acc, 'Accuracy', config.save_path + 'accuracy_log.png')
                        inter_epoch = epoch

                print("best checkpoint", ckpt_tracker)

                if int(config.finetune_whole_cnn) == 1:

                        # Restore previously (best) checkpoint
                        saver.restore(sess, save_dir + 'ckpt-' + str(ckpt_tracker))


                        # Train the entire CNN
                        sess.run(full_init)
                        print('Fine-tuning the whole network')
                        for epoch in range(int(config.num_epochs_full)):
                                if ctr_full >= es_full:
                                        break
                                # Run an epoch over the training data
                                print('Epoch %d/%d' % (epoch+inter_epoch, int(config.num_epochs_fc)))
                                sess.run(train_init_op)
                                while True:
                                        try:
                                                loss_full, s = sess.run([full_train_op, merged_summary], {is_training: True})
                                                if iteration%50==0:
                                                    print('Iteration %d Training Loss: %f' % (iteration, loss_full))
                                                iteration += 1
                                                writer.add_summary(s, iteration)
                                        except tf.errors.OutOfRangeError:
                                                break
                                # Check metrics on train and validation sets
                                train_loss, train_loss_std, train_acc, train_acc_std = check_metrics(sess, prediction, labels, loss, is_training, train_init_op)
                                val_loss, val_loss_std, val_acc, val_acc_std = check_metrics(sess, prediction, labels, loss, is_training, val_init_op)
                                print('Epoch %d Training Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, train_loss, train_loss_std, train_acc, train_acc_std))
                                print('Epoch %d Validation Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, val_loss, val_loss_std, val_acc, val_acc_std))
                                # Test-split performance
                                # test_loss, test_loss_std, test_acc, test_acc_std = check_metrics(sess, prediction, labels, loss, is_training, test_init_op)
                                # print('Epoch %d Test Loss %f +/- %f Accuracy %f +/- %f' % (epoch + 1, test_loss, test_loss_std, test_acc, test_acc_std))
                                tr_loss.append(train_loss)
                                vl_loss.append(val_loss)
                                tr_acc.append(train_acc)
                                vl_acc.append(val_acc)
                                # Test-split performance
                                if val_loss < val_monitor_full:
                                        val_monitor_full = val_loss
                                        ctr_full = 0
                                        print('Saving checkpoint...', config.save_path+'ckpt-'+str(iteration))
                                        saver.save(sess=sess, save_path=save_dir + 'ckpt', global_step=iteration)
                                        # test_set_perf = { 'Loss': str(test_loss) + '+/-' + str(test_loss_std), 'Accuracy': str(test_acc) + '+/-' + str(test_acc_std)}
                                        # save_json(test_set_perf, save_dir + 'test_perf.json')
                                else:
                                        ctr_full += 1
                                plot_metrics(tr_loss, vl_loss, 'Cross_Entropy_Loss', save_dir + 'loss_log.png')
                                plot_metrics(tr_acc, vl_acc, 'Accuracy', save_dir + 'accuracy_log.png')



def validate_and_train(config_json):
        config = parse_json(config_json)
        print('Training Script Arguments..')
        pprint(config)
        config = DotMap(config)

        # Create directory to save all settings
        if not os.path.exists(config.save_path):
                os.mkdir(config.save_path)

        # Get hyper-params
        hyper_dict = DotMap(generate_hyperparameters(config))
        print(hyper_dict)
        lr_fc_ls = hyper_dict.lr_fc
        lr_full_ls = hyper_dict.lr_full
        wt_dec_ls = hyper_dict.wt_dec

        # Generate all hyper-paramter combinations
        hparam_comb = list(itertools.product(*[lr_fc_ls, lr_full_ls, wt_dec_ls]))
        print("hparam_comb")
        pprint(hparam_comb)

        for i in range(len(hparam_comb)):
                print('Running with settings..')
                print('-------------------------------')
                print('Setting {} of {}'.format(i, len(hparam_comb)))
                pprint({'lr_fc': hparam_comb[i][0], 'lr_full': hparam_comb[i][1], 'wt_dec': hparam_comb[i][2]})
                print('-------------------------------')
                run_training(config, hparam_comb[i][0], hparam_comb[i][1], hparam_comb[i][2], int(config.batch_size))
                print('-------------------------------')
                print('-------------------------------')
                print('-------------------------------')
