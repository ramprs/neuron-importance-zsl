"""
Code to extract and save alphas from a trained network
1. Note that we're gonna have to coincide the use of previous and new finetuning dataset JSONs
2. Take bypassing into account
3. Data loader can be the same as the CNN finetuning scheme
"""
import os
import sys
import json
import codecs
import random
import importlib

# Add slim folder path to syspath
sys.path.insert(0, '/nethome/rrs6/models/research/slim')

import numpy as np
import pandas as pd
import scipy.io as scio
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from PIL import Image
from tqdm import tqdm
from pprint import pprint
from dotmap import DotMap
from tensorflow.python import pywrap_tensorflow

# Activation bypass mapping
bypass_dict = {'Max2SumPool': 'MaxPool', 'Max2SumWeighted': 'MaxPool', 'Max2SumInvWeighted': 'MaxPool', 'Identity': 'Relu'}

random.seed(123)

def overlayed_maps(gcam, imgf, im_size, save_path):
        # Load regular image
        size = im_size, im_size
        image = Image.open(imgf)
        image = image.resize(size, Image.ANTIALIAS)
        image.save(save_path.replace('.png', '_orig.png'))
        # Create colormap and upsample
        cmf = mpl.cm.get_cmap('jet')
        cam = Image.fromarray(np.uint8(cmf(gcam)*255))
        cam = cam.resize(size, Image.ANTIALIAS)
        cam.save(save_path.replace('.jpg', '.png'), 'PNG')


def parse_json(json_file):
        with open(json_file, 'r') as f:
                data = json.load(f)
        return data

def get_key(dic, val):
        # Get key corresponding to value in dictionary
        key = list(dic.keys())[list(dic.values()).index(val)]
        return key

@tf.RegisterGradient('Max2SumPool')
def _max2sumpool(op, grad):
        """
        Bad solution below. Needs a general PR fix
        """
        s = int(op.inputs[0].shape[1])
        return tf.image.resize_nearest_neighbor(grad, [s, s])

@tf.RegisterGradient('Max2SumWeighted')
def _max2sumw(op, grad):
        """
        Weighted upsampling
        """
        s = int(op.inputs[0].shape[1])
        temp = tf.image.resize_nearest_neighbor(grad, [s, s])
        act = op.inputs[0]
        act_sum = tf.reduce_sum(op.inputs[0], [1, 2])
        inv_act_sum = tf.reciprocal(act_sum + 1)
        coeff = tf.multiply(act, inv_act_sum)
        return tf.multiply(temp, coeff)

@tf.RegisterGradient('Max2SumInvWeighted')
def _max2suminvw(op, grad):
        """
        Inverse weighted upsampling
        """
        eps = 1e-5
        s = int(op.inputs[0].shape[1])
        temp = tf.image.resize_nearest_neighbor(grad, [s, s])
        act = op.inputs[0] + eps
        inv_act = tf.reciprocal(act)
        inv_act_sum = tf.reduce_sum(inv_act, [1, 2])
        inv_inv_act_sum = tf.reciprocal(inv_act_sum)
        coeff = tf.multiply(inv_act, inv_inv_act_sum)
        return tf.multiply(temp, coeff)

def save_alphas(save_path, class_names, class_idmaps, gt_alp, pred_alp, gt_labl, pred_labl, orig_labl, img, gt_gc, pred_gc, ctr):
        """
        Save the extracted alphas
        """
        gt_labl = gt_labl.tolist()
        pred_labl = pred_labl.tolist()
        orig_labl = orig_labl.tolist()
        img = img.tolist()
        gt_alp = gt_alp.tolist()
        pred_alp = pred_alp.tolist()
        gt_gc = np.array(gt_gc)
        pred_gc = np.array(pred_gc)
        for i in range(len(gt_labl)):
                res_dict = {}
                res_dict['image'] = img[i].decode('utf-8')
                print(orig_labl[i])
                res_dict['gt_class'] = class_names[int(orig_labl[i])]
                res_dict['pred_class'] = class_names[int(get_key(class_idmaps, pred_labl[i]))]
                print('Image is: ' + img[i].decode('utf-8'))
                print('Ground Truth Class is: ' + class_names[int(orig_labl[i])])
                print('Predicted Class is: ' + class_names[int(get_key(class_idmaps, pred_labl[i]))])
                res_dict['gt_alpha'] = gt_alp[i]
                res_dict['pred_alpha'] = pred_alp[i]
                res_dict['gt_cid'] = orig_labl[i]
                res_dict['pred_cid'] = get_key(class_idmaps, pred_labl[i])
                json.dump(res_dict, codecs.open(save_path + '/' + os.path.splitext(os.path.basename(res_dict['image']))[0] + '_' + class_names[orig_labl[i]] + '.json', 'w', encoding='utf-8'), separators=(',',':'), sort_keys=True, indent=4)

                # Save limited gradcam maps
                if ctr < 50:
                        print('Saving maps..')
                        gcam_save_path = save_path + '_gcam'
                        if not os.path.exists(gcam_save_path):
                                os.mkdir(gcam_save_path)
                        overlayed_maps(gt_gc[i], res_dict['image'], 224, gcam_save_path + '/' + os.path.splitext(os.path.basename(res_dict['image']))[0] + '_gt_' + class_names[orig_labl[i]] + '.png')
                        overlayed_maps(pred_gc[i], res_dict['image'], 224, gcam_save_path + '/' + os.path.splitext(os.path.basename(res_dict['image']))[0] + '_pred_' + class_names[int(get_key(class_idmaps, pred_labl[i]))] + '.png')
                        ctr += 1
        return ctr


def prepare_data(dataset_json, class_idmaps):
        """
        Function to load the dataset splits and extract alphas
        """
        json_files = dataset_json.split(',')
        dataset = []
        for i in json_files:
                dataset += parse_json(i)

        # Add original class-IDs into the mix!
        image_list, label_list, orig_label_list = [], [], []
        print('Preparing data..')
        for i in tqdm(range(len(dataset))):
                image_list.append(dataset[i][0])
                label_list.append(dataset[i][1])
                orig_label_list.append(get_key(class_idmaps, dataset[i][1]))

        prepared_dataset = list(zip(image_list, label_list, orig_label_list))
        return prepared_dataset

def get_alphas(config_json):
        # Load arguments from config file
        print('Reading Arguments..')
        config = parse_json(config_json)
        pprint(config)
        config = DotMap(config)

        # Initialize random seed
        random.seed(int(config.random_seed))
        # Number of classes
        n_classes = int(config.num_classes)
        # Load class-list
        class_names = [x.strip('\n').split(' ')[1] for x in open(config.class_list_f, 'r').readlines()]
        # Load class-ID maps
        class_idmaps = parse_json(config.class_idmaps_json)

        # Prepare data
        print('Loading and preparing data..')
        data = prepare_data(config.dataset_json, class_idmaps)

        # Extract filenames and labels
        files, labels, orig_labels = map(list, zip(*data))
        labels = np.array(labels).astype('int32')
        orig_labels = np.array(orig_labels).astype('int32')

        # Write alpha extraction here
        print('Creating graph..')
        graph = tf.Graph()
        with graph.as_default():
                # Load the preprocessing function based on argument
                preprocess_module_name = 'preprocessing.' + config.preprocess_fn
                preprocess_module = importlib.import_module(preprocess_module_name)
                print("preprocess module", preprocess_module)
                model_class = getattr(nets, config.model_class, None)
                model_name = getattr(model_class, config.model_name, None)
                im_size = int(config.image_size)

                def preprocess(filename, label, orig_label):
                        image_file = tf.read_file(filename)
                        image = tf.image.decode_jpeg(image_file, channels=3)
                        processed_image = preprocess_module.preprocess_image(image, im_size, im_size, is_training=False)
                        return filename, processed_image, label, orig_label

                # Dataset split construction
                files = tf.constant(files)
                labels = tf.constant(labels)
                orig_labels = tf.constant(orig_labels)
                dataset = tf.contrib.data.Dataset.from_tensor_slices((files, labels, orig_labels))
                dataset = dataset.map(preprocess, num_threads=int(config.num_workers), output_buffer_size=int(config.batch_size))
                batched_dataset = dataset.batch(int(config.batch_size))

                # Define iterator
                iterator = tf.contrib.data.Iterator.from_structure(batched_dataset.output_types, batched_dataset.output_shapes)
                fname, images, labels, orig_labels = iterator.get_next()

                # Dataset init ops
                init_op = iterator.make_initializer(batched_dataset)

                # Boolean variable for train-vs-test
                is_training = tf.placeholder(tf.bool)

                # (No need for is_training boolean variable; Always evaluation)
                global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

                # Get model arg_scope
                arg_scope = getattr(model_class, config.scope, None)
                with slim.arg_scope(arg_scope()):
                        bypass_name = None if config.bypass_name == 'None' else config.bypass_name
                        if bypass_name != None:
                                print('Bypassing.. ' + bypass_name)
                                bypass_keys = bypass_name.split(',')
                                if len(bypass_keys) == 1:
                                        with graph.gradient_override_map({bypass_dict[bypass_keys[0]]: bypass_keys[0]}):
                                                logits, endpoints = model_name(images, num_classes=n_classes, is_training=is_training)
                                else:
                                        with graph.gradient_override_map({bypass_dict[bypass_keys[0]]: bypass_keys[0], bypass_dict[bypass_keys[1]]: bypass_keys[1]}):
                                                logits, endpoints = model_name(images, num_classes=n_classes, is_training=is_training)
                        else:
                                logits, endpoints = model_name(images, num_classes=n_classes, is_training=is_training)

                # Print activations to decide which ones to use
                # print(endpoints)

                # Load variables from original and new check point
                # Find missing variables in new checkpoint that are present in old checkpoint

                #if config.ckpt == 'old':
                orig_ckpt = config.orig_ckpt_path
                orig_ckpt_reader = pywrap_tensorflow.NewCheckpointReader(orig_ckpt)
                new_ckpt_reader = pywrap_tensorflow.NewCheckpointReader(config.ckpt_path)
                new_var_to_shape_map = new_ckpt_reader.get_variable_to_shape_map()
                orig_var_to_shape_map = orig_ckpt_reader.get_variable_to_shape_map()
                vars_in_orig_ckpt = [key for key in sorted(orig_var_to_shape_map)]
                vars_in_new_ckpt = [key for key in sorted(new_var_to_shape_map)]
                # New ckpt init function
                new_ckpt_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.ckpt_path, [x for x in tf.contrib.framework.get_variables() if x.name.split(':')[0] in vars_in_new_ckpt])
                # Initializer
                # var_init = tf.variables_initializer(vars_to_init + [global_step_tensor])





                # # Fix variable loading from checkpoint
                # # Load variables from checkpoint
                # # Restore the ones from the checkpoint
                # # Initialize the rest
                # ckpt_reader = pywrap_tensorflow.NewCheckpointReader(config.ckpt_path)
                # var_to_shape_map = ckpt_reader.get_variable_to_shape_map()
                # vars_in_graph = tf.contrib.framework.get_variables()
                # vars_in_ckpt = []
                # for key in sorted(var_to_shape_map):
                # 	vars_in_ckpt.append(key)
                # vars_to_restore = [x for x in vars_in_graph if x.name.split(':')[0] in vars_in_ckpt]
                # ckpt_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(config.ckpt_path, vars_to_restore)
                # vars_to_init = list(set(vars_in_graph) - set(vars_to_restore))
                # var_init = tf.variables_initializer(vars_to_init + [global_step_tensor])

                # Squeeze logits
                logits = tf.squeeze(logits)

                # Get probabilities
                probs = tf.nn.softmax(logits)
                predictions = tf.to_int32(tf.argmax(logits, 1))

                # Compute alphas for both predicted and ground-truth classes
                layer_name = config.layer_name
                activations = endpoints[layer_name]
                gt_one_hot = tf.one_hot(labels, n_classes, 1.0)
                pred_one_hot = tf.one_hot(predictions, n_classes, 1.0)
                gt_loss = tf.multiply(logits, gt_one_hot)
                pred_loss = tf.multiply(logits, pred_one_hot)
                gt_grads = tf.gradients(gt_loss, activations)
                pred_grads = tf.gradients(pred_loss, activations)
                gt_alphas = tf.squeeze(tf.reduce_sum(gt_grads, [2,3]))
                pred_alphas = tf.squeeze(tf.reduce_sum(pred_grads, [2,3]))

                gt_gcam = tf.reduce_sum(tf.multiply(activations, tf.reshape(gt_alphas, [tf.shape(gt_alphas)[0], 1, 1, tf.shape(gt_alphas)[1]])), axis=3)
                pred_gcam = tf.reduce_sum(tf.multiply(activations, tf.reshape(pred_alphas, [tf.shape(pred_alphas)[0], 1, 1, tf.shape(pred_alphas)[1]])), axis=3)

                gt_gcam = (gt_gcam - tf.reduce_min(gt_gcam)) / (tf.reduce_max(gt_gcam) - tf.reduce_min(gt_gcam))
                pred_gcam = (pred_gcam - tf.reduce_min(pred_gcam)) / (tf.reduce_max(pred_gcam) - tf.reduce_min(pred_gcam))

                # Initialize the filewriter
                writer = tf.summary.FileWriter(config.save_path + '_summ')

                # Finalize graph
                tf.get_default_graph().finalize()

        if not os.path.exists(config.save_path):
                os.mkdir(config.save_path)

        # Counter to save Grad-CAM maps for sanity checks
        ctr = 0
        print('Starting session to extract and save alphas..')
        with tf.Session(graph=graph) as sess:
                # Assign from checkpoint
                # ckpt_init_fn(sess)
                new_ckpt_init_fn(sess)

                # Initialize variables
                # sess.run(var_init)

                print("Saving alpas from {} to {}".format(config.ckpt_path, config.save_path))
                # Add the model graph to tensorboard
                writer.add_graph(sess.graph)
                sess.run(init_op)
                while True:
                        try:
                                gt_alp, pred_alp, gt_labl, pred_labl, orig_labl, file, gt_gc, pred_gc = sess.run([gt_alphas, pred_alphas, labels, predictions, orig_labels, fname, gt_gcam, pred_gcam], {is_training: False})
                                # Save alphas for the batch in JSONs
                                ctr = save_alphas(config.save_path, class_names, class_idmaps, gt_alp, pred_alp, gt_labl, pred_labl, orig_labl, file, gt_gc, pred_gc, ctr)

                        except tf.errors.OutOfRangeError:
                                break
                print("Saved alpas from {} to {}".format(config.ckpt_path, config.save_path))
