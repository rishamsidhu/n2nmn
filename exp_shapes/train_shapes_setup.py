from __future__ import absolute_import, division, print_function

import random
import pickle
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import tensorflow as tf
# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))
import json

from models_shapes.nmn3_assembler import Assembler
from models_shapes.nmn3_model import NMN3ModelAtt

# Module parameters
H_im = 30
W_im = 30
num_choices = 2
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 256
num_layers = 2
encoder_dropout = True
decoder_dropout = True
decoder_sampling = True
T_encoder = 15
T_decoder = 11
N = 256

# Training parameters
weight_decay = 5e-4
max_grad_l2_norm = 10
max_iter = 40000
snapshot_interval = 10000
exp_name = "shapes_gt_layout_" + input("Experiment identifying numbers:")
snapshot_dir = './exp_shapes/tfmodel/%s/' % exp_name

# Log params
log_interval = 20
log_dir = './exp_shapes/tb/%s/' % exp_name

# Data files
vocab_shape_file = './exp_shapes/data/vocabulary_shape.txt'
vocab_layout_file = './exp_shapes/data/vocabulary_layout.txt'
image_sets = ['train.large', 'train.med', 'train.small', 'train.tiny']
training_text_files = './exp_shapes/shapes_dataset/%s.query_str.txt'
training_image_files = './exp_shapes/shapes_dataset/%s.input.npy'
training_label_files = './exp_shapes/shapes_dataset/%s.output'
training_gt_layout_file = './exp_shapes/data/%s.query_layout_symbols.json'
image_mean_file = './exp_shapes/data/image_mean.npy'

# Load vocabulary
with open(vocab_shape_file) as f:
    vocab_shape_list = [s.strip() for s in f.readlines()]
vocab_shape_dict = {vocab_shape_list[n]:n for n in range(len(vocab_shape_list))}
num_vocab_txt = len(vocab_shape_list)

assembler = Assembler(vocab_layout_file)
num_vocab_nmn = len(assembler.module_names)

#random seed
np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time()) + 9)

# Network inputs
text_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_batch = tf.placeholder(tf.float32, [None, H_im, W_im, 3])
expr_validity_batch = tf.placeholder(tf.bool, [None])
vqa_label_batch = tf.placeholder(tf.int32, [None])
use_gt_layout = tf.constant(True, dtype=tf.bool)
gt_layout_batch = tf.placeholder(tf.int32, [None, None])

# The model
nmn3_model = NMN3ModelAtt(image_batch, text_seq_batch,
    seq_length_batch, T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim,
    num_layers=num_layers, EOS_idx=assembler.EOS_idx,
    encoder_dropout=encoder_dropout,
    decoder_dropout=decoder_dropout,
    decoder_sampling=decoder_sampling,
    num_choices=num_choices, use_gt_layout=use_gt_layout,
    gt_layout_batch=gt_layout_batch)

compiler = nmn3_model.compiler
scores = nmn3_model.scores
log_seq_prob = nmn3_model.log_seq_prob

# Loss function
softmax_loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=scores, labels=vqa_label_batch)
# The final per-sample loss, which is vqa loss for valid expr
# and invalid_expr_loss for invalid expr
final_loss_per_sample = softmax_loss_per_sample  # All exprs are valid

avg_sample_loss = tf.reduce_mean(final_loss_per_sample)
seq_likelihood_loss = tf.reduce_mean(-log_seq_prob)

total_training_loss = seq_likelihood_loss + avg_sample_loss
total_loss = total_training_loss + weight_decay * nmn3_model.l2_reg

# Train with Adam
solver = tf.train.AdamOptimizer()
gradients = solver.compute_gradients(total_loss)

# Clip gradient by L2 norm
# gradients = gradients_part1+gradients_part2
gradients = [(tf.clip_by_norm(g, max_grad_l2_norm), v)
             for g, v in gradients]
solver_op = solver.apply_gradients(gradients)

# Training operation
# Partial-run can't fetch training operations
# some workaround to make partial-run work
with tf.control_dependencies([solver_op]):
    train_step = tf.constant(0)

sess.run(tf.global_variables_initializer())

#varaible access
prefix = 'neural_module_network/layout_execution/'
mods = ['TransformModule', 'FindModule', 'AnswerModule']

swaps = dict.fromkeys(mods)
old = dict.fromkeys(mods, 0)
num_swaps = int(input("Number of swaps?"))

temp = {}

for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    y = x.eval(session = sess)
    temp[x.name] = np.array(y)
with open(os.path.join(snapshot_dir, "start_vars.txt"), "wb") as f:
    pickle.dump(temp, f)

for mod in mods:
    swaps[mod] = []
    for i in range(num_swaps):
        d = {}
        for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = prefix + mod):
            y = x.eval(session = sess)
            if i == 0:
                d[x.name] = np.array(y)
            else:
                if "bias" in x.name:
                    d[x.name] = np.zeros(y.shape)
                elif "Adam" in x.name:
                    d[x.name] = np.zeros(y.shape)
                else:
                    #var = tf.get_variable(name = "temp" + str(count), initializer = tf.contrib.layers.xavier_initializer(), shape = y.shape)
                    var = tf.Variable(tf.contrib.layers.xavier_initializer()(shape = y.shape))
                    sess.run(tf.local_variables_initializer())
                    sess.run(tf.global_variables_initializer())
                    y = var.eval(session = sess)
                    d[x.name] = np.array(y)
        swaps[mod] += [d]

with open(os.path.join(snapshot_dir, "start_swaps.txt"), "wb") as f:
    pickle.dump(swaps, f)
