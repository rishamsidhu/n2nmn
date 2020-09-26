from __future__ import absolute_import, division, print_function

import random
import time
import pickle
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
#image_sets = ['train.large', 'train.med', 'train.small', 'train.tiny']
#image_sets = ['train.nab.large', 'train.nab.med', 'train.nab.small', 'train.nab.tiny']
image_sets = ['flip.train']
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

# Load training data
training_questions = []
training_labels = []
training_images_list = []
gt_layout_list = []

for image_set in image_sets:
    with open(training_text_files % image_set) as f:
        training_questions += [l.strip() for l in f.readlines()]
    with open(training_label_files % image_set) as f:
        training_labels += [l.strip() == 'true' for l in f.readlines()]
    training_images_list.append(np.load(training_image_files % image_set))
    with open(training_gt_layout_file % image_set) as f:
        gt_layout_list += json.load(f)

num_questions = len(training_questions)
training_images = np.concatenate(training_images_list)

# Shuffle the training data
# fix random seed for data repeatibility
np.random.seed(3)
shuffle_inds = np.random.permutation(num_questions)

training_questions = [training_questions[idx] for idx in shuffle_inds]
training_labels = [training_labels[idx] for idx in shuffle_inds]
training_images = training_images[shuffle_inds]
gt_layout_list = [gt_layout_list[idx] for idx in shuffle_inds]

# number of training batches
num_batches = np.ceil(num_questions / N)

# Turn the questions into vocabulary indices
text_seq_array = np.zeros((T_encoder, num_questions), np.int32)
seq_length_array = np.zeros(num_questions, np.int32)
gt_layout_array = np.zeros((T_decoder, num_questions), np.int32)
for n_q in range(num_questions):
    tokens = training_questions[n_q].split()
    seq_length_array[n_q] = len(tokens)
    for t in range(len(tokens)):
        text_seq_array[t, n_q] = vocab_shape_dict[tokens[t]]
    gt_layout_array[:, n_q] = assembler.module_list2tokens(
        gt_layout_list[n_q], T_decoder)

image_mean = np.load(image_mean_file)
image_array = (training_images - image_mean).astype(np.float32)
vqa_label_array = np.array(training_labels, np.int32)

#setting random seed
np.random.seed(int(time.time()))
tf.set_random_seed(int(time.time()))

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

# Write summary to TensorBoard
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
loss_ph = tf.placeholder(tf.float32, [])
entropy_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
val_acc = tf.placeholder(tf.float32, [], name = "val_acc")
tr_acc = tf.placeholder(tf.float32, [], name = "train_acc")
tf.summary.scalar("training_accuracy", tr_acc)
tf.summary.scalar("validation_accuracy", val_acc)
tf.summary.scalar("avg_sample_loss", loss_ph)
tf.summary.scalar("entropy", entropy_ph)
tf.summary.scalar("avg_accuracy", accuracy_ph)
log_step = tf.summary.merge_all()

os.makedirs(snapshot_dir, exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots

sess.run(tf.global_variables_initializer())

avg_accuracy = 0
accuracy_decay = 0.99

#validation data
validation_questions = []
validation_labels = []
validation_images_list = []
validation_gt_layout_list = []

for image_set in ['val']:
    with open(training_text_files % image_set) as f:
        validation_questions += [l.strip() for l in f.readlines()]
    with open(training_label_files % image_set) as f:
        validation_labels += [l.strip() == 'true' for l in f.readlines()]
    validation_images_list.append(np.load(training_image_files % image_set))
    with open(training_gt_layout_file % image_set) as f:
        validation_gt_layout_list += json.load(f)

num_val_questions = len(validation_questions)
validation_images = np.concatenate(validation_images_list)

num_val_batches = np.ceil(num_val_questions / N)

validation_text_seq_array = np.zeros((T_encoder, num_val_questions), np.int32)
validation_seq_length_array = np.zeros(num_val_questions, np.int32)
validation_gt_layout_array = np.zeros((T_decoder, num_val_questions), np.int32)
for n_q in range(num_val_questions):
    tokens = validation_questions[n_q].split()
    validation_seq_length_array[n_q] = len(tokens)
    for t in range(len(tokens)):
        validation_text_seq_array[t, n_q] = vocab_shape_dict[tokens[t]]
    validation_gt_layout_array[:, n_q] = assembler.module_list2tokens(
        validation_gt_layout_list[n_q], T_decoder)

validation_image_array = (validation_images - image_mean).astype(np.float32)
validation_vqa_label_array = np.array(validation_labels, np.int32)

#varaible access
prefix = 'neural_module_network/layout_execution/'
mods = ['TransformModule', 'FindModule', 'AnswerModule']

swaps = dict.fromkeys(mods)
old = dict.fromkeys(mods, 0)
num_swaps = int(input("Number of swaps?"))
answer_accuracy = 0

for n_iter in range(max_iter):
    n_begin = int((n_iter % num_batches)*N)
    n_end = int(min(n_begin+N, num_questions))

    # set up input and output tensors
    h = sess.partial_run_setup(
        [nmn3_model.predicted_tokens, nmn3_model.entropy_reg,
         scores, avg_sample_loss, train_step],
        [text_seq_batch, seq_length_batch, image_batch, gt_layout_batch,
         compiler.loom_input_tensor, vqa_label_batch])

    # Part 0 & 1: Run Convnet and generate module layout
    tokens, entropy_reg_val = sess.partial_run(h,
        (nmn3_model.predicted_tokens, nmn3_model.entropy_reg),
        feed_dict={text_seq_batch: text_seq_array[:, n_begin:n_end],
                   seq_length_batch: seq_length_array[n_begin:n_end],
                   image_batch: image_array[n_begin:n_end],
                   gt_layout_batch: gt_layout_array[:, n_begin:n_end]})
    # Assemble the layout tokens into network structure
    expr_list, expr_validity_array = assembler.assemble(tokens)
    # all expr should be valid (since they are ground-truth)
    assert(np.all(expr_validity_array))
    labels = vqa_label_array[n_begin:n_end]
    # Build TensorFlow Fold input for NMN
    expr_feed = compiler.build_feed_dict(expr_list)
    expr_feed[vqa_label_batch] = labels

    #initialization for swapping
    if num_swaps != 0 and n_iter == 0:
        for mod in mods:
            swaps[mod] = []
            for i in range(num_swaps):
                d = {}
                for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = prefix + mod):
                    y = x.eval(session = sess)
                    if "bias" in x.name:
                        d[x.name] = np.zeros(y.shape)
                    elif "Adam" in x.name:
                        d[x.name] = np.zeros(y.shape)
                    else:
                        var = tf.Variable(tf.contrib.layers.xavier_initializer()(shape = y.shape))
                        sess.run(tf.local_variables_initializer())
                        sess.run(tf.global_variables_initializer())
                        y = var.eval(session = sess)
                        d[x.name] = np.array(y)
                swaps[mod] += [d]
        with open(os.path.join(snapshot_dir, "start_swaps.txt"), "wb") as f:
            pickle.dump(swaps, f)
        print("initial swaps saved to " + os.path.join(snapshot_dir, "start_swaps.txt"))
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (0))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)

    # Part 2: Run NMN and learning steps
    scores_val, avg_sample_loss_val, _ = sess.partial_run(
        h, (scores, avg_sample_loss, train_step), feed_dict=expr_feed)

    # compute accuracy
    predictions = np.argmax(scores_val, axis=1)
    accuracy = np.mean(np.logical_and(expr_validity_array,
                                      predictions == labels))
    avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)

    # validation once per epoch
    if n_begin == 0:
        answer_correct = 0
        layout_correct = 0
        layout_valid = 0

        for v_iter in range(int(num_val_batches)):
            v_begin = int((v_iter % num_val_batches)*N)
            v_end = int(min(v_begin+N, num_val_questions))

            # set up input and output tensors
            h = sess.partial_run_setup(
                [nmn3_model.predicted_tokens, scores],
                [text_seq_batch, seq_length_batch, image_batch, gt_layout_batch,
                 compiler.loom_input_tensor, expr_validity_batch])

            # Part 0 & 1: Run Convnet and generate module layout
            tokens = sess.partial_run(h, nmn3_model.predicted_tokens,
                feed_dict={text_seq_batch: validation_text_seq_array[:, v_begin:v_end],
                           seq_length_batch: validation_seq_length_array[v_begin:v_end],
                           image_batch: validation_image_array[v_begin:v_end],
                           gt_layout_batch: validation_gt_layout_array[:, v_begin:v_end]})

            # compute the accuracy of the predicted layout
            gt_tokens = validation_gt_layout_array[:, v_begin:v_end]
            layout_correct += np.sum(np.all(np.logical_or(tokens == gt_tokens,
                                                          gt_tokens == assembler.EOS_idx),
                                            axis=0))

            # Assemble the layout tokens into network structure
            expr_list, expr_validity_array = assembler.assemble(tokens)
            layout_valid += np.sum(expr_validity_array)
            labels = validation_vqa_label_array[v_begin:v_end]
            # Build TensorFlow Fold input for NMN
            expr_feed = compiler.build_feed_dict(expr_list)
            expr_feed[expr_validity_batch] = expr_validity_array

            # Part 2: Run NMN and learning steps
            scores_val = sess.partial_run(h, scores, feed_dict=expr_feed)

            # compute accuracy
            predictions = np.argmax(scores_val, axis=1)
            answer_correct += np.sum(np.logical_and(expr_validity_array,
                                                    predictions == labels))

        answer_accuracy = answer_correct / num_val_questions
        print("validation accuracy =", answer_accuracy)
        #summary = sess.run(log_step, {val_acc : answer_accuracy})
        #log_writer.add_summary(summary, n_iter)

    # Add to TensorBoard summary
    if n_iter % log_interval == 0 or (n_iter+1) == max_iter:
        print("iter = %d\n\tloss = %f, accuracy (cur) = %f, "
              "accuracy (avg) = %f, entropy = %f" %
              (n_iter, avg_sample_loss_val, accuracy,
               avg_accuracy, -entropy_reg_val))
        summary = sess.run(log_step, {loss_ph: avg_sample_loss_val,
                                          entropy_ph: -entropy_reg_val,
                                          accuracy_ph: avg_accuracy,
                                          tr_acc: accuracy,
                                          val_acc : answer_accuracy})
        log_writer.add_summary(summary, n_iter)

    # Save snapshot
    if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)
        with open(os.path.join(snapshot_dir, "swaps%d.txt" % (n_iter + 1)), "wb") as f:
            pickle.dump(swaps, f)


    #better swapping
    if num_swaps != 0 and n_iter % 10 == 0: # and n_iter <= 30000:
        for mod in mods:
            new = random.randint(0, num_swaps - 1)
            for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = prefix + mod):
                y = x.eval(session = sess)
                swaps[mod][old[mod]][x.name] = y
                x.load(swaps[mod][new][x.name], session = sess)
            old[mod] = new

with open(os.path.join(snapshot_dir, "all_swaps.txt"), "wb") as f:
    pickle.dump(swaps, f)
