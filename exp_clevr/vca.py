from __future__ import absolute_import, division, print_function

gpu_id = 0  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import numpy as np
import tensorflow as tf
# Start the session BEFORE importing tensorflow_fold
# to avoid taking up all GPU memory
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    allow_soft_placement=False, log_device_placement=False))

from models_clevr.nmn3_assembler import Assembler
from models_clevr.nmn3_model import NMN3Model
from util.clevr_train.data_reader import DataReader

from models_clevr.nmn3_modules import Modules
from models_clevr.nmn3_assembler import _module_input_num

# Module parameters
H_feat = 10
W_feat = 15
D_feat = 512
embed_dim_txt = 300
embed_dim_nmn = 300
lstm_dim = 512
num_layers = 2
T_encoder = 45
T_decoder = 20
N = 1
prune_filter_module = True

exp_name = "clevr_v0_gt_layout_prune"
snapshot_name = "00600000"
# tst_image_set = 'trn'
tst_image_set = 'val'
snapshot_file = './exp_clevr/tfmodel/%s/%s' % (exp_name, snapshot_name)

# Data files
vocab_question_file = './exp_clevr/data/vocabulary_clevr.txt'
vocab_layout_file = './exp_clevr/data/vocabulary_layout.txt'
vocab_answer_file = './exp_clevr/data/answers_clevr.txt'

imdb_file_tst = './exp_clevr/data/imdb/imdb_%s.npy' % tst_image_set

save_dir = './exp_clevr/results/%s/%s.%s' % (exp_name, snapshot_name, tst_image_set)
os.makedirs(save_dir, exist_ok=True)

assembler = Assembler(vocab_layout_file)

# shuffle data for visualization
np.random.seed(3)
data_reader_tst = DataReader(imdb_file_tst, shuffle=True, one_pass=True,
                             batch_size=N,
                             T_encoder=T_encoder,
                             T_decoder=T_decoder,
                             assembler=assembler,
                             vocab_question_file=vocab_question_file,
                             vocab_answer_file=vocab_answer_file,
                             prune_filter_module=prune_filter_module)

num_vocab_txt = data_reader_tst.batch_loader.vocab_dict.num_vocab
num_vocab_nmn = len(assembler.module_names)
num_choices = data_reader_tst.batch_loader.answer_dict.num_vocab

# Network inputs
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(tf.float32, [None, H_feat, W_feat, D_feat])
expr_validity_batch = tf.placeholder(tf.bool, [None])

# The model for testing
nmn3_model_tst = NMN3Model(
    image_feat_batch, input_seq_batch,
    seq_length_batch, T_decoder=T_decoder,
    num_vocab_txt=num_vocab_txt, embed_dim_txt=embed_dim_txt,
    num_vocab_nmn=num_vocab_nmn, embed_dim_nmn=embed_dim_nmn,
    lstm_dim=lstm_dim, num_layers=num_layers,
    assembler=assembler,
    encoder_dropout=False,
    decoder_dropout=False,
    decoder_sampling=False,
    num_choices=num_choices)

image_feature_grid = nmn3_model_tst.image_feat_grid
word_vecs = nmn3_model_tst.word_vecs
atts = nmn3_model_tst.atts

image_feat_grid_ph = tf.placeholder(tf.float32, image_feature_grid.get_shape())
word_vecs_ph = tf.placeholder(tf.float32, word_vecs.get_shape())

batch_idx = tf.constant([0], tf.int32)
time_idx = tf.placeholder(tf.int32, [1])
input_0 = tf.placeholder(tf.float32, [1, H_feat, W_feat, 1])
input_1 = tf.placeholder(tf.float32, [1, H_feat, W_feat, 1])

# Manually construct each module outside TensorFlow fold for visualization
module_outputs = {}
with tf.variable_scope("neural_module_network/layout_execution", reuse=True):
    modules = Modules(image_feat_grid_ph, word_vecs_ph, num_choices)
    module_outputs['_Scene'] = modules.SceneModule(time_idx, batch_idx)
    module_outputs['_Find'] = modules.FindModule(time_idx, batch_idx)
    module_outputs['_FindSameProperty'] = modules.FindSamePropertyModule(input_0, time_idx, batch_idx)
    module_outputs['_Transform'] = modules.TransformModule(input_0, time_idx, batch_idx)
    module_outputs['_And'] = modules.AndModule(input_0, input_1, time_idx, batch_idx)
    module_outputs['_Filter'] = modules.FilterModule(input_0, time_idx, batch_idx)
    module_outputs['_Or'] = modules.OrModule(input_0, input_1, time_idx, batch_idx)
    module_outputs['_Exist'] = modules.ExistModule(input_0, time_idx, batch_idx)
    module_outputs['_Count'] = modules.CountModule(input_0, time_idx, batch_idx)
    module_outputs['_EqualNum'] = modules.EqualNumModule(input_0, input_1, time_idx, batch_idx)
    module_outputs['_MoreNum'] = modules.MoreNumModule(input_0, input_1, time_idx, batch_idx)
    module_outputs['_LessNum'] = modules.LessNumModule(input_0, input_1, time_idx, batch_idx)
    module_outputs['_SameProperty'] = modules.SamePropertyModule(input_0, input_1, time_idx, batch_idx)
    module_outputs['_Describe'] = modules.DescribeModule(input_0, time_idx, batch_idx)

def eval_module(module_name, inputs, t, image_feat_grid_val, word_vecs_val):
    feed_dict = {image_feat_grid_ph: image_feat_grid_val,
                 word_vecs_ph: word_vecs_val,
                 time_idx: [t]}
    # print('evaluating module ' + module_name)
    if 'input_0' in inputs:
        feed_dict[input_0] = inputs['input_0']
    if 'input_1' in inputs:
        feed_dict[input_1] = inputs['input_1']
    if module_name in module_outputs:
        result = sess.run(module_outputs[module_name], feed_dict)
    else:
        raise ValueError("invalid module name: " + module_name)

    return result

def eval_expr(layout_tokens, image_feat_grid_val, word_vecs_val):
    invalid_scores = np.array([[0, 0]], np.float32)
    # Decoding Reverse Polish Notation with a stack
    decoding_stack = []
    all_output_stack = []
    for t in range(len(layout_tokens)):
        # decode a module/operation
        module_idx = layout_tokens[t]
        if module_idx == assembler.EOS_idx:
            break
        module_name = assembler.module_names[module_idx]
        input_num = _module_input_num[module_name]

        # Get the input from stack
        inputs = {}
        for n_input in range(input_num-1, -1, -1):
            stack_top = decoding_stack.pop()
            inputs["input_%d" % n_input] = stack_top
        result = eval_module(module_name, inputs, t,
                             image_feat_grid_val, word_vecs_val)
        decoding_stack.append(result)
        all_output_stack.append((t, module_name, result[0]))

    assert(len(decoding_stack) == 1)
    result = decoding_stack[0]
    return result, all_output_stack

def expr2str(expr, indent=4):
    name = expr['module']
    input_str = []
    if 'input_0' in expr:
        input_str.append('\n'+' '*indent+expr2str(expr['input_0'], indent+4))
    if 'input_1' in expr:
        input_str.append('\n'+' '*indent+expr2str(expr['input_1'], indent+4))
    expr_str = name[1:]+('[%d]'%expr['time_idx'])+"("+", ".join(input_str)+")"
    return expr_str

snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
snapshot_saver.restore(sess, snapshot_file)

import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams.update({'font.size': 6})
def run_visualization(dataset_tst):
    if dataset_tst is None:
        return
    print('Running test...')
    answer_word_list = dataset_tst.batch_loader.answer_dict.word_list
    vocab_list = dataset_tst.batch_loader.vocab_dict.word_list
    for n, batch in enumerate(dataset_tst.batches()):
        if n >= 100: break
        # set up input and output tensors
        h = sess.partial_run_setup(
            [nmn3_model_tst.predicted_tokens, nmn3_model_tst.scores, word_vecs, atts],
            [input_seq_batch, seq_length_batch, image_feat_batch,
             nmn3_model_tst.compiler.loom_input_tensor, expr_validity_batch])

        # Part 0 & 1: Run Convnet and generate module layout
        tokens, word_vecs_val, atts_val =\
            sess.partial_run(h, (nmn3_model_tst.predicted_tokens, word_vecs, atts),
            feed_dict={input_seq_batch: batch['input_seq_batch'],
                       seq_length_batch: batch['seq_length_batch'],
                       image_feat_batch: batch['image_feat_batch']})
        image_feat_grid_val = batch['image_feat_batch']

        # Assemble the layout tokens into network structure
        expr_list, expr_validity_array = assembler.assemble(tokens)
        labels = batch['answer_label_batch']
        # Build TensorFlow Fold input for NMN
        expr_feed = nmn3_model_tst.compiler.build_feed_dict(expr_list)
        expr_feed[expr_validity_batch] = expr_validity_array

        # Part 2: Run NMN and learning steps
        scores_val = sess.partial_run(h, nmn3_model_tst.scores, feed_dict=expr_feed)

        predictions = np.argmax(scores_val, axis=1)

        # Part 3: Visualization
        print('visualizing %d' % n)
        layout_tokens = tokens.T[0]
        result, all_output_stack = eval_expr(layout_tokens, image_feat_grid_val, word_vecs_val)
        # check that the results are consistent
        diff = np.max(np.abs(result - scores_val))
        assert(np.all(diff < 1e-4))

        encoder_words = [vocab_list[w]
                         for n_w, w in enumerate(batch['input_seq_batch'][:, 0])
                         if n_w < batch['seq_length_batch'][0]]
        decoder_words = [assembler.module_names[w][1:]+'[%d]'%n_w
                         for n_w, w in enumerate(layout_tokens)
                         if w != assembler.EOS_idx]
        atts_val = atts_val[:len(decoder_words), :len(encoder_words)]
        plt.figure(figsize=(12, 12))
        plt.subplot(4, 3, 1)
        plt.imshow(plt.imread(batch['image_path_list'][0]))
        plt.colorbar()
        question = ' '.join(encoder_words[:10]) + '\n' + \
            ' '.join(encoder_words[10:20]) + '\n' + \
            ' '.join(encoder_words[20:30]) + '\n' + \
            ' '.join(encoder_words[30:40]) + '\n' + \
            ' '.join(encoder_words[40:]) + '\n'
        plt.title(question)
        plt.axis('off')
        plt.subplot(4, 3, 2)
        plt.axis('off')
        plt.imshow(np.ones((3, 3, 3), np.float32))
        plt.text(0, 1,
                 'Predicted layout:\n\n' + expr2str(expr_list[0]) +
                 '\n\nlabel: '+ answer_word_list[labels[0]] +
                 '\nprediction: '+ answer_word_list[predictions[0]])
        plt.subplot(4, 3, 3)
        plt.imshow(atts_val.reshape(atts_val.shape[:2]), interpolation='nearest', cmap='Reds')
        plt.xticks(np.arange(len(encoder_words)), encoder_words, rotation=90)
        plt.yticks(np.arange(len(decoder_words)), decoder_words)
        plt.colorbar()
        for t, module_name, results in all_output_stack:
            result = all_output_stack[0][2]
            plt.subplot(4, 3, t+4)
            if results.ndim > 2:
                plt.imshow(results[..., 0], interpolation='nearest', vmin=-1.5, vmax=1.5, cmap='Reds')
                plt.axis('off')
            else:
                plot = np.tile(results.reshape((1, num_choices)), (2, 1))
                plt.imshow(plot, interpolation='nearest', vmin=-1.5, vmax=1.5, cmap='Reds')
                plt.xticks(range(len(answer_word_list)), answer_word_list, rotation=90)
            plt.title('output from '+module_name[1:]+'[%d]'%t)
            plt.colorbar()

        plt.savefig(os.path.join(save_dir, '%08d.jpg' % n))
        plt.close('all')

run_visualization(data_reader_tst)
