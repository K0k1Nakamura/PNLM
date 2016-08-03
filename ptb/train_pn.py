#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
import six

import six.moves.cPickle as pickle

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import chainer.functions as F

import net
import numpy as np
import word2vec


parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=39, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=650, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help='learning minibatch size')
parser.add_argument('--bproplen', '-l', type=int, default=35,
                    help='length of truncated BPTT')
parser.add_argument('--gradclip', '-c', type=int, default=5,
                    help='gradient norm threshold to clip')
parser.add_argument('--production', dest='test', action='store_false')
parser.set_defaults(test=True)

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch   # number of epochs
n_units = args.unit  # number of units per layer
batchsize = args.batchsize   # minibatch size
bprop_len = args.bproplen   # length of truncated BPTT
grad_clip = args.gradclip    # gradient norm threshold to clip

# Prepare dataset (preliminary download dataset by ./download.py)
vocab = {}
w2v = word2vec.load('ptb.train.bin')


def load_data(filename):
    words = open(filename).read().strip().split()
    dataset = np.ndarray((len(words), 100), dtype=np.float32)
    wordset = []
    for i, word in enumerate(words):
        try:
            dataset[i] = w2v[word]
            wordset.append(word)
        except:
            dataset[i] = w2v['<unk>']
            wordset.append('<unk>')

    return dataset, wordset


train_data, train_word = load_data('ptb.train.txt')
if args.test:
    train_data = train_data[:100]
    train_word = train_word[:100]
valid_data, valid_word = load_data('ptb.valid.txt')
if args.test:
    valid_data = valid_data[:100]
    valid_word = valid_word[:100]
test_data, test_word = load_data('ptb.test.txt')
if args.test:
    test_data = test_data[:100]
    test_data = test_data[:100]


# Prepare RNNLM model, defined in net.py
model = net.PNLM(100, n_units)
for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)


def evaluate(dataset, wordset):
    # Evaluation routine
    evaluator = model.copy()  # to use different state
    evaluator.reset_state()  # initialize state
    evaluator.train = False  # dropout does nothing

    # sum_log_perp = 0
    correct100 = 0.0
    correct10 = 0.0
    correct5 = 0.0
    correct1 = 0.0
    pred = xp.zeros((1,100)).astype(np.float32)

    for i in six.moves.range(len(dataset) - 1):
        x = chainer.Variable(xp.asarray([dataset[i]]).astype(np.float32) - pred, volatile='on')
        result = evaluator(x).data[0]
        pred = result

        inner = np.dot(w2v.vectors, chainer.cuda.to_cpu(result))

        result_cpu = chainer.cuda.to_cpu(inner)
        idx_arr = result_cpu.argsort()[::-1]

        idx = np.where(w2v.vocab == wordset[i+1])[0][0]

        if idx in idx_arr[:100]:
            correct100 += 1
        if idx in idx_arr[:10]:
            correct10 += 1
        if idx in idx_arr[:5]:
            correct5 += 1
        if idx in idx_arr[:1]:
            correct1 += 1

    correct100 /= dataset.size - 1
    correct10 /= dataset.size - 1
    correct5 /= dataset.size - 1
    correct1 /= dataset.size - 1

    return correct1, correct5, correct10, correct100


# Learning loop
whole_len = len(train_data)
jump = whole_len // batchsize
cur_log_loss = 0
epoch = 0
start_at = time.time()
cur_at = start_at
accum_loss = 0
batch_idxs = list(range(batchsize))
print('going to train {} iterations'.format(jump * n_epoch))

pred = xp.zeros((20, 100)).astype(np.float32)

for i in six.moves.range(jump * n_epoch):
    x_arr = xp.asarray(
        [train_data[(jump * j + i) % whole_len] for j in batch_idxs]).astype(np.float32)
    x = chainer.Variable(x_arr - pred)
    t = chainer.Variable(xp.asarray(
        [train_data[(jump * j + i + 1) % whole_len] for j in batch_idxs]).astype(np.float32))
    y = model(x)
    loss_i = F.mean_squared_error(y, t)
    accum_loss += loss_i
    cur_log_loss += loss_i.data
    pred = y.data

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        model.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = 0
        optimizer.update()

    if (i + 1) % 10000 == 0:
        now = time.time()
        throuput = 10000. / (now - cur_at)
        print('iter {} training loss: {} ({:.2f} iters/sec)'.format(
            i + 1, cur_log_loss, throuput))
        cur_at = now
        cur_log_loss = 0

    if (i + 1) % jump == 0:
        epoch += 1
        print('evaluate')
        now = time.time()
        cr1, cr5, cr10, cr100 = evaluate(valid_data, valid_word)
        print('epoch {} validation correct rate(100): {:.5f}'.format(epoch, cr100))
        print('epoch {} validation correct rate(10): {:.5f}'.format(epoch, cr10))
        print('epoch {} validation correct rate(5): {:.5f}'.format(epoch, cr5))
        print('epoch {} validation correct rate(1): {:.5f}'.format(epoch, cr1))
        cur_at += time.time() - now  # skip time of evaluation

        if epoch >= 6:
            optimizer.lr /= 1.2
            print('learning rate =', optimizer.lr)

    sys.stdout.flush()

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data, test_word)
cr1, cr5, cr10, cr100 = evaluate(valid_data, valid_word)
print('epoch {} test correct rate(100): {:.5f}'.format(epoch, cr100))
print('epoch {} test correct rate(10): {:.5f}'.format(epoch, cr10))
print('epoch {} test correct rate(5): {:.5f}'.format(epoch, cr5))
print('epoch {} test correct rate(1): {:.5f}'.format(epoch, cr1))

# Save the model and the optimizer
print('save the model')
serializers.save_npz('w2v.model', model)
print('save the optimizer')
serializers.save_npz('w2v.state', optimizer)
