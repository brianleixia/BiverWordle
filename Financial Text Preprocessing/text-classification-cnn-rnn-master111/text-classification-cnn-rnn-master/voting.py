from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from rnn_model import TRNNConfig, TextRNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from rnn_model import TRNNConfig

base_dir = 'data/fyp_rnn'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/fyp_checkpoints/textcnn'
save_dir2 = 'checkpoints/fyp_checkpoints/textrnn'

save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
save_path2 = os.path.join(save_dir2, 'best_validation')  # 最佳验证结果保存路径


def voting():
    print("model voting")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)


    # print('Testing...')
    # loss_test, acc_test = evaluate(session, x_test, y_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))
    #
    # print('Testing2...')
    # loss_test, acc_test = evaluate(session2, x_test, y_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    y_pred_cls1 = np.loadtxt('cnn_probability.txt')  # 保存预测概率1
    y_pred_cls2 = np.loadtxt('rnn_probability.txt')  # 保存预测概率2
    y_pred_cls3 = np.loadtxt('bert_probability.txt')  # 保存预测概率2


    #print(type(y_pred_cls2))
    #np.savetxt('probability.txt', y_pred_cls2, fmt="%f", delimiter=" ")
    #y_vote1 = np.rint(y_pred_cls1)
    #y_vote2 = np.rint(y_pred_cls2)
    #
    y_ensemble = (y_pred_cls1 + y_pred_cls2)/2

    #np.savetxt('result.txt', y_pred_cls, fmt="%d", delimiter=" ")

    y_pred_ensemble = y_ensemble.argmax(axis=1)
    # # Confusion Matrix and report

    print("here2!")

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_ensemble, target_names=categories, digits=4))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_ensemble)
    print(cm)


if __name__ == '__main__':

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    voting()