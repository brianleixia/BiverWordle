#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

base_dir = 'data/fyp_origin_data'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/fyp_checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
save_path1 = os.path.join(save_dir, 'best_validation1')  # 最佳验证结果保存路径
save_path2 = os.path.join(save_dir, 'best_validation2')  # 最佳验证结果保存路径
save_path3 = os.path.join(save_dir, 'best_validation3')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session1 = tf.Session()
    session1.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session1, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session1, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session1.run(model.y_pred_cls, feed_dict=feed_dict)

    print(y_pred_cls)
    #np.savetxt('result.txt', y_pred_cls, fmt="%d", delimiter=" ")

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories,digits=10))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def voting():
    print("model voting")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path1)  # 读取保存的模型

    session2 = tf.Session()
    session2.run(tf.global_variables_initializer())
    saver2 = tf.train.Saver()
    saver2.restore(sess=session2, save_path=save_path2)  # 读取保存的模型

    session3 = tf.Session()
    session3.run(tf.global_variables_initializer())
    saver3 = tf.train.Saver()
    saver3.restore(sess=session3, save_path=save_path3)  # 读取保存的模型

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
    y_pred_cls1 = np.empty([len(x_test), 10], dtype=np.float32)  # 保存预测概率1
    y_pred_cls2 = np.empty([len(x_test), 10], dtype=np.float32)  # 保存预测概率2
    y_pred_cls3 = np.empty([len(x_test), 10], dtype=np.float32)  # 保存预测概率3

    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
        #y_pred_cls2[start_id:end_id] = session2.run(model.logits, feed_dict=feed_dict).reshape(128)
        #test = session.run(model.pro, feed_dict=feed_dict)
        #y_pred_cls2 = np.concatenate((y_pred_cls2, test),axis=1)
        y_pred_cls1[start_id:end_id] = session.run(model.pro, feed_dict=feed_dict)
        y_pred_cls2[start_id:end_id] = session2.run(model.pro, feed_dict=feed_dict)
        y_pred_cls3[start_id:end_id] = session3.run(model.pro, feed_dict=feed_dict)

    #print(type(y_pred_cls2))
    #np.savetxt('probability.txt', y_pred_cls2, fmt="%f", delimiter=" ")
    #y_vote1 = np.rint(y_pred_cls1)
    #y_vote2 = np.rint(y_pred_cls2)
    #
    y_ensemble = (y_pred_cls1 + y_pred_cls2 + y_pred_cls3)/3
    print(y_ensemble)
    print("here1!")
    #np.savetxt('result.txt', y_pred_cls, fmt="%d", delimiter=" ")
    y_pred_ensemble = y_ensemble.argmax(axis=1)
    # # Confusion Matrix and report

    print("here2!")
    np.savetxt('predict.txt', y_ensemble, fmt="%f", delimiter=" ")
    np.savetxt('predict_label.txt', y_pred_ensemble, fmt="%d", delimiter=" ")

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_ensemble, target_names=categories, digits=10))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_ensemble)
    print(cm)

def voting2():
    print("model voting")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型


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
    y_pred_cls1 = np.empty([len(x_test), 10], dtype=np.float32)  # 保存预测概率1


    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
        #y_pred_cls2[start_id:end_id] = session2.run(model.logits, feed_dict=feed_dict).reshape(128)
        #test = session.run(model.pro, feed_dict=feed_dict)
        #y_pred_cls2 = np.concatenate((y_pred_cls2, test),axis=1)
        y_pred_cls1[start_id:end_id] = session.run(model.pro, feed_dict=feed_dict)


    #print(type(y_pred_cls2))
    np.savetxt('cnn_probability.txt', y_pred_cls1, fmt="%f", delimiter=" ")

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test','voting']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'voting':
        voting2()
