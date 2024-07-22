import tensorflow as tf
import numpy as np
import pandas as pd
# from scipy.stats.stats import pearsonr
from tensorflow.contrib import layers
import data_import_cv as data_import 
import os
import shutil
import datetime
import math
import utils_pm25 as utils
import gdal

from data_import_cv import DataSet
from gtnnwr_pm25 import S_T_NETWORK
from gtnnwr_pm25 import GTW_NETWORK
from gtnnwr_pm25 import CNN_GTW_NETWORK
from gtnnwr_pm25 import BASE_NETWORK
from gtnnwr_pm25 import DIAGNOSIS_SLM
from tensorflow.python.client import timeline
import matplotlib as mpl

from shp_process import get_smooth_coord

mpl.use('Agg')  # No display
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
# from scipy.stats import t
import shutil

import rasterio
# import gdal
from affine import Affine
from pyproj import Proj, transform
from functools import reduce
from init_config import ConfigSetting

if __name__ == '__main__':
    Grid_data = False

    training = False
    #
    # training = True
    # raster_compute = False

    train_3d = False
    # train_3d = True

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if training:
        # data_setting_file = "setting_para/setting_data_gtnnwr_simulation.cfg"
        data_setting_file = "setting_para/setting_data_gtnnwr_simulation_st_lv.cfg"
        model_setting_file = "setting_para/setting_model.cfg"
        loop_length = 1
    else:
        # date_strs_gen = ["20210301222504"]
        # data_files_gen = ["simulated_Data_low_high_20191211094904_1"]
        # +时间
        # # temp 456 -0510
        # date_strs_gen = ["20210402094127"]
        # temp 789 -0810
        # date_strs_gen = ["20210402101433"]
        # temp 123   -0310
        # date_strs_gen = ["20210401183738"]

        # temp 1012  -1020
        date_strs_gen = ["20210402112815"]



        # in
        # 170810
        # date_strs_gen = ["20210402152003"]
        # 160810
        # date_strs_gen = ["20210402165249"]

        # 180810
        date_strs_gen = ["20210402185851"]

        data_files_gen = ["3Y789M_all_2018"]
        # data_files_gen = ["18temp_no_zero_1012"]
        data_setting_files = ["setting_data_gtnnwr_simulation_st_lv.cfg"]
        # data_setting_files = ["setting_data_gtnnwr_simulation.cfg"]

        model_setting_files = ["setting_model.cfg"]
        models_gen = ["gtnnwr"]
        repeat_cv_indexs = [0, 10]
        loop_length = len(date_strs_gen)
        repeat_compute = True
        raster_compute = True
        # raster_compute = False




    loop_iter = 0

    while loop_iter < loop_length:
        if not training:
            date_str_gen = date_strs_gen[loop_iter]
            data_file_gen = data_files_gen[loop_iter]
            model_gen = models_gen[loop_iter]

            # data_setting_file = os.path.join(BASE_DIR, "setting_para" , data_file_gen, model_gen,  \
            #                     date_str_gen,  data_setting_files[loop_iter])
            # model_setting_file = os.path.join(BASE_DIR, "setting_para", data_file_gen, model_gen, \
            #                      date_str_gen, model_setting_files[loop_iter])

            data_setting_file = os.path.join(BASE_DIR, "setting_para", data_setting_files[loop_iter])
            model_setting_file = os.path.join(BASE_DIR, "setting_para", model_setting_files[loop_iter])

            repeat_cv_index = repeat_cv_indexs[loop_iter]

        loop_iter = loop_iter + 1
        # global setting
        cp = ConfigSetting(data_setting_path=data_setting_file, model_setting_path=model_setting_file)
        diff_weight, batch_norm, create_force, random_fixed, base_path, log_y, simple_stnn, stannr, seed, \
        iter_num, stop_num, stop_iter, train_r2_cri, \
        test_r2_cri, log_delete, test_model = cp.get_global_para()

        col_data_x, col_data_y, col_coordx, col_coordy, col_coordz, col_coordt, seasons, models, simple_stnns, stannrs, \
        datafile, log_y, normalize_y, train_ratio, validation_ratio, st_weight_init, \
        gtw_weight_init, epochs, start_lr, max_lr, total_up_steps, up_decay_steps, maintain_maxlr_steps, \
        delay_steps, delay_rate, keep_prob_ratio, val_early_stopping, val_early_stopping_begin_step, \
        model_comparison_criterion = cp.get_data_para()

        print("epochs:")
        print(epochs)
        print("models:")
        print(models)
        # print("epochs:" + epochs)
        # print("epochs:" + epochs)
        # print("epochs:" + epochs)

        snn_hidden_layer_count, snn_neural_sizes, snn_output_size, tnn_hidden_layer_count, tnn_neural_sizes, tnn_output_size, \
        stnn_hidden_layer_count, stnn_neural_sizes, stnn_output_size, gtwnn_factor, gtwnn_hidden_node_limit, \
        gtwnn_max_layer_count, kernel_size, kernel_num, smooth_coords_path, x_len, \
        cnngtwnn_factor, cnngtwnn_hidden_node_limit, cnngtwnn_max_layer_count = cp.get_model_structure()

        datafile_origin = datafile

        for model_index in range(len(models) * iter_num):
            if model_index > 0:
                tf.reset_default_graph()

            # season = seasons[int(model_index % len(seasons))]
            # datafile = 'china_sites_2017_pm25_' + season + '.csv'

            model = models[model_index % len(models)]
            date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print('Finding ' + str(stop_iter + 1) + ' excellent result.')

            # 如果不是在训练，应该直接指定数据、模型等信息
            if not training:
                datafile = data_file_gen + '.csv'
                model = model_gen
                date_str = date_str_gen

            if '*' in datafile_origin:
                datafile = datafile_origin.replace('*', str(model_index + 1))

            datafile_name = datafile[0:datafile.rfind('.')]
            # 简单时空network，没有隐含层
            simple_stnn = simple_stnns[model_index % len(models)]

            # 是否包含自相关网络
            stannr = stannrs[model_index % len(models)]

            # model para
            data_path = base_path + datafile
            no_space, s_no_network, no_time, t_no_network, st_no_network, s_each_dir, t_cycle, s_activate_fun, t_activate_fun, st_activate_fun, \
            gtw_activate_fun, no_cnn, dataset_path = cp.get_model_para(model)

            # reading data
            # ！！！！！！！！！！！！！注意改！！！！！！！！date_numeric
            # date_numeric = True;
            # create_force = True
            if train_3d:
                train_sets, val_sets, test_sets, miny, maxy, dataname = data_import.init_dataset_cv_train(data_path,
                                                                                                      train_val_ratio=1,
                                                                                                      cv_fold=10,
                                                                                                      s_each_dir=True,
                                                                                                      t_cycle=t_cycle,
                                                                                                      log_y=log_y,
                                                                                                      normalize_y=normalize_y,
                                                                                                      date_str=date_str,
                                                                                                      create_force=create_force,
                                                                                                      random_fixed=random_fixed,
                                                                                                      seed=seed,
                                                                                                      col_data_x=col_data_x,
                                                                                                      col_data_y=col_data_y,
                                                                                                      col_date=[],
                                                                                                      col_coordx=col_coordx,
                                                                                                      col_coordy=col_coordy,
                                                                                                      col_coordz=col_coordz,
                                                                                                      Grid_data=False)
            else:
                train_sets, val_sets, test_sets, miny, maxy, dataname = data_import.init_dataset_cv_train_2d(data_path,
                                                                                                      train_val_ratio=1,
                                                                                                      cv_fold=10,
                                                                                                      s_each_dir=True,
                                                                                                      t_cycle=t_cycle,
                                                                                                      log_y=log_y,
                                                                                                      normalize_y=normalize_y,
                                                                                                      date_str=date_str,
                                                                                                      create_force=create_force,
                                                                                                      random_fixed=random_fixed,
                                                                                                      seed=seed,
                                                                                                      col_data_x=col_data_x,
                                                                                                      col_data_y=col_data_y,
                                                                                                      col_date=col_coordt,
                                                                                                      col_coordx=col_coordx,
                                                                                                      col_coordy=col_coordy,

                                                                                                      Grid_data=False)




            # y值如果没有归一化，就不需要转换，将最大最小值设为1和0
            if not normalize_y:
                miny = 0
                maxy = 1

            outputs_train = []
            outputs_val = []
            outputs_test = []

            for cv_index in range(len(train_sets)):
                if cv_index!=8:
                    continue
                else:
                    print('Cross Validation ' + str(cv_index + 1))

                # 如果不是在训练，应该直接指定cv_index
                # if not training:
                #     cv_index = 5

                print('Cross Validation ' + str(cv_index + 1))
                # train_val_set = train_val_sets[cv_index]
                train_set = train_sets[cv_index]
                valiset = val_sets[cv_index]
                testset = test_sets[cv_index]
                if Grid_data:
                    grid_datasets = None
                    gridset = grid_datasets[0]

                if cv_index > 0:
                    date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    tf.reset_default_graph()

                # total training and validation datasets
                # x_train_val = train_val_set.x_data
                # y_train_val = train_val_set.y_data
                # s_dis_train_val = train_val_set.space_dis
                # t_dis_train_val = train_val_set.time_dis
                # s_dis_train_val_weight = train_val_set.s_dis_weight

                print(train_set.space_dis.shape)
                print(train_set.space_dis)
                # Training dataset
                x_train = train_set.x_data
                y_train = train_set.y_data
                s_dis_train = train_set.space_dis
                # s_dis_train = trainset.s_dis_weight
                t_dis_train = train_set.time_dis
                s_dis_train_weight = train_set.s_dis_weight
                # Validation dataset
                x_vali = valiset.x_data
                y_vali = valiset.y_data
                s_dis_vali = valiset.space_dis
                # s_dis_vali = valiset.s_dis_weight
                t_dis_vali = valiset.time_dis
                s_dis_vali_weight = valiset.s_dis_weight

                # Testing dataset
                x_test = testset.x_data
                y_test = testset.y_data
                s_dis_test = testset.space_dis
                # s_dis_test = testset.s_dis_weight
                t_dis_test = testset.time_dis
                s_dis_test_weight = testset.s_dis_weight

                # Grid Dataset
                if Grid_data:
                    x_grid = gridset.x_data
                    y_grid = gridset.y_data
                    s_dis_grid = gridset.space_dis
                    # s_dis_test = testset.s_dis_weight
                    t_dis_grid = gridset.time_dis
                    s_dis_grid_weight = gridset.s_dis_weight

                sample_size = x_train.shape[0]

                # dis_size是1
                dis_size = s_dis_train.shape[2]
                # dis_size = s_dis_train.shape[1]
                # dropout的keep_prob
                keep_prob_st = tf.placeholder(tf.float32)
                keep_prob_gtw = tf.placeholder(tf.float32)
                bn_is_training = tf.placeholder(tf.bool)

                gtnnwr_output_size = train_set.x_data.shape[1]
                x_data = tf.placeholder(tf.float32, [None, gtnnwr_output_size])
                y_data = tf.placeholder(tf.float32, [None, 1])
                distance = tf.placeholder(tf.float32, [None, None])
                dis_weight = tf.placeholder(tf.float32, [None, sample_size])

                yhat_train__ = np.matmul(s_dis_train_weight, y_train)
                print(utils.getR2(y_train, yhat_train__))

                if not no_space:
                    s_input_size = s_dis_train.shape[2]

                    s_network = S_T_NETWORK('space', sample_size, s_input_size, snn_hidden_layer_count,
                                            snn_neural_sizes,
                                            s_activate_fun, keep_prob_st, output_layer_size=snn_output_size,
                                            batch_norm=batch_norm, bn_is_training=bn_is_training,
                                            weight_init=st_weight_init, diff_weight=diff_weight,
                                            no_network=s_no_network)

                    snn_output_size = s_network.output_size
                    snn_output = tf.reshape(s_network.output, [-1, sample_size, snn_output_size])
                    snn_x_data = s_network.dist_data
                else:
                    snn_output_size = 0
                    snn_output = None
                    snn_x_data = tf.placeholder(tf.float32, [None, None, None])

                if not no_time:
                    t_input_size = t_dis_train.shape[2]

                    t_network = S_T_NETWORK('time', sample_size, t_input_size, tnn_hidden_layer_count,
                                            tnn_neural_sizes,
                                            t_activate_fun, keep_prob_st, output_layer_size=tnn_output_size,
                                            batch_norm=batch_norm, bn_is_training=bn_is_training,
                                            weight_init=st_weight_init, diff_weight=diff_weight,
                                            no_network=t_no_network)

                    # t_network = S_T_NETWORK('time', sample_size, dis_size, t_input_size, tnn_hidden_layer_count,
                    #                         tnn_neural_sizes,
                    #                         t_activate_fun, keep_prob_st, output_layer_size=tnn_output_size,
                    #                         batch_norm=batch_norm, bn_is_training=bn_is_training,
                    #                         weight_init=st_weight_init, diff_weight=diff_weight,
                    #                         no_network=t_no_network)
                    tnn_output_size = t_network.output_size
                    tnn_output = tf.reshape(t_network.output, [-1, sample_size, tnn_output_size])
                    tnn_x_data = t_network.dist_data
                else:
                    tnn_output_size = 0
                    tnn_output = None
                    tnn_x_data = tf.placeholder(tf.float32, [None, None, None])

                # 对SNN和TNN的输入输出做转换，输入到STNN
                if no_space:
                    st_input = tnn_output
                elif no_time:
                    st_input = snn_output
                else:
                    st_input = tf.concat([snn_output, tnn_output], 2)
                st_input_size = snn_output_size + tnn_output_size
                st_input = tf.reshape(st_input, [-1, st_input_size])

                if simple_stnn:
                    stnn_hidden_layer_count = 0
                    stnn_neural_sizes = [0]
                    stnn_output_size = 1

                st_network = S_T_NETWORK('space_time', sample_size, st_input_size, stnn_hidden_layer_count,
                                         stnn_neural_sizes,
                                         st_activate_fun, keep_prob_st, output_layer_size=stnn_output_size,
                                         dist_data=st_input,
                                         batch_norm=batch_norm,
                                         bn_is_training=bn_is_training, weight_init=st_weight_init,
                                         diff_weight=diff_weight,
                                         no_network=st_no_network)

                gtwnn_hidden_layer_count, gtwnn_neural_sizes = utils.hidden_layers(sample_size, factor=gtwnn_factor,
                                                                                   hidden_node_limit=gtwnn_hidden_node_limit,
                                                                                   max_layer_count=gtwnn_max_layer_count)

                if no_cnn:
                    stnn_output_size = st_network.output_size
                    gtwnn_input = tf.reshape(st_network.output, [-1, sample_size * stnn_output_size])
                    # gtwnn_input = tf.reshape(st_network.output, [-1, dis_size * stnn_output_size])

                    # gtw_network = GTW_NETWORK('gtw_network', gtwnn_input, gtwnn_hidden_layer_count, gtwnn_neural_sizes, gtnnwr_output_size,
                    # ！！！！！【这里缺一个output size的参数】
                    gtw_network = GTW_NETWORK('gtw_network', gtwnn_input, gtwnn_hidden_layer_count, gtwnn_neural_sizes, gtnnwr_output_size,
                                              gtw_activate_fun, keep_prob_gtw, batch_norm=True,
                                              bn_is_training=bn_is_training,
                                              weight_init=gtw_weight_init)

                else:
                    gtwnn_input = tf.reshape(st_network.output, [-1, int(dis_size / x_len), x_len, 1])
                    gtw_network = CNN_GTW_NETWORK('cnn_gtw_network', gtwnn_input, cnngtwnn_factor,
                                                  cnngtwnn_hidden_node_limit, cnngtwnn_max_layer_count,
                                                  gtnnwr_output_size, gtw_activate_fun, keep_prob_gtw, gtw_weight_init,
                                                  kernel_size, kernel_num,
                                                  bn_is_training=bn_is_training)

                # 多元线性回归（要用训练数据集）
                # CV 这里用训练集算的
                linear_beta = tf.squeeze(utils.linear(tf.constant(train_set.x_data, dtype=tf.float32),
                                                      tf.constant(
                                                          np.reshape(train_set.y_data, [len(train_set.y_data), 1]),
                                                          dtype=tf.float32)))
                linear_beta_mat = tf.diag(linear_beta)
                # 回归部分
                # yhat_rg = tf.diag_part(tf.matmul(tf.matmul(x_data, linear_beta_mat), tf.transpose(gtw_network.gtweight)))

                # gtbeta = tf.transpose(tf.matmul(linear_beta_mat, tf.transpose(gtw_network.gtweight)))

                # 自相关部分   【论文部分】
                # 隐藏层以及神经元
                stawnn_hidden_layer_count, stawnn_neural_sizes = utils.hidden_layers_2(sample_size, factor=16,
                                                                                       layer_count=2)
                stawnn_hidden_layer_count = 3
                # stawnn_neural_sizes = [2048, 4096, 8192, 4096, 2048]
                # stawnn_neural_sizes = [2048, 4096, 2048]
                stawnn_neural_sizes = [512, 1024, 512]
                # stawnn_neural_sizes = [1024, 2048, 1024]
                # stawnn_neural_sizes = [512, 2048, 8192, 2048, 512]
                # 输出层的shape
                stawnn_output_size = 1
                # 输入以及相关网络
                stawnn_input = tf.reshape(st_network.output, [-1, sample_size * stnn_output_size])
                # staw_network = GTW_NETWORK_A('staw_network', stawnn_input, stawnn_hidden_layer_count, stawnn_neural_sizes,
                #                            stawnn_output_size,
                #                            gtw_activate_fun, keep_prob_gtw, batch_norm=True, bn_is_training=bn_is_training,
                #                            weight_init=gtw_weigt_init, y_train = tf.constant(np.reshape(s_dis_train, [x_train.shape[0],x_train.shape[0]]), dtype=tf.float32))
                staw_network = GTW_NETWORK('staw_network', stawnn_input, stawnn_hidden_layer_count, stawnn_neural_sizes,
                                           sample_size,
                                           gtw_activate_fun, keep_prob_gtw, batch_norm=True,
                                           bn_is_training=bn_is_training,
                                           weight_init=gtw_weight_init)
                # 输出  【应该就是ρ矩阵?】
                gt_weight = staw_network.gtweight
                # gt_weight = tf.minimum(gt_weight,1)
                # gt_weight = tf.maximum(gt_weight, 0.99)
                # wei = tf.matmul(tf.diag(tf.squeeze(gt_weight)), dis_weight)
                # wei = tf.matmul(tf.diag(tf.squeeze(gt_weight)), dis_weight)
                wei = tf.multiply(gt_weight, dis_weight)
                # wei = gt_weight
                yhat_ar = tf.matmul(wei, tf.constant(np.reshape(train_set.y_data, [len(train_set.y_data), 1]),
                                                     dtype=tf.float32))

                yhat = yhat_ar
                print("!!!! yhat.shape:")
                print(yhat.shape)
                # # 考虑自相关!! 这一块目前还未测试通过
                # if stannr:
                #     stawnn_hidden_layer_count, stawnn_neural_sizes = utils.hidden_layers_2(sample_size, factor=16, layer_count=2)
                #     stawnn_output_size = train_set.x_data.shape[0]
                #     staw_network = GTW_NETWORK('staw_network', gtwnn_input, stawnn_hidden_layer_count, stawnn_neural_sizes, stawnn_output_size,
                #                               gtw_activate_fun, keep_prob_gtw, batch_norm=True, bn_is_training=bn_is_training,
                #                               weight_init=gtw_weight_init)
                #
                #     yhat_ar = tf.matmul(staw_network.gtweight, tf.constant(np.reshape(train_set.y_data, [len(train_set.y_data), 1]), dtype=tf.float32))
                #
                #     yhat_parts = tf.concat([yhat_ar, tf.reshape(yhat_rg, [-1, 1])], 1)
                #
                #     # 回归项 + 自相关项
                #     # yhat_rg = yhat_ar + yhat_rg
                #
                #     # 回归项 + 自相关项 学习
                #     yhat_network = BASE_NETWORK('yhat_network', yhat_parts, 0, [], 1)
                #     yhat = tf.squeeze(yhat_network.outputs)

                # loss = tf.reduce_mean(tf.square(yhat - tf.squeeze(y_data)))
                loss = tf.reduce_mean(tf.square(yhat - y_data))
                cross_loss = -tf.reduce_mean(tf.squeeze(y_data) * tf.log(yhat))

                # 这部分均为GTNNWR的假设检验
                diagnosis = DIAGNOSIS_SLM(x_data, y_data, yhat, linear_beta_mat, wei, miny, maxy, y_train)
                loss_rg = diagnosis.loss_rg
                loss_convert = diagnosis.loss_convert
                loss_add_reg = diagnosis.loss_add_reg
                # 绝对误差与相对误差
                ave_abs_error = diagnosis.ave_abs_error
                ave_rel_error = diagnosis.ave_rel_error

                loss_add_reg = loss_add_reg

                # r2与修正后r2
                r2_pearson = diagnosis.r2_pearson
                r2_coeff = diagnosis.r2_coeff
                # r2_adjusted_coeff = diagnosis.r2_adjusted_coeff
                global_step = tf.Variable(0, trainable=False)
                # learning_rate_decay = tf.train.exponential_decay(start_lr, global_step, delay_steps, delay_rate,
                #                                                  staircase=True)
                # 初始学习率
                # 运行了几轮batch_size的计数器
                # 多少轮batch size后，更新一次学习率（一般为总样本数/batch_size）
                # 学习率衰减率
                # learning_rate_decay = tf.train.exponential_decay(start_lr, global_step, delay_steps, delay_rate,
                #                                                  staircase=True)
                # 自定义学习率，先上升再下降
                print("delay_steps")
                print(delay_steps)
                learning_rate_decay = utils.exponential_decay_norm(start_lr, max_lr, global_step, delay_steps,
                                                                   delay_rate,
                                                                   total_up_steps, up_decay_steps, maintain_maxlr_steps,
                                                                   staircase=True)

                # batch normalization 的初始化工作
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Ensures that we execute the update_ops before performing the train_step
                    train = tf.train.GradientDescentOptimizer(learning_rate_decay).minimize(loss_add_reg,
                                                                                            global_step=global_step)
                    # train = tf.train.AdamOptimizer(1).minimize(loss_rg_add_reg)

                    # train = tf.train.AdamOptimizer(1).minimize(loss_rg_add_reg)

                # 初始化变量
                init = tf.initialize_all_variables()

                feed_train = {x_data: x_train, y_data: y_train, snn_x_data: s_dis_train,
                              tnn_x_data: t_dis_train,
                              keep_prob_st: 1,
                              keep_prob_gtw: 1, bn_is_training: True, dis_weight: s_dis_train_weight}
                feed_val = {x_data: x_vali, y_data: y_vali, snn_x_data: s_dis_vali, tnn_x_data: t_dis_vali,
                            keep_prob_st: 1,
                            keep_prob_gtw: 1, bn_is_training: False, dis_weight: s_dis_vali_weight}
                feed_test = {x_data: x_test, y_data: y_test, snn_x_data: s_dis_test, tnn_x_data: t_dis_test,
                             keep_prob_st: 1,
                             keep_prob_gtw: 1, bn_is_training: False, dis_weight: s_dis_test_weight}


                # feed_train_val = {x_data: x_train_val, y_data: y_train_val, snn_x_data: s_dis_train_val, tnn_x_data: t_dis_train_val,
                #               keep_prob_st: 1, keep_prob_gtw: 1, bn_is_training: False, dis_weight:s_dis_train_val_weight}

                if Grid_data:
                    feed_grid = {x_data: x_grid, y_data: y_grid, snn_x_data: s_dis_grid, tnn_x_data: t_dis_grid,
                                 keep_prob_st: 1,
                                 keep_prob_gtw: 1, bn_is_training: False, dis_weight: s_dis_grid_weight}
                # 存储变量，用于early stopping
                # 这个变量必须放到外面，不能放到里面去，不然restore好像就出问题
                # 存储变量，用于early stopping
                saver = tf.train.Saver()

                if training:
                    # summary
                    log_dir = 'Data/logs/' + datafile_name + '/' + model

                    writer_train = tf.summary.FileWriter(log_dir + '/train/' + date_str, tf.get_default_graph())
                    writer_validation = tf.summary.FileWriter(log_dir + '/validation/' + date_str,
                                                              tf.get_default_graph())
                    writer_test = tf.summary.FileWriter(log_dir + '/test/' + date_str, tf.get_default_graph())

                    summary_loss = tf.summary.scalar('loss', loss)
                    # summary_loss_convert = tf.summary.scalar('loss_real', loss_rg_convert)
                    # summary_loss_add_reg = tf.summary.scalar('loss_rg_add_reg', loss_rg_add_reg)
                    summary_pearson_r2 = tf.summary.scalar('r2_pearson', r2_pearson)
                    summary_coeff_r2 = tf.summary.scalar('r2_coeff', r2_coeff)
                    # summary_coeff_adjust_r2 = tf.summary.scalar('r2_adjusted_coeff', r2_adjusted_coeff)
                    # summary_traceS = tf.summary.scalar('traceS', traceS)
                    # summary_AICc = tf.summary.scalar('AICc', AICc)
                    # summary_AIC = tf.summary.scalar('AIC', AIC)
                    summary_ave_abs_error = tf.summary.scalar('ave_abs_error', ave_abs_error)
                    summary_ave_rel_error = tf.summary.scalar('ave_rel_error', ave_rel_error)
                    # summary_f1 = tf.summary.scalar('f1', f1)
                    # summary_f2 = tf.summary.scalar('f2', f2)
                    # summary_f3 = tf.summary.scalar('f3_param1', f3_dict['f3_param_0'])
                    # summary_f3_2 = tf.summary.scalar('f3_param1_2', f3_dict_2['f3_param_0'])
                    # summary_f3 = tf.summary.scalar('f3_param2', f3_dict['f3_param_1'])
                    # summary_f3 = tf.summary.scalar('f3_param3', f3_dict['f3_param_2'])
                    # summary_f3 = tf.summary.scalar('f3_param4', f3_dict['f3_param_3'])
                    # summary_f3 = tf.summary.scalar('f3_param5', f3_dict['f3_param_4'])
                    # summary_f3 = tf.summary.scalar('f3_param6', f3_dict['f3_param_5'])
                    # summary_z12 = tf.summary.scalar('z12', z12)
                    # summary_z_square = tf.summary.scalar('z_square', z_square)
                    summary_merged = tf.summary.merge_all()

                    # 绘制结果
                    plot_buf_ph = tf.placeholder(tf.string)
                    image = tf.image.decode_png(plot_buf_ph, channels=4)
                    image = tf.expand_dims(image, 0)  # make it batched
                    plot_image_summary = tf.summary.image('result', image, max_outputs=10)

                    save_model_dir = 'Data/model_para/' + datafile_name + '/' + model + '/' + date_str + '/'
                    if os.path.exists(save_model_dir):
                        shutil.rmtree(save_model_dir)
                    os.makedirs(save_model_dir)

                    # 存储模型的配置信息
                    save_setting_para_dir = 'Data/setting_para/' + datafile_name + '/' + model + '/' + date_str + '/'
                    if os.path.exists(save_setting_para_dir):
                        shutil.rmtree(save_setting_para_dir)
                    os.makedirs(save_setting_para_dir)
                    shutil.copy(data_setting_file, save_setting_para_dir)
                    shutil.copy(model_setting_file, save_setting_para_dir)

                    summary_step = 20
                    # early_stop_loss = ave_abs_error
                    early_stop_loss = loss
                    val_early_stop_loss_min = 1000
                    r2_p_max = 0
                    # val_loss_min = 1000
                    # val_loss_noimprove_max_count = int(epochs / summary_step / 10)
                    val_loss_noimprove_max_count = 500
                    val_loss_no_improve_count = 0
                    val_loss_best_step = 0

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    # time_line_file = open('Data/timeline/timeline_01.json', 'w')

                    batch_size = int(train_set.num_examples / 10)
                    batch_size = 2 ** math.floor(math.log2(batch_size))

                    with tf.Session() as sess:
                        sess.run(init)
                        # t_Rs = t.ppf(0.99, sess.run(Rs, feed_dict=feed_train))
                        # confidence_param = weight_i[0][0] + sigma_num * t_Rs
                        # confidence_y = yhat_rg[0] + sigma_num_y * t_Rs
                        # 循环训练
                        for step in range(epochs):
                            batch = train_set.next_batch(batch_size)
                            feed = {x_data: batch[0], y_data: batch[1], snn_x_data: batch[2], tnn_x_data: batch[3],
                                    keep_prob_st: keep_prob_ratio,
                                    keep_prob_gtw: keep_prob_ratio, bn_is_training: True, dis_weight: batch[4]}
                            # if step % delay_steps == 0:
                            #     cur_learning_rate = sess.run(learning_rate_decay)

                            # 生成run_metadata
                            if step % int(epochs / 5) == 0:
                                sess.run(train, feed_dict=feed, options=run_options, run_metadata=run_metadata)
                                writer_train.add_run_metadata(run_metadata, 'step%d' % step)
                            else:
                                sess.run(train, feed_dict=feed)

                            # 另一种生成run_metadata的方式
                            # sess.run(train, feed_dict=feed, options=run_options, run_metadata=run_metadata)
                            # Create the Timeline object, and write it to a json file
                            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            # time_line_file.write(chrome_trace)

                            if step % summary_step == 0:
                                # training summary
                                current_epoch = int(step / 10)

                                # training summary
                                [summary, train_loss, train_r2_p, train_r2_c, yhat_train_temp,
                                 cur_learning_rate, gtweight, y_data_, yhat_] = sess.run(
                                    [summary_merged, loss, r2_pearson, r2_coeff, yhat, learning_rate_decay,
                                     staw_network.gtweight, y_data, yhat],
                                    feed_dict=feed_train)
                                writer_train.add_summary(summary, step)

                                # [train_gtweight_3d, train_hatS_temp, train_hatS, train_traceS, train_AIC] = sess.run(
                                #     [gtweight_3d, hatS_temp, hatS, traceS, AICc], feed_dict=feed_train)

                                # validation summary
                                [summary, val_loss, val_loss_add_reg, val_early_stop_loss, val_r2_p, val_r2_c,
                                 yhat_val_temp] = sess.run(
                                    [summary_merged, loss, loss_add_reg, early_stop_loss, r2_pearson, r2_coeff, yhat],
                                    feed_dict=feed_val)
                                writer_validation.add_summary(summary, step)

                                # test summary
                                [summary, test_loss] = sess.run(
                                    [summary_merged, loss], feed_dict=feed_test)
                                writer_test.add_summary(summary, step)

                                if val_early_stopping:
                                    if step >= val_early_stopping_begin_step:
                                        # if (train_loss) - val_early_stop_loss_min < 0:
                                        if (val_early_stop_loss) - val_early_stop_loss_min < 0:
                                            val_early_stop_loss_min = (val_early_stop_loss)
                                            # val_early_stop_loss_min = (train_loss)
                                            # r2_p_max = train_r2_p
                                            model_file_name = os.path.join(save_model_dir, 'model')
                                            saver.save(sess, model_file_name, global_step=step)
                                            val_loss_no_improve_count = 0
                                            val_loss_best_step = step
                                        else:
                                            val_loss_no_improve_count = val_loss_no_improve_count + 1
                                            if val_loss_no_improve_count >= val_loss_noimprove_max_count:
                                                saver.restore(sess,
                                                              saver.last_checkpoints[len(saver.last_checkpoints) - 1])
                                                break

                                print('Training_' + str(step) + ' loss_rg: ' + str(train_loss) + ', r2_pearson: ' + str(
                                    train_r2_p) + ', r2_coeff: ' + str(train_r2_c) + ', cur_learning_rate: ' + str(
                                    cur_learning_rate) + ', val_loss_best_step: ' + str(val_early_stop_loss_min) + '.')
                                print('Valdation_' + str(step) + ' loss_rg: ' + str(val_loss) + ', r2_pearson: ' + str(
                                    val_r2_p) + ', r2_coeff: ' + str(val_r2_c) + ', r2_max: ' + str(
                                    r2_p_max) + ', val_loss_best_step: ' + str(val_loss_best_step) + '.')

                                # print('GtW:' + str(gtweight))
                        # print(len(saver.last_checkpoints))
                        saver.restore(sess, saver.last_checkpoints[len(saver.last_checkpoints) - 1])
                        print('Cross Validation ' + str(cv_index + 1) + '-----------Restore the best model!-----------')
                        print('Early Stopping Best Step: ' + str(val_loss_best_step))

                        # t_Rs = t.ppf(0.99, sess.run(Rs, feed_dict=feed_train))
                        # print(sess.run([weight_i[0][0], sigma_num, confidence_param], feed_dict=feed_train))
                        # print(sess.run([yhat_rg[0], sigma_num_y, confidence_y], feed_dict=feed_train))
                        # print(sess.run(yhat_rg, feed_dict=feed_train), sess.run(sigma_num, feed_dict=feed_train),sess.run(tf.matmul(hatS_temp, tf.transpose(hatS_temp)), feed_dict=feed_train).shape)

                        # output
                        # output_indictors = [diagnosis.RSS, diagnosis.MS_train, diagnosis.MS_common, r2_coeff, r2_adjusted_coeff,
                        #                     diagnosis.A_R2, AICc, diagnosis.RMSE, ave_abs_error, ave_rel_error, diagnosis.r_pearson]
                        # output_indictor_names = ['RSS', 'MS_degree', 'MS_common', 'R2', 'Adjusted_R2_degree', 'Adjusted_R2', 'AICc',
                        #                          'RMSE', 'MAE', 'MAPE', 'r']
                        # output结果输出
                        output_indictors = [loss, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error]
                        output_indictor_names = ['rmse', 'r2_pear', 'R2', 'MAE', 'MAPE']
                        # training final result
                        [train_loss, yhat_train, train_R2_pearson, train_R2_coeff, train_ave_abs_error,
                         train_ave_rel_error, train_linear_beta, gtweight] = sess.run(
                            [loss, yhat, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error,
                             linear_beta, staw_network.gtweight], feed_dict=feed_train)
                        print('Training dataset loss_rg: ' + str(train_loss) + ', r2_pearson: ' + str(
                            train_R2_pearson) + ', r2_coeff: ' + str(train_R2_coeff) + '.')

                        plot_buf = utils.get_plot_actual_pred_buf(y_train, yhat_train)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_train.add_summary(image_summary, global_step=0)

                        plot_buf = utils.get_plot_actual_pred_buf(y_train, yhat_train, convert=True, miny=miny,
                                                                  maxy=maxy)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_train.add_summary(image_summary, global_step=1)

                        # validation final result
                        [val_loss, yhat_val, val_R2_pearson, val_R2_coeff, val_ave_abs_error,
                         val_ave_rel_error] = sess.run(
                            [loss, yhat, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error],
                            feed_dict=feed_val)
                        print(
                                'Validation dataset loss_rg: ' + str(val_loss) + ', r2_pearson: ' + str(
                            val_R2_pearson) + ', r2_coeff: ' + str(
                            val_R2_coeff) + '.')

                        plot_buf = utils.get_plot_actual_pred_buf(y_vali, yhat_val)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_validation.add_summary(image_summary, global_step=0)

                        plot_buf = utils.get_plot_actual_pred_buf(y_vali, yhat_val, convert=True, miny=miny, maxy=maxy)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_validation.add_summary(image_summary, global_step=1)

                        plot_buf = utils.get_plot_actual_pred_buf(y_vali, yhat_val)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_validation.add_summary(image_summary, global_step=0)

                        plot_buf = utils.get_plot_actual_pred_buf(y_vali, yhat_val, convert=True, miny=miny, maxy=maxy)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_validation.add_summary(image_summary, global_step=1)

                        # training and validation final result
                        # [train_val_loss, yhat_train_val, train_val_R2_pearson, train_val_R2_coeff, gtweight_train_val, gtbeta_train_val,
                        #  train_val_ave_abs_error, train_val_ave_rel_error, train_val_f1, train_val_f2, train_val_linear_beta, train_val_f3_dict] = sess.run(
                        #     [loss, yhat, r2_pearson, r2_coeff, gtw_network.gtweight, gtbeta, ave_abs_error, ave_rel_error, f1, f2,
                        #      linear_beta, f3_dict], feed_dict=feed_train_val)
                        # print('Training and Validation datasets loss_rg: ' + str(train_val_loss) + ', r2_pearson: ' + str(
                        #     train_val_R2_pearson) + ', r2_coeff: ' + str(train_val_R2_coeff) + '.')

                        # test result
                        [test_loss, yhat_test, test_R2_pearson, test_R2_coeff, test_ave_abs_error,
                         test_ave_rel_error] = sess.run(
                            [loss, yhat, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error],
                            feed_dict=feed_test)
                        print(
                                'Testing dataset loss_rg: ' + str(test_loss) + ', r2_pearson: ' + str(
                            test_R2_pearson) + ', r2_coeff: ' + str(
                            test_R2_coeff) + '.')

                        plot_buf = utils.get_plot_actual_pred_buf(y_test, yhat_test)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_test.add_summary(image_summary, global_step=0)

                        plot_buf = utils.get_plot_actual_pred_buf(y_test, yhat_test, convert=True, miny=miny, maxy=maxy)
                        image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
                        writer_test.add_summary(image_summary, global_step=1)

                        if Grid_data:
                            [yhat_grid] = sess.run([yhat], feed_dict=feed_grid)
                            print('Grid_y_pre' + str(yhat_grid))
                        # 结果输出
                        output_indictors_train = sess.run(output_indictors, feed_dict=feed_train)
                        output_indictors_val = sess.run(output_indictors, feed_dict=feed_val)
                        output_indictors_test = sess.run(output_indictors, feed_dict=feed_test)

                        outputs_train.append(output_indictors_train)
                        outputs_val.append(output_indictors_val)
                        outputs_test.append(output_indictors_test)

                        # 把结果存储到excel中
                        yhat_train_ar = pd.DataFrame(yhat_train)
                        writer_yhat_ar = pd.ExcelWriter('Data/dataset/yhat_train' + str(cv_index + 1) + '.xls')
                        yhat_train_ar.to_excel(writer_yhat_ar, 'sheet1')
                        yhat_val_ar = pd.DataFrame(yhat_val)
                        yhat_val_ar.to_excel(writer_yhat_ar, 'sheet2')
                        yhat_test_ar = pd.DataFrame(yhat_test)
                        yhat_test_ar.to_excel(writer_yhat_ar, 'sheet3')
                        if Grid_data:
                            yhat_grid_ar = pd.DataFrame(yhat_grid)
                            yhat_grid_ar.to_excel(writer_yhat_ar, 'sheet4')
                        writer_yhat_ar.save()
                        print(utils.getR2(y_train, yhat_train))

                        good_res = 0
                        if train_R2_coeff >= train_r2_cri and test_R2_coeff >= test_r2_cri:
                            # stop_iter = stop_iter + 1
                            good_res = 1

                        # # 随机后的原始数据存储路径
                        # file_save_path = dataset_path + dataname + '_' + str(cv_index+1) + '.xls'
                        # # 结果输出路径
                        # result_save_path = 'Data/results/' + datafile_name + '/' + model + '/'
                        # if not os.path.exists(result_save_path):
                        #     os.makedirs(result_save_path)
                        # result_file = result_save_path + dataname + '_' + date_str + '_' + str(good_res) + '.xls'
                        #
                        # rb = open_workbook(file_save_path)
                        # wb = copy(rb)
                        #
                        # # wb = utils.add_result_excel(wb, rb, 'train_val', maxy, miny, yhat_train_val, gtbeta_train_val, gtweight_train_val,
                        # #                             index=0, col_data_x=col_data_x, col_data_y=col_data_y, add_col=0)
                        # wb = utils.add_result_excel(wb, rb, 'train', maxy, miny, yhat_train, gtbeta_train, gtweight_train,
                        #                             index=0, col_data_x=col_data_x, col_data_y=col_data_y, add_col=0)
                        # wb = utils.add_result_excel(wb, rb, 'validation', maxy, miny, yhat_val, gtbeta_val, gtweight_val,
                        #                             index=len(yhat_train), col_data_x=col_data_x, col_data_y=col_data_y, add_col=0)
                        # wb = utils.add_result_excel(wb, rb, 'test', maxy, miny, yhat_test, gtbeta_test, gtweight_test,
                        #                             index=len(yhat_train) + len(yhat_val), col_data_x=col_data_x,
                        #                             col_data_y=col_data_y, add_col=0)
                        #
                        # worksheet = wb.get_sheet('result')
                        # worksheet.write(0, 0, 'result')
                        #
                        # for i in range(len(output_indictor_names)):
                        #     worksheet.write(0, 1 + i, output_indictor_names[i])
                        #     worksheet.write(1, 1 + i, str(output_indictors_train[i]))
                        #     worksheet.write(2, 1 + i, str(output_indictors_val[i]))
                        #     worksheet.write(3, 1 + i, str(output_indictors_test[i]))
                        #
                        # curindex = len(output_indictor_names) + 1
                        #
                        # worksheet.write(0, curindex, 'f1')
                        # worksheet.write(1, curindex, train_f1.item())
                        # curindex = curindex + 1
                        # worksheet.write(0, curindex, 'f2')
                        # worksheet.write(1, curindex, train_f2.item())
                        # curindex = curindex + 1
                        # for i in range(len(train_f3_dict)):
                        #     worksheet.write(0, curindex, 'f3_param_' + str(i))
                        #     worksheet.write(1, curindex, train_f3_dict['f3_param_' + str(i)].item())
                        #     curindex = curindex + 1
                        #
                        # worksheet.write(1, 0, 'train')
                        # worksheet.write(2, 0, 'val')
                        # worksheet.write(3, 0, 'test')
                        #
                        # for i in range(len(col_data_x)):
                        #     worksheet.write(5, i, 'linear_' + col_data_x[i] + '_weight')
                        #     worksheet.write(6, i, train_linear_beta[i].item())
                        #
                        # worksheet.write(5, len(col_data_x), 'linear_constant_weight')
                        # worksheet.write(6, len(col_data_x), train_linear_beta[len(col_data_x)].item())
                        #
                        # if os.path.exists(result_file):
                        #     os.remove(result_file)
                        # wb.save(result_file)



                        # 不知道这个raster compute是做什么的 ------------------------------lv
                        # 需要加时间距离....好复杂
                        if raster_compute:
                            print("---------- training: raster_compute -------------")
                            # 输入
                            file_list = data_import.raster_dataset_list(data_path, dataset_path, s_each_dir=s_each_dir,
                                                                        t_cycle=t_cycle, seed=seed,
                                                                        raster_compute_x=False)
                            yhat_raster = []
                            gtbeta_raster = []
                            gtweight_raster = []

                            for file in file_list:
                                raster_set, raster_index = data_import.open_raster_dataset(file)
                                # Training dataset
                                x_raster = raster_set.x_data
                                s_dis_raster = raster_set.space_dis
                                s_dis_weight_raster = raster_set.s_dis_weight

                                print(s_dis_raster.shape)
                                raster_len = len(s_dis_raster)
                                current_index = 0
                                while raster_len / 100000 - current_index > 0:
                                    start = current_index * 100000
                                    if raster_len / 100000 - current_index < 1:
                                        end = raster_len
                                    else:
                                        end = (current_index + 1) * 100000

                                    print(file + ': ' + str(start) + '-' + str(end))
                                    # tmp_x_raster = x_raster[start:end, :]
                                    tmp_s_dis_raster = s_dis_raster[start:end]
                                    s_dis_weight_raster = s_dis_weight_raster[start:end]

                                    feed_raster = {snn_x_data: tmp_s_dis_raster,
                                                   keep_prob_st: 1, keep_prob_gtw: 1, bn_is_training: False,
                                                   dis_weight: s_dis_weight_raster}
                                    # feed_test = {x_data: x_test, y_data: y_test, snn_x_data: s_dis_test,
                                    #              tnn_x_data: t_dis_test,
                                    #              keep_prob_st: 1,
                                    #              keep_prob_gtw: 1, bn_is_training: False, dis_weight: s_dis_test_weight}
                                    [tmp_yhat_raster] = sess.run([yhat], feed_dict=feed_raster)

                                    if len(yhat_raster) == 0:
                                        yhat_raster = tmp_yhat_raster
                                        # gtbeta_raster = tmp_gtbeta_raster
                                        # gtweight_raster = tmp_gtweight_raster
                                    else:
                                        yhat_raster = np.concatenate((yhat_raster, tmp_yhat_raster))
                                        # gtbeta_raster = np.concatenate((gtbeta_raster, tmp_gtbeta_raster))
                                        # gtweight_raster = np.concatenate((gtweight_raster, tmp_gtweight_raster))

                                    current_index = current_index + 1

                            raster_name = 'EXTRACT_DATA/zj_mask_big.tif'
                            output_path = 'Final_Result/' + model + '/'
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)

                            dest_names = col_data_y

                            ds = gdal.Open(raster_name)
                            print(ds.GetProjection())
                            gt = ds.GetGeoTransform()

                            yhat_raster = yhat_raster * (maxy - miny) + miny

                            # writer_yhat_ar = pd.ExcelWriter('Data/dataset/yhat_train1500' + str(cv_index + 1) + '.xls')
                            # yhat_grid_ar = pd.DataFrame(yhat_raster)
                            # yhat_grid_ar.to_excel(writer_yhat_ar, 'sheet4')
                            # writer_yhat_ar.save()

                            yhat_raster[np.isnan(yhat_raster)] = -9999
                            print(yhat_raster.shape)
                            yhat_raster = np.reshape(yhat_raster, [len(yhat_raster), 1])

                            dest_datas = yhat_raster

                            # Read raster
                            with rasterio.open(raster_name) as r:
                                data = r.read()  # pixel values
                                data_shape = data.shape

                            # all_data_grid = get_smooth_coord(2536553.973442, -2017792.558142, 5812841.266425,
                            #                                  2215746.487588, 40)
                            # # 读取格网构造输入
                            # # all_data_grid = pd.read_csv('Data/PM25/smooth_coords_300.csv', engine='python')
                            # all_data_grid.dropna(inplace=True)
                            # all_data_grid = pd.DataFrame(all_data_grid)
                            #
                            # # 空间坐标列
                            # data_coordx_all_grid = all_data_grid['ProjX']
                            # data_coordy_all_grid = all_data_grid['ProjY']

                            for res_i in range(len(dest_names)):
                                # data_value = np.reshape(dest_datas,[])
                                # data_value = np.ones([data_shape[1], data_shape[2]]) * -9999
                                # data_value = data_value.ravel()
                                # data_value[raster_index] = dest_datas[:, res_i]
                                data_value = np.reshape(dest_datas, [-1, x_len])

                                driver = gdal.GetDriverByName('GTiff')
                                filename = output_path + model + '_result_' + dest_names[
                                    res_i] + '_' + dataname + '_' + date_str + '.tif'
                                dataset = driver.Create(filename, data_value.shape[1] + 1, data_value.shape[0] + 1, 1,
                                                        gdal.GDT_Float32)
                                # dataset = driver.Create(filename, 501, 781, 1,
                                #                         gdal.GDT_Float32)
                                # get_smooth_coord(2720340.960900, -2254509.200100, 6095653.216400, 1950000.487588, 1500)

                                # ynum = int(x_len * (6095653.216400 - 1950000.487588) / (2720340.960900 + 2254509.200100) + 1)
                                # ynum = int(
                                #     x_len * (3429918.330000 - 2993202.730000) / (824329.150000 - 544095.650000) + 1)
                                # ynum = int(
                                #     x_len * (2930000 - 187000) / (4530000 - 867000) + 1)
                                ynum = int(
                                    x_len * (4389982 - 161206) / (5225227 - 22033) + 1)
                                # ynum = int(
                                #     x_len * (3429918.330000 - 2994539.070000) / (824329.150000 - 544095.650000) + 1)

                                # data_value = np.transpose(data_value)

                                # dataset.SetGeoTransform((-2254509.200100,(2720340.960900 + 2254509.200100)/x_len, 0, 6095653.216400, 0, (1950000.487588 - 6095653.216400)/ynum))
                                # dataset.SetGeoTransform((544095.650000, (824329.150000 - 544095.650000) / x_len, 0, 3429918.330000, 0, (2993202.730000 - 3429918.330000) / ynum))
                                # dataset.SetGeoTransform((100, 0.01, 0, 100, 0, -0.01))
                                dataset.SetGeoTransform((22033, (5225227 - 22033) / x_len, 0,
                                                         4389982, 0,
                                                         (161206 - 4389982) / ynum))
                                # dataset.SetGeoTransform((544095.650000, (824329.150000 - 544095.650000) / x_len, 0,
                                #                          3429918.330000, 0, (2994539.070000 - 3429918.330000) / ynum))
                                dataset.SetProjection(ds.GetProjection())
                                # print(data_value.shape)
                                # data_value1 = np.ones([780,500])
                                dataset.GetRasterBand(1).WriteArray(data_value)
                                dataset.GetRasterBand(1).SetNoDataValue(-9999)
                                dataset.FlushCache()

                                raster_name_1 = filename
                                ds1 = gdal.Open(raster_name_1)
                                gt1 = ds1.GetGeoTransform()
                else:
                    # not training
                    print("**** interpolation ****")
                    date_str = date_str_gen
                    save_model_dir = 'Data/model_para/' + datafile_name + '/' + model + '/' + date_str + '/'
                    with tf.Session() as sess:
                        sess.run(init)
                        ckpt = tf.train.get_checkpoint_state(save_model_dir)
                        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
                        if ckpt and ckpt.model_checkpoint_path:
                            saver.restore(sess, ckpt.model_checkpoint_path)

                            if repeat_compute:
                                # # output
                                # output_indictors = [diagnosis.RSS, diagnosis.MS_train, diagnosis.MS_common, r2_coeff,
                                #                     r2_adjusted_coeff, diagnosis.A_R2, AICc, diagnosis.RMSE, ave_abs_error, ave_rel_error,
                                #                     diagnosis.r_pearson]
                                # output_indictor_names = ['RSS', 'MS_degree', 'MS_common', 'R2', 'Adjusted_R2_degree', 'Adjusted_R2',
                                #                          'AICc', 'RMSE', 'MAE', 'MAPE', 'r']
                                # output结果输出
                                output_indictors = [loss, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error]
                                output_indictor_names = ['rmse', 'r2_pear', 'R2', 'MAE', 'MAPE']

                                # training final result
                                [train_loss, yhat_train, train_R2_pearson, train_R2_coeff, train_ave_abs_error,
                                 train_ave_rel_error, train_linear_beta, gtweight] = sess.run(
                                    [loss, yhat, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error,
                                     linear_beta, staw_network.gtweight], feed_dict=feed_train)
                                print('Training dataset loss_rg: ' + str(train_loss) + ', r2_pearson: ' + str(
                                    train_R2_pearson) + ', r2_coeff: ' + str(train_R2_coeff) + '.')

                                # validation final result
                                # validation final result
                                [val_loss, yhat_val, val_R2_pearson, val_R2_coeff, val_ave_abs_error,
                                 val_ave_rel_error] = sess.run(
                                    [loss, yhat, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error],
                                    feed_dict=feed_val)
                                print(
                                        'Validation dataset loss_rg: ' + str(val_loss) + ', r2_pearson: ' + str(
                                    val_R2_pearson) + ', r2_coeff: ' + str(
                                    val_R2_coeff) + '.')

                                # training and validation final result
                                # [train_val_loss, yhat_train_val, train_val_R2_pearson, train_val_R2_coeff,
                                #  gtweight_train_val, gtbeta_train_val,
                                #  train_val_ave_abs_error, train_val_ave_rel_error, train_val_f1, train_val_f2,
                                #  train_val_linear_beta, train_val_f3_dict] = sess.run(
                                #     [loss, yhat, r2_pearson, r2_coeff, gtw_network.gtweight, gtbeta, ave_abs_error,
                                #      ave_rel_error, f1, f2,
                                #      linear_beta, f3_dict], feed_dict=feed_train_val)
                                # print('Training and Validation datasets loss_rg: ' + str(
                                #     train_val_loss) + ', r2_pearson: ' + str(
                                #     train_val_R2_pearson) + ', r2_coeff: ' + str(train_val_R2_coeff) + '.')

                                # test result
                                [test_loss, yhat_test, test_R2_pearson, test_R2_coeff, test_ave_abs_error,
                                 test_ave_rel_error] = sess.run(
                                    [loss, yhat, r2_pearson, r2_coeff, ave_abs_error, ave_rel_error],
                                    feed_dict=feed_test)
                                print(
                                        'Testing dataset loss_rg: ' + str(test_loss) + ', r2_pearson: ' + str(
                                    test_R2_pearson) + ', r2_coeff: ' + str(
                                    test_R2_coeff) + '.')
                                # 结果输出
                                output_indictors_train = sess.run(output_indictors, feed_dict=feed_train)
                                output_indictors_val = sess.run(output_indictors, feed_dict=feed_val)
                                output_indictors_test = sess.run(output_indictors, feed_dict=feed_test)

                            if raster_compute:
                                print("raster_compute:")
                                # 输入
                                file_list = data_import.raster_dataset_list(data_path, dataset_path,
                                                                            s_each_dir=s_each_dir, t_cycle=t_cycle,
                                                                            seed=seed, raster_compute_x=False)
                                yhat_raster = []
                                gtbeta_raster = []
                                gtweight_raster = []

                                for file in file_list:
                                    raster_set, raster_index = data_import.open_raster_dataset(file)
                                    # Training dataset
                                    x_raster = raster_set.x_data
                                    s_dis_raster \
                                        = raster_set.space_dis
                                    s_dis_weight_raster = raster_set.s_dis_weight
                                    # -------------lv 313 -----------------
                                    t_dis_raster = raster_set.time_dis
                                    # ------------end

                                    print(s_dis_raster.shape)
                                    raster_len = len(s_dis_raster)
                                    current_index = 0
                                    while raster_len / 100000 - current_index > 0:
                                        start = current_index * 100000
                                        if raster_len / 100000 - current_index < 1:
                                            end = raster_len
                                        else:
                                            end = (current_index + 1) * 100000

                                        print(file + ': ' + str(start) + '-' + str(end))
                                        # tmp_x_raster = x_raster[start:end, :]
                                        tmp_s_dis_raster = s_dis_raster[start:end]
                                        s_dis_weight_raster = s_dis_weight_raster[start:end]


                                        # lv  ------------ 313
                                        # feed_raster = {snn_x_data: tmp_s_dis_raster,
                                        #                keep_prob_st: 1, keep_prob_gtw: 1, bn_is_training: False,
                                        #                dis_weight: s_dis_weight_raster}
                                        tmp_t_dis_raster = t_dis_raster[start:end]
                                        feed_raster = {snn_x_data: tmp_s_dis_raster,
                                                     tnn_x_data: tmp_t_dis_raster,
                                                     keep_prob_st: 1,
                                                     keep_prob_gtw: 1, bn_is_training: False, dis_weight: s_dis_weight_raster}
                                        # 算出来结果一样
                                        [tmp_yhat_raster] = sess.run([yhat], feed_dict=feed_raster)

                                        if len(yhat_raster) == 0:
                                            yhat_raster = tmp_yhat_raster
                                            # gtbeta_raster = tmp_gtbeta_raster
                                            # gtweight_raster = tmp_gtweight_raster
                                        else:
                                            yhat_raster = np.concatenate((yhat_raster, tmp_yhat_raster))
                                            # gtbeta_raster = np.concatenate((gtbeta_raster, tmp_gtbeta_raster))
                                            # gtweight_raster = np.concatenate((gtweight_raster, tmp_gtweight_raster))

                                        current_index = current_index + 1
                                raster_name = 'EXTRACT_DATA/zj_mask_big.tif'
                                output_path = 'Final_Result/' + model + '/'
                                if not os.path.exists(output_path):
                                    os.makedirs(output_path)

                                dest_names = col_data_y

                                ds = gdal.Open(raster_name)
                                print(ds.GetProjection())
                                gt = ds.GetGeoTransform()

                                # # ------------------- vl -----------------
                                # yhat_raster_max = np.max(yhat_raster)
                                # yhat_raster_min = np.min(yhat_raster)
                                # # 缩放到[0,1]
                                # yhat_raster = (yhat_raster - yhat_raster_min)/( yhat_raster_max -  yhat_raster_min)
                                # # --------------------------end



                                # writer_yhat_ar = pd.ExcelWriter('Data/dataset/yhat_train1500' + str(cv_index + 1) + '.xls')
                                # yhat_grid_ar = pd.DataFrame(yhat_raster)
                                # yhat_grid_ar.to_excel(writer_yhat_ar, 'sheet4')
                                # writer_yhat_ar.save()

                                yhat_raster = yhat_raster * (maxy - miny) + miny
                                yhat_raster[np.isnan(yhat_raster)] = -9999
                                print(yhat_raster.shape)
                                yhat_raster = np.reshape(yhat_raster, [len(yhat_raster), 1])



                                dest_datas = yhat_raster

                                # Read raster
                                with rasterio.open(raster_name) as r:
                                    data = r.read()  # pixel values
                                    data_shape = data.shape

                                for res_i in range(len(dest_names)):
                                    # data_value = np.reshape(dest_datas,[])
                                    # data_value = np.ones([data_shape[1], data_shape[2]]) * -9999
                                    # data_value = data_value.ravel()
                                    # data_value[raster_index] = dest_datas[:, res_i]
                                    # x_len = 500
                                    # !!!!!!!!!!!!注意改！！！！！！！！！！
                                    x_len = 150
                                    coords_xmin = 2040704.101
                                    coords_xmax = 2275542.952
                                    coords_ymin = 3092464.708
                                    coords_ymax = 3570306.098

                                    data_value = np.reshape(dest_datas, [-1, x_len])

                                    driver = gdal.GetDriverByName('GTiff')
                                    filename = output_path + model + '_result_' + dest_names[
                                        res_i] + '_' + dataname + '_' + date_str + '50.tif'
                                    print(data_value.shape)
                                    dataset = driver.Create(filename, data_value.shape[1], data_value.shape[0],
                                                            1, gdal.GDT_Float32)
                                    # dataset = driver.Create(filename, 501, 781, 1,
                                    #                         gdal.GDT_Float32)
                                    # get_smooth_coord(2720340.960900, -2254509.200100, 6095653.216400, 1950000.487588, 1500)

                                    # ynum = int(x_len * (6095653.216400 - 1950000.487588) / (2720340.960900 + 2254509.200100) + 1)
                                    # ynum = int(
                                    #     x_len * (3350000 - 188000) / (5225227 - 22033) + 1)
                                    # ynum = 304

                                    ynum = int(
                                        x_len * (coords_ymax - coords_ymin) / (coords_xmax-coords_xmin) + 1)
                                    print(ynum)
                                    # ynum = int(
                                    #     x_len * (3429918.330000 - 2994539.070000) / (824329.150000 - 544095.650000) + 1)

                                    # data_value = np.transpose(data_value)

                                    # dataset.SetGeoTransform((-2254509.200100,(2720340.960900 + 2254509.200100)/x_len, 0, 6095653.216400, 0, (1950000.487588 - 6095653.216400)/ynum))
                                    # dataset.SetGeoTransform((544095.650000, (824329.150000 - 544095.650000) / x_len, 0, 3429918.330000, 0, (2993202.730000 - 3429918.330000) / ynum))
                                    # dataset.SetGeoTransform((100, 0.01, 0, 100, 0, -0.01))
                                    # dataset.SetGeoTransform((22033, (5225227 - 22033) / x_len, 0,
                                    #                          3350000, 0,
                                    #                          (188000 - 3350000) / ynum))
                                    # dataset.SetGeoTransform((115, (160 - 115) / x_len, 0,
                                    #                          30, 0,
                                    #                          (1 - 30) / ynum))

                                    # GeoTransform[0],GeoTransform[3]  左上角位置
                                    # GeoTransform[1]是像元宽度
                                    # GeoTransform[5]是像元高度
                                    # 如果影像是指北的,GeoTransform[2]和GeoTransform[4]这两个参数的值为0。
                                    dataset.SetGeoTransform((coords_xmin , (coords_xmax - coords_xmin ) / x_len, 0,
                                                             coords_ymax, 0,
                                                             (coords_ymin - coords_ymax) / ynum))

                                    # dataset.SetGeoTransform((100, 0.01, 0, 100, 0, -0.01))

                                    # dataset.SetGeoTransform((544095.650000, (824329.150000 - 544095.650000) / x_len, 0,
                                    #                          3429918.330000, 0, (2994539.070000 - 3429918.330000) / ynum))
                                    dataset.SetProjection(ds.GetProjection())
                                    # print(data_value.shape)
                                    # data_value1 = np.ones([780,500])
                                    dataset.GetRasterBand(1).WriteArray(data_value)
                                    dataset.GetRasterBand(1).SetNoDataValue(-9999)
                                    dataset.FlushCache()

                                    print("*********** finish ***********")
                                    # raster_name_1 = filename
                                    # ds1 = gdal.Open(raster_name_1)
                                    # gt1 = ds1.GetGeoTransform()
                                # for res_i in range(len(dest_names)):
                                #     data_value = np.ones([data_shape[1], data_shape[2]]) * -9999
                                #     data_value = data_value.ravel()
                                #     data_value[raster_index] = dest_datas[:, res_i]
                                #     data_value = np.reshape(data_value, [data_shape[1], data_shape[2]])
                                #
                                #     driver = gdal.GetDriverByName('GTiff')
                                #     filename = output_path + model + '_result_' + dest_names[res_i] + '_' + dataname + '_' + date_str + '.tif'
                                #     dataset = driver.Create(filename, data_shape[2], data_shape[1], 1, gdal.GDT_Float32)
                                #     dataset.SetGeoTransform(gt)
                                #     dataset.SetProjection(ds.GetProjection())
                                #     dataset.GetRasterBand(1).WriteArray(data_value)
                                #     dataset.GetRasterBand(1).SetNoDataValue(-9999)
                                #     dataset.FlushCache()

                    break

            if not training:
                break

            outputs_train = np.array(outputs_train)
            outputs_val = np.array(outputs_val)
            outputs_test = np.array(outputs_test)

            cv_train_result = outputs_train.mean(axis=0)
            cv_val_result = outputs_val.mean(axis=0)
            cv_test_result = outputs_test.mean(axis=0)

            print('Cross Validation Result:')
            print('Training dataset loss: ' + str(cv_train_result[0]) + ', r2_coeff: ' + str(cv_train_result[1]) + '.')
            print('Validation dataset loss: ' + str(cv_val_result[0]) + ', r2_coeff: ' + str(cv_val_result[1]) + '.')
            print('Testing dataset loss: ' + str(cv_test_result[0]) + ', r2_coeff: ' + str(cv_test_result[1]) + '.')

            train_R2_coeff = cv_test_result[1]
            test_R2_coeff = cv_test_result[1]

            xls = xlwt.Workbook()
            worksheet = xls.add_sheet('result')
            worksheet.write(0, 0, 'result')
            for i in range(len(output_indictor_names)):
                worksheet.write(0, 1 + i, output_indictor_names[i])

            for cv_index in range(len(outputs_train)):
                worksheet.write(1 + cv_index * 3, 0, 'train_' + str(cv_index + 1))
                worksheet.write(2 + cv_index * 3, 0, 'val_' + str(cv_index + 1))
                worksheet.write(3 + cv_index * 3, 0, 'test_' + str(cv_index + 1))

                for i in range(len(output_indictor_names)):
                    worksheet.write(1 + cv_index * 3, 1 + i, str(outputs_train[cv_index][i]))
                    worksheet.write(2 + cv_index * 3, 1 + i, str(outputs_val[cv_index][i]))
                    worksheet.write(3 + cv_index * 3, 1 + i, str(outputs_test[cv_index][i]))

            if not os.path.exists('Data/results/' + datafile_name + '/' + model):
                os.makedirs('Data/results/' + datafile_name + '/' + model)
            xls.save('Data/results/' + datafile_name + '/' + model + '/' + dataname + '_' + date_str + '_cv.xls')

            if train_R2_coeff >= train_r2_cri and test_R2_coeff >= test_r2_cri:
                stop_iter = stop_iter + 1
                if stop_iter >= stop_num:
                    break
