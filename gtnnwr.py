import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import utils


class S_T_NETWORK():
    def __init__(self, network_name, sample_size, input_size, hidden_layer_count, hidden_layer_neurals, hidden_activate_fun,
                 keep_prop, batch_norm, bn_is_training, output_layer_size=1, output_acitvate_fun=tf.identity, weight_init='std', dist_data=None, diff_weight=False,
                 no_network=False):
        self.network_name = network_name
        self.sample_size = sample_size
        self.input_size = input_size
        self.activate_fun = hidden_activate_fun
        self.output_acitvate_fun = output_acitvate_fun
        self.keep_prop = keep_prop
        self.output_size = output_layer_size

        if weight_init == 'std':
            weight_init_std = True
        else:
            weight_init_std = False

        if dist_data is None:
            self.dist_data = tf.placeholder(tf.float32, [None, self.sample_size, self.input_size])
        else:
            self.dist_data = dist_data

        if no_network:
            self.output = self.dist_data
            self.output_size = self.input_size
        else:
            with tf.variable_scope(network_name):
                hidden_W_name_base = 'hidden_W'
                hidden_b_name_base = 'hidden_b'
                output_W_name_base = 'output_W'
                output_b_name_base = 'output_b'
                if not diff_weight:
                    dist_data_input = tf.reshape(self.dist_data, [-1, self.input_size])
                    if hidden_layer_count > 0:
                        for layer_index in range(hidden_layer_count):
                            if layer_index == 0:
                                input_size = self.input_size
                                input_layer = dist_data_input
                                factor = 2.0
                            else:
                                input_size = hidden_layer_neurals[layer_index - 1]
                                input_layer = hidden_layer
                                factor = 2.0

                            # hidden layer
                            if weight_init_std:
                                hidden_W = utils.init_weight_std(hidden_W_name_base + str(layer_index),
                                                                 [input_size, hidden_layer_neurals[layer_index]])
                            else:
                                hidden_W = utils.init_weight_he(hidden_W_name_base + str(layer_index),
                                                                [input_size, hidden_layer_neurals[layer_index]], input_size, factor)

                            hidden_b = utils.init_bias(hidden_b_name_base + str(layer_index),
                                                       [hidden_layer_neurals[layer_index]])

                            hidden_layer = utils.fully_connected(self.network_name + '_network_hidden_layer_' + str(layer_index),
                                input_layer,
                                hidden_W, hidden_b, batch_norm, bn_is_training)
                            if self.activate_fun == "prelu":
                                hidden_layer = utils.para_relu(hidden_layer, "prelu" + str(layer_index),
                                                               init_value=0.1)
                            else:
                                hidden_layer = self.activate_fun(hidden_layer)

                            hidden_layer = tf.nn.dropout(hidden_layer, self.keep_prop)

                        last_hidden_layer_size = [hidden_layer_neurals[hidden_layer_count - 1], self.output_size]
                    else:
                        last_hidden_layer_size = [self.input_size, self.output_size]
                        hidden_layer = dist_data_input

                    # output layer
                    if weight_init_std:
                        output_W = utils.init_weight_std(output_W_name_base,
                                                         last_hidden_layer_size)
                    else:
                        output_W = utils.init_weight_he(output_W_name_base,
                                                        last_hidden_layer_size, last_hidden_layer_size[0])

                    output_b = utils.init_bias(output_b_name_base, [self.output_size])

                    cur_st_output = utils.fully_connected(self.network_name + '_network_output_layer', hidden_layer,
                                                          output_W,
                                                          output_b)
                    cur_st_output = self.output_acitvate_fun(cur_st_output)
                    self.output = tf.reshape(cur_st_output, [-1, sample_size])

                else:
                    # 时空模型都不同时，由于训练比较慢，最多只考虑一个隐含层
                    self.neural_size = hidden_layer_neurals[0]
                    for index in range(self.sample_size):
                        cur_x_data = self.dist_data[:, index * self.input_size:(index + 1) * self.input_size]
                        hidden_W_name = hidden_W_name_base + '_' + str(index)
                        hidden_b_name = hidden_b_name_base + '_' + str(index)
                        output_W_name = output_W_name_base + '_' + str(index)
                        output_b_name = output_b_name_base + '_' + str(index)

                        # hidden layer
                        hidden_W = utils.init_weight_std(hidden_W_name, [self.input_size, self.neural_size])
                        hidden_b = utils.init_bias(hidden_b_name, [self.neural_size])

                        # output layer
                        output_W = utils.init_weight_std(output_W_name, [self.neural_size, 1])
                        output_b = utils.init_bias(output_b_name, [1])

                    hidden_layer = utils.fully_connected(self.network_name + '_network_hidden_layer', cur_x_data, hidden_W,
                                                         hidden_b, batch_norm, bn_is_training)
                    if self.activate_fun == "prelu":
                        hidden_layer = utils.para_relu(hidden_layer, "prelu" + str(index), init_value=0.1)
                    else:
                        hidden_layer = self.activate_fun(hidden_layer)
                    hidden_layer = tf.nn.dropout(hidden_layer, self.keep_prop)

                    cur_st_output = utils.fully_connected(self.network_name + '_network_output_layer', hidden_layer,
                                                          output_W,
                                                          output_b, batch_norm, bn_is_training)
                    cur_st_output = self.output_acitvate_fun(cur_st_output)
                    if index == 0:
                        self.output = cur_st_output
                    else:
                        self.output = tf.concat([self.output, cur_st_output], 1)


class GTW_NETWORK():
    def __init__(self, network_name, st_network_input, hidden_layer_count, hidden_layer_neurals, output_size, activate_fun,
                 keep_prop, batch_norm, bn_is_training, weight_init='std'):
        self.network_name = network_name
        self.sample_size = st_network_input.shape[1].value
        self.output_size = output_size
        self.activate_fun = activate_fun
        self.keep_prob = keep_prop

        if weight_init == 'std':
            weight_init_std = True
        else:
            weight_init_std = False

        with tf.variable_scope(network_name):
            for layer_index in range(hidden_layer_count):
                if layer_index == 0:
                    input_size = self.sample_size
                    input_layer = st_network_input
                    factor = 2.0
                else:
                    input_size = hidden_layer_neurals[layer_index - 1]
                    input_layer = hidden_layer
                    factor = 2.0

                # hidden layer
                if weight_init_std:
                    hidden_W = utils.init_weight_std('hidden_W_' + str(layer_index),
                                                     [input_size, hidden_layer_neurals[layer_index]])
                else:
                    hidden_W = utils.init_weight_he('hidden_W_' + str(layer_index),
                                                    [input_size, hidden_layer_neurals[layer_index]], input_size, factor)

                hidden_b = utils.init_bias('hidden_b_' + str(layer_index), [hidden_layer_neurals[layer_index]])

                hidden_layer = utils.fully_connected(network_name + '_hidden_layer_' + str(layer_index), input_layer, hidden_W,
                                                     hidden_b, batch_norm, bn_is_training)

                if self.activate_fun == "prelu":
                    hidden_layer = utils.para_relu(hidden_layer, "prelu" + str(layer_index))
                else:
                    hidden_layer = self.activate_fun(hidden_layer)
                hidden_layer = tf.nn.dropout(hidden_layer, keep_prop)

            # output layer
            if weight_init_std:
                output_W = utils.init_weight_std('output_W',
                                                 [hidden_layer_neurals[hidden_layer_count - 1], self.output_size])
            else:
                output_W = utils.init_weight_he('output_W',
                                                [hidden_layer_neurals[hidden_layer_count - 1], self.output_size],
                                                hidden_layer_neurals[hidden_layer_count - 1])

            output_b = utils.init_bias('output_b', [self.output_size])

            self.gtweight = utils.fully_connected(network_name + '_output_layer', hidden_layer, output_W, output_b)

class GTW_NETWORK_A():
    def __init__(self, network_name, st_network_input, hidden_layer_count, hidden_layer_neurals, output_size, activate_fun,
                 keep_prop, batch_norm, bn_is_training, weight_init='std', y_train = None):
        self.network_name = network_name
        self.sample_size = st_network_input.shape[1].value
        self.output_size = output_size
        self.activate_fun = activate_fun
        self.keep_prob = keep_prop
        self.y_train = y_train

        if weight_init == 'std':
            weight_init_std = True
        else:
            weight_init_std = False

        with tf.variable_scope(network_name):
            for layer_index in range(hidden_layer_count):
                if layer_index == 0:
                    input_size = self.sample_size
                    input_layer = st_network_input
                    input_y = self.y_train
                    factor = 2.0
                else:
                    input_size = hidden_layer_neurals[layer_index - 1]
                    input_layer = hidden_layer
                    input_y = W_2_layer
                    factor = 2.0

                # hidden layer
                if weight_init_std:
                    hidden_W = utils.init_weight_std('hidden_W_' + str(layer_index),
                                                     [input_size, hidden_layer_neurals[layer_index]])
                else:
                    hidden_W = utils.init_weight_he('hidden_W_' + str(layer_index),
                                                    [input_size, hidden_layer_neurals[layer_index]], input_size, factor)

                hidden_b = utils.init_bias('hidden_b_' + str(layer_index), [hidden_layer_neurals[layer_index]])

                hidden_layer = utils.fully_connected(network_name + '_hidden_layer_' + str(layer_index), input_layer, hidden_W,
                                                     hidden_b, batch_norm, bn_is_training)

                W_2_layer = utils.fully_connected(network_name + '_W_2_layer_' + str(layer_index), input_y, hidden_W,
                                                     hidden_b, batch_norm, bn_is_training)
                if self.activate_fun == "prelu":
                    hidden_layer = utils.para_relu(hidden_layer, "prelu" + str(layer_index))
                    W_2_layer = utils.para_relu(W_2_layer, "w2prelu" + str(layer_index))
                else:
                    hidden_layer = self.activate_fun(hidden_layer)
                    W_2_layer = self.activate_fun(W_2_layer)
                hidden_layer = tf.nn.dropout(hidden_layer, keep_prop)

            # output layer
            if weight_init_std:
                output_W = utils.init_weight_std('output_W',
                                                 [hidden_layer_neurals[hidden_layer_count - 1], self.output_size])
            else:
                output_W = utils.init_weight_he('output_W',
                                                [hidden_layer_neurals[hidden_layer_count - 1], self.output_size],
                                                hidden_layer_neurals[hidden_layer_count - 1])

            output_b = utils.init_bias('output_b', [self.output_size])

            # self.gtweight = utils.fully_connected(network_name + '_output_layer', hidden_layer, output_W, output_b)
            self.gtweight =  utils.fully_connected(network_name + '_output_layer', hidden_layer, output_W, output_b)
            self.W_2 = utils.fully_connected(network_name + '_output_layer', W_2_layer, output_W, output_b)

class BASE_NETWORK():
    def __init__(self, network_name, network_input, hidden_layer_count, hidden_layer_neurals, output_size, activate_fun=tf.identity,
                 keep_prop=1.0, batch_norm=False, bn_is_training=False, weight_init='std'):
        self.network_name = network_name
        self.sample_size = network_input.shape[1].value
        self.output_size = output_size
        self.activate_fun = activate_fun
        self.keep_prob = keep_prop

        if weight_init == 'std':
            weight_init_std = True
        else:
            weight_init_std = False

        with tf.variable_scope(network_name):
            if hidden_layer_count > 0:
                for layer_index in range(hidden_layer_count):
                    if layer_index == 0:
                        input_size = self.sample_size
                        input_layer = network_input
                        factor = 2.0
                    else:
                        input_size = hidden_layer_neurals[layer_index - 1]
                        input_layer = hidden_layer
                        factor = 2.0

                    # hidden layer
                    if weight_init_std:
                        hidden_W = utils.init_weight_std('hidden_W_' + str(layer_index),
                                                         [input_size, hidden_layer_neurals[layer_index]])
                    else:
                        hidden_W = utils.init_weight_he('hidden_W_' + str(layer_index),
                                                        [input_size, hidden_layer_neurals[layer_index]], input_size, factor)

                    hidden_b = utils.init_bias('hidden_b_' + str(layer_index), [hidden_layer_neurals[layer_index]])

                    hidden_layer = utils.fully_connected(network_name + '_hidden_layer_' + str(layer_index), input_layer, hidden_W,
                                                         hidden_b, batch_norm, bn_is_training)
                    if self.activate_fun == "prelu":
                        hidden_layer = utils.para_relu(hidden_layer, "prelu" + str(layer_index))
                    else:
                        hidden_layer = self.activate_fun(hidden_layer)
                    hidden_layer = tf.nn.dropout(hidden_layer, keep_prop)
                last_hidden_layer_size = [hidden_layer_neurals[hidden_layer_count - 1], self.output_size]

            else:
                last_hidden_layer_size = [self.sample_size, self.output_size]
                hidden_layer = network_input

            # output layer
            if weight_init_std:
                output_W = utils.init_weight_std('output_W', last_hidden_layer_size)
            else:
                output_W = utils.init_weight_he('output_W', last_hidden_layer_size, last_hidden_layer_size[0])

            output_b = utils.init_bias('output_b', [self.output_size])

            self.outputs = utils.fully_connected(network_name + '_output_layer', hidden_layer, output_W, output_b)


class DIAGNOSIS():
    def __init__(self, x_data, y_data, yhat, linear_beta_mat, gtweight=None , miny=None, maxy=None):
        self.x_data = x_data
        self.y_data = y_data
        self.linear_beta_mat = linear_beta_mat
        self.gtweight = gtweight

        with tf.variable_scope('DIAGNOSIS'):
            self.yhat = yhat
            # self.yhat_rg = self.yhat_rg = tf.matmul(self.gtweight,self.y_data)
            self.yhat_rg = tf.diag_part(tf.matmul(tf.matmul(x_data, linear_beta_mat), tf.transpose(gtweight)))

            # 正则化项
            # regularizer = layers.l1_l2_regularizer(0.0, 0.0005)
            regularizer = layers.l1_l2_regularizer(0.0, 0.0)
            weight_list = [weight for weight in tf.global_variables() if weight.name.find('_W') != -1]
            loss_reg = layers.apply_regularization(regularizer, weight_list)

            # 最小化方差
            self.loss_rg = tf.reduce_mean(tf.square(self.yhat_rg - tf.squeeze(y_data)))
            self.loss_rg_add_reg = self.loss_rg + loss_reg

            self.y_data_convert = y_data * tf.constant((maxy - miny), dtype=tf.float32) + tf.constant(miny, dtype=tf.float32)
            self.yhat_rg_convert = self.yhat_rg * tf.constant((maxy - miny), dtype=tf.float32) + tf.constant(miny, dtype=tf.float32)
            self.loss_rg_convert = tf.reduce_mean(tf.square(self.yhat_rg_convert - tf.squeeze(self.y_data_convert)))

            # 帽子矩阵
            x_data_shape = tf.shape(x_data)
            n_size = x_data_shape[0]
            x_data_tile = tf.tile(x_data, [n_size, 1])
            x_data_tile = tf.reshape(x_data_tile, [n_size, n_size, x_data_shape[1]])
            x_data_tile_t = tf.transpose(x_data_tile, perm=[0, 2, 1])
            gtweight_3d = tf.matrix_diag(gtweight)
            # gtweight_3d_convert = tf.matrix_diag(gtweight) * tf.constant((maxy - miny), dtype=tf.float32)

            hatS_temp = tf.matmul(gtweight_3d, tf.matmul(tf.matrix_inverse(tf.matmul(x_data_tile_t, x_data_tile)), x_data_tile_t))
            hatS = tf.matmul(tf.reshape(x_data, [-1, 1, x_data_shape[1]]), hatS_temp)
            hatS = tf.reshape(hatS, [-1, n_size])
            traceS = tf.reduce_sum(tf.diag_part(hatS))
            # traceS = traceS * tf.constant((maxy - miny), dtype=tf.float32)
            # yhat_from_hat = tf.matmul(hatS, y_data)

            # AIC
            n_size_f = tf.cast(n_size, tf.float32)
            const_2 = tf.constant(2, dtype=tf.float32)
            const_2pi = tf.constant(2 * 3.1415)
            self.AICc = const_2 * n_size_f * tf.log(tf.sqrt(self.loss_rg_convert)) + n_size_f * tf.log(
                const_2pi) + n_size_f * (n_size_f + traceS) / (n_size_f - const_2 - traceS)
            self.AIC = const_2 * n_size_f * tf.log(tf.sqrt(self.loss_rg_convert)) + n_size_f * tf.log(
                const_2pi) + n_size_f + traceS

            # 统计检验量（F1，F2，F3等）
            I = tf.matrix_diag(tf.ones(shape=[n_size]))
            # OLS的帽子矩阵等
            hat_OLS = tf.matmul(x_data,
                                tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(x_data), x_data)), tf.transpose(x_data)))
            R_OLS = tf.matmul(tf.transpose(I - hat_OLS), (I - hat_OLS))
            RSS_OLS = tf.matmul(tf.transpose(y_data), tf.matmul(R_OLS, y_data))
            trace_OLS = tf.trace(R_OLS)

            # GTNNWR帽子矩阵等
            hatMatrix = hatS
            R = tf.matmul(tf.transpose(I - hatMatrix), (I - hatMatrix))
            RSS1 = tf.matmul(tf.transpose(y_data), tf.matmul(R, y_data))
            traceR1 = tf.trace(R)

            # F1检验
            self.f1 = tf.squeeze(tf.divide(tf.divide(RSS1, traceR1), tf.divide(RSS_OLS, trace_OLS)))

            # F2检验
            self.f2 = tf.squeeze(tf.divide(tf.divide((RSS_OLS - RSS1), trace_OLS - traceR1), tf.divide(RSS_OLS, trace_OLS)))

            # F3检验
            ek_zeros = np.zeros([6])
            ek_dict = {}
            self.f3_dict = {}
            self.f3_dict_2 = {}
            for i in range(x_data.shape[1]):
                ek_zeros[i] = 1
                ek_dict['ek' + str(i)] = tf.reshape(tf.reshape(tf.tile(tf.constant(ek_zeros), [n_size]), [n_size, -1]),
                                                    [-1, 1, x_data_shape[1]])
                hatB = tf.matmul(tf.cast(ek_dict['ek' + str(i)], dtype=tf.float32), hatS_temp)
                hatB = tf.reshape(hatB, [-1, n_size])

                J_n = tf.cast(1 / n_size, tf.float32) * tf.ones([n_size, n_size], dtype=tf.float32)
                L = tf.matmul(tf.transpose(hatB), tf.matmul(I - J_n, hatB))

                vk2 = tf.cast(1 / n_size, tf.float32) * tf.matmul(tf.transpose(y_data), tf.matmul(L, y_data))
                trace_L = tf.trace(tf.cast(1 / n_size, tf.float32) * L)
                f3 = tf.squeeze(tf.divide(tf.divide(vk2, trace_L), tf.divide(RSS1, traceR1)))
                self.f3_dict['f3_param_' + str(i)] = f3

                bk = tf.matmul(hatB, y_data)
                vk2_2 = tf.cast(1 / n_size, tf.float32) * tf.reduce_sum(tf.square(bk - tf.reduce_mean(bk)))
                f3_2 = tf.squeeze(tf.divide(tf.divide(vk2_2, trace_L), tf.divide(RSS1, traceR1)))
                self.f3_dict_2['f3_param_' + str(i)] = f3_2

            # model comparison criterion
            # y_sub_gnnwr = tf.abs(tf.subtract(tf.reshape(self.yhat_rg, [n_size, 1]), y_data))
            # y_sub_ols = tf.abs(tf.subtract(tf.matmul(x_data, tf.reshape(linear_beta, [x_data.shape[1], 1])), y_data))
            # f12_boolean = tf.logical_and(y_sub_gnnwr <= model_comparison_criterion, y_sub_ols > model_comparison_criterion)
            # f12_int = tf.cast(f12_boolean, tf.int32)
            # f12 = tf.cast(tf.count_nonzero(f12_boolean), tf.float32)
            #
            # f21_boolean = tf.logical_and(y_sub_gnnwr > model_comparison_criterion, y_sub_ols <= model_comparison_criterion)
            # f21_int = tf.cast(f21_boolean, tf.int32)
            # f21 = tf.cast(tf.count_nonzero(f21_boolean), tf.float32)
            #
            # z12 = tf.divide(f12 - f21, tf.sqrt(f12 + f21))
            #
            # z_square = tf.divide(tf.square(tf.abs(f12 - f21) - 1), f12 + f21)

            # 回归参数的置信区间&预测值的置信区间
            # weight_i = tf.diag(gtweight[0])
            # C = tf.matmul(weight_i, tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(x_data), x_data)), tf.transpose(x_data)))
            # D = tf.matmul(C, tf.transpose(C))
            # dkki = D[0][0]
            # self.Rs = tf.divide(tf.square(traceR1), tf.trace(tf.matmul(R, R)))
            # self.sigma_num = tf.sqrt(tf.divide(RSS1, traceR1) * dkki)
            # x_0 = x_data[0, :]
            # x_0 = tf.reshape(x_0, [1, -1])
            #
            # s0 = tf.matmul(x_0, tf.matmul(weight_i, tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(x_data), x_data)), weight_i)))
            # s0 = tf.matmul(s0, tf.transpose(x_0))
            # self.sigma_num_y = tf.sqrt(tf.matmul(tf.divide(RSS1, traceR1), 1 +

            self.loss = tf.reduce_mean(tf.square(self.yhat - tf.squeeze(y_data)))
            self.loss_add_reg = self.loss + loss_reg
            self.yhat_convert = self.yhat * tf.constant((maxy - miny), dtype=tf.float32) + tf.constant(miny, dtype=tf.float32)
            self.loss_convert = tf.reduce_mean(tf.square(self.yhat_convert - tf.squeeze(self.y_data_convert)))

            # R2
            self.r2_pearson = utils.pearson_r2_tensor_cal(tf.squeeze(y_data), self.yhat)
            self.r2_coeff = utils.coeff_r2_tensor_cal(tf.squeeze(y_data), self.yhat)
            # 调整后的R2
            # traceR1 竟然大于n！！
            self.r2_adjusted_coeff = utils.coeff_r2_adjusted_tensor_cal(tf.squeeze(y_data), self.yhat, traceR1, n_size_f - tf.constant(1.0))

            # 平均绝对误差&平均相对误差
            self.ave_abs_error = tf.reduce_mean(tf.abs(self.yhat_convert - tf.squeeze(self.y_data_convert)))
            self.ave_rel_error = tf.reduce_mean(tf.divide(tf.abs(self.yhat_convert - tf.squeeze(self.y_data_convert)),
                                                          tf.maximum(tf.squeeze(self.y_data_convert),
                                                                     tf.constant(0.01, dtype=tf.float32))))

            # output
            self.RSS = tf.reduce_sum(tf.square(self.yhat_convert - tf.squeeze(self.y_data_convert)))
            self.MS_train = self.RSS / traceR1
            self.MS_common = self.RSS / n_size_f
            self.A_R2 = utils.coeff_r2_adjusted_tensor_cal(tf.squeeze(y_data), self.yhat,
                                                           n_size_f - tf.cast(x_data_shape[1], tf.float32),
                                                           n_size_f - tf.constant(1.0))
            self.RMSE = tf.sqrt(self.MS_common)
            self.r_pearson = tf.sqrt(self.r2_pearson)


class DIAGNOSIS_SLM():
    def __init__(self, x_data, y_data, yhat, linear_beta_mat, gtweight=None , miny=None, maxy=None, y_train = None):
        self.x_data = x_data
        self.y_data = y_data
        self.linear_beta_mat = linear_beta_mat
        self.gtweight = gtweight
        self.y_train = tf.constant(y_train, dtype=tf.float32)

        with tf.variable_scope('DIAGNOSIS'):
            self.yhat = yhat
            self.yhat_rg = self.yhat_rg = tf.matmul(self.gtweight,self.y_train)
            # self.yhat_rg = tf.diag_part(tf.matmul(tf.matmul(x_data, linear_beta_mat), tf.transpose(gtweight)))

            # 正则化项
            # regularizer = layers.l1_l2_regularizer(0.0, 0.0005)
            regularizer = layers.l1_l2_regularizer(0.0, 0.0)
            weight_list = [weight for weight in tf.global_variables() if weight.name.find('_W') != -1]
            loss_reg = layers.apply_regularization(regularizer, weight_list)

            # 最小化方差
            # self.loss_rg = tf.reduce_mean(tf.square(self.yhat_rg - tf.squeeze(y_data)))
            self.loss_rg = tf.reduce_mean(tf.square(self.yhat - self.y_data))
            self.loss_rg_add_reg = self.loss_rg + loss_reg

            self.y_data_convert = y_data * tf.constant((maxy - miny), dtype=tf.float32) + tf.constant(miny, dtype=tf.float32)
            self.yhat_rg_convert = self.yhat_rg * tf.constant((maxy - miny), dtype=tf.float32) + tf.constant(miny, dtype=tf.float32)
            self.loss_rg_convert = tf.reduce_mean(tf.square(self.yhat_rg_convert - tf.squeeze(self.y_data_convert)))

            # self.loss = tf.reduce_mean(tf.square(self.yhat - tf.squeeze(y_data)))
            self.loss = tf.reduce_mean(tf.square(self.yhat - self.y_data))
            self.loss_add_reg = self.loss + loss_reg
            self.yhat_convert = self.yhat * tf.constant((maxy - miny), dtype=tf.float32) + tf.constant(miny, dtype=tf.float32)
            self.loss_convert = tf.reduce_mean(tf.square(self.yhat_convert - tf.squeeze(self.y_data_convert)))

            # R2
            # self.r2_pearson = utils.pearson_r2_tensor_cal(tf.squeeze(y_data), self.yhat)
            self.r2_pearson = utils.pearson_r2_tensor_cal(self.y_data, self.yhat)
            # self.r2_coeff = utils.coeff_r2_tensor_cal(tf.squeeze(y_data), self.yhat)
            self.r2_coeff = utils.coeff_r2_tensor_cal(self.y_data, self.yhat)
            # 调整后的R2


            # 平均绝对误差&平均相对误差
            self.ave_abs_error = tf.reduce_mean(tf.abs(self.yhat_convert - tf.squeeze(self.y_data_convert)))
            self.ave_rel_error = tf.reduce_mean(tf.divide(tf.abs(self.yhat_convert - tf.squeeze(self.y_data_convert)),
                                                          tf.maximum(tf.squeeze(self.y_data_convert),
                                                                     tf.constant(0.01, dtype=tf.float32))))
