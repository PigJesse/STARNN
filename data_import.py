# 数据处理模块
import pandas as pd
import numpy as np
import pickle
import os
from pandas import datetime
import xlwt
import random
import sys


# 定义数据集类，用于train,validation,test管理
class DataSet(object):
    def __init__(self, x_data, y_data, space_dis, time_dis, s_dis_weight):
        assert x_data.shape[0] == y_data.shape[0] == space_dis.shape[0] == time_dis.shape[0], 'x,y,distance矩阵形状错误'
        self._x_data = x_data
        self._y_data = y_data
        if len(space_dis.shape) == 2:
            space_dis = np.reshape(space_dis, [space_dis.shape[0], space_dis.shape[1], 1])
        elif len(space_dis.shape) == 3:
            space_dis = np.reshape(space_dis, [space_dis.shape[0], space_dis.shape[1], space_dis.shape[2], 1])
        self._space_dis = space_dis
        if len(time_dis.shape) == 2:
            time_dis = np.reshape(time_dis, [time_dis.shape[0], time_dis.shape[1], 1])
        elif len(time_dis.shape) == 3:
            space_dis = np.reshape(time_dis, [time_dis.shape[0], time_dis.shape[1], time_dis.shape[2], 1])
        self._time_dis = time_dis
        self._num_examples = x_data.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._s_dis_weight = s_dis_weight

    @property
    def x_data(self):
        return self._x_data

    @property
    def y_data(self):
        return self._y_data

    @property
    def space_dis(self):
        return self._space_dis

    @property
    def time_dis(self):
        return self._time_dis

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def s_dis_weight(self):
        return self._s_dis_weight

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # 完成一轮做shuffle
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            shuffle = np.arange(self._num_examples)
            np.random.shuffle(shuffle)
            self._x_data = self._x_data[shuffle]
            self._y_data = self._y_data[shuffle]
            self._space_dis = self._space_dis[shuffle]
            self._time_dis = self._time_dis[shuffle]

            # 开始终止重新计数
            start = 0
            self._index_in_epoch = batch_size
            assert self._index_in_epoch <= self._num_examples
        end = self._index_in_epoch
        return self._x_data[start:end], self._y_data[start:end], self._space_dis[start:end], self._time_dis[start:end], self._s_dis_weight[start:end]


def init_dataset(data_path, train_ratio=0.75, validation_ratio=0.15, s_each_dir=True, t_cycle=True, log_y=False, date_str = "", create_force=False, random_fixed=True, seed=10):
    # data_path = 'E:\Study\SCI\GcTWR\Matlab\zjexport\HSZLJC_SZ_2016_5_new.csv'

    index_1 = data_path.rfind('/')
    index_2 = data_path.rfind('.')

    dataname = data_path[index_1 + 1:index_2]

    if s_each_dir:
        dataname = dataname + '_sdir'
    if t_cycle:
        dataname = dataname + '_tcyc'

    # 如果不是固定的话，都要重新建，以日期信息结尾
    if not random_fixed:
        seed = random.randrange(100)
        dataname = dataname + '_' + date_str + '_' + str(seed)
    else:
        dataname = dataname + '_' + str(seed)

    if not os.path.exists('Data/dataset/'):
        os.makedirs('Data/dataset/')

    file_save_path = 'Data/dataset/' + dataname + '.pckl'

    if create_force:
        if (os.path.isfile(file_save_path)):
            os.remove(file_save_path)

    if (os.path.isfile(file_save_path)):
        return open_dataset(file_save_path)
    else:
        dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d %H:%M')
        all_data = pd.read_csv(data_path, parse_dates=[3], engine='python', date_parser=dateparse)
        all_data.dropna(inplace=True)
        all_data = pd.DataFrame(all_data)

        col_data_x = ['DO', 'PH', 'COD', 'PO4', 'TN']
        col_data_y = ['CHLA']
        col_date = ['Monitor_Date']
        col_coordx = ['ProjX']
        col_coordy = ['ProjY']
        # 自变量
        data_varxs_all = all_data[col_data_x]
        data_varxs_all_norm = (data_varxs_all - data_varxs_all.min()) / (data_varxs_all.max() - data_varxs_all.min())
        data_varxs_all_norm['Constant'] = pd.Series(np.ones(len(all_data)), data_varxs_all_norm.index)
        data_varxs_all_norm = data_varxs_all_norm.values
        # 因变量
        data_vary_all = all_data[col_data_y]
        if log_y:
            data_vary_all_values = np.log2(data_vary_all.values + 1)
        else:
            data_vary_all_values = data_vary_all.values
        miny = np.min(data_vary_all_values)
        maxy = np.max(data_vary_all_values)
        data_vary_all_norm = (data_vary_all_values - miny) / (maxy - miny)

        # 时间、空间坐标列
        data_date_all = all_data[col_date]
        data_coordx_all = all_data[col_coordx]
        data_coordy_all = all_data[col_coordy]

        t_dis_extent = time_distance_extent_cal(data_date_all, cycle=t_cycle)
        s_dis_extent = space_distance_extent_cal(data_coordx_all, data_coordy_all, each_dir=s_each_dir)

        # 将数据分为训练集，验证集，测试集
        all_size = len(all_data)
        shuffle = np.arange(all_size)
        np.random.seed(seed)
        np.random.shuffle(shuffle)

        train_size = int(all_size * train_ratio)
        validation_size = int(all_size * validation_ratio)
        test_size = all_size - train_size - validation_size

        train_index = shuffle[0:train_size]
        validation_index = shuffle[train_size:train_size + validation_size]
        test_index = shuffle[train_size + validation_size:]

        data_date_train = data_date_all.iloc[train_index]
        data_date_validation = data_date_all.iloc[validation_index]
        data_date_test = data_date_all.iloc[test_index]

        data_coordx_train = data_coordx_all.iloc[train_index]
        data_coordx_validation = data_coordx_all.iloc[validation_index]
        data_coordx_test = data_coordx_all.iloc[test_index]

        data_coordy_train = data_coordy_all.iloc[train_index]
        data_coordy_validation = data_coordy_all.iloc[validation_index]
        data_coordy_test = data_coordy_all.iloc[test_index]

        t_dis_frame_train = time_distance_cal(data_date_train, data_date_train, cycle=t_cycle)
        t_dis_frame_validation = time_distance_cal(data_date_validation, data_date_train, cycle=t_cycle)
        t_dis_frame_test = time_distance_cal(data_date_test, data_date_train, cycle=t_cycle)

        # 空间距离
        s_dis_frame_train = space_distance_cal(data_coordx_train, data_coordy_train, data_coordx_train,
                                               data_coordy_train, each_dir=s_each_dir)
        s_dis_frame_validation = space_distance_cal(data_coordx_validation, data_coordy_validation, data_coordx_train,
                                                    data_coordy_train, each_dir=s_each_dir)
        s_dis_frame_test = space_distance_cal(data_coordx_test, data_coordy_test, data_coordx_train, data_coordy_train,
                                              each_dir=s_each_dir)

        s_dis_frame_train_thres = pd.DataFrame.copy(s_dis_frame_train)
        s_dis_frame_validation_thres = pd.DataFrame.copy(s_dis_frame_validation)
        s_dis_frame_test_thres = pd.DataFrame.copy(s_dis_frame_test)

        # 根据阈值到0-1
        # s_dis_frame_train_thres = transWeightByDis(np.transpose(s_dis_frame_train_thres), 32459.1)
        # s_dis_frame_validation_thres  = transWeightByDis(np.transpose(s_dis_frame_validation_thres), 32459.1)
        # s_dis_frame_test_thres  = transWeightByDis(np.transpose(s_dis_frame_test_thres), 32459.1)

        #距离反权重化
        s_dis_frame_train_thres = space_inverse_distance_cal(np.transpose(s_dis_frame_train_thres))
        s_dis_frame_validation_thres = space_inverse_distance_cal(np.transpose(s_dis_frame_validation_thres))
        s_dis_frame_test_thres = space_inverse_distance_cal(np.transpose(s_dis_frame_test_thres))

        # 行标准化
        s_dis_frame_train_thres = rowNorm(s_dis_frame_train_thres)
        s_dis_frame_validation_thres = rowNorm(s_dis_frame_validation_thres)
        s_dis_frame_test_thres = rowNorm(s_dis_frame_test_thres)

        if not s_each_dir:
            s_dis_train = np.transpose(
                ((s_dis_frame_train - s_dis_extent[0]) / (s_dis_extent[1] - s_dis_extent[0])).values)
            s_dis_val = np.transpose(((s_dis_frame_validation - s_dis_extent[0]) / (
                s_dis_extent[1] - s_dis_extent[0])).values)
            s_dis_test = np.transpose(
                ((s_dis_frame_test - s_dis_extent[0]) / (s_dis_extent[1] - s_dis_extent[0])).values)
            print(s_dis_train.shape)
            s_dis_frame_train_norm = s_dis_train
            s_dis_frame_validation_norm = s_dis_val
            s_dis_frame_test_norm = s_dis_test
        else:
            for i in range(len(s_dis_frame_train)):
                s_dis_train = np.transpose(
                    ((s_dis_frame_train[i] - s_dis_extent[i][0]) / (s_dis_extent[i][1] - s_dis_extent[i][0])).values)
                s_dis_val = np.transpose(((s_dis_frame_validation[i] - s_dis_extent[i][0]) / (
                    s_dis_extent[i][1] - s_dis_extent[i][0])).values)
                s_dis_test = np.transpose(
                    ((s_dis_frame_test[i] - s_dis_extent[i][0]) / (s_dis_extent[i][1] - s_dis_extent[i][0])).values)

                if i == 0:
                    s_dis_frame_train_norm = s_dis_train
                    s_dis_frame_validation_norm = s_dis_val
                    s_dis_frame_test_norm = s_dis_test
                else:
                    s_dis_frame_train_norm = np.dstack((s_dis_frame_train_norm, s_dis_train))
                    s_dis_frame_validation_norm = np.dstack((s_dis_frame_validation_norm, s_dis_val))
                    s_dis_frame_test_norm = np.dstack((s_dis_frame_test_norm, s_dis_test))

        if not t_cycle:
            t_dis_train = np.transpose(
                ((t_dis_frame_train - t_dis_extent[0]) / (t_dis_extent[1] - t_dis_extent[0])).values)
            t_dis_val = np.transpose(((t_dis_frame_validation - t_dis_extent[0]) / (
                t_dis_extent[1] - t_dis_extent[0])).values)
            t_dis_test = np.transpose(
                ((t_dis_frame_test - t_dis_extent[0]) / (t_dis_extent[1] - t_dis_extent[0])).values)

            t_dis_frame_train_norm = t_dis_train
            t_dis_frame_validation_norm = t_dis_val
            t_dis_frame_test_norm = t_dis_test
        else:
            for i in range(len(t_dis_frame_train)):
                t_dis_train = np.transpose(
                    ((t_dis_frame_train[i] - t_dis_extent[i][0]) / (t_dis_extent[i][1] - t_dis_extent[i][0])).values)
                t_dis_val = np.transpose(((t_dis_frame_validation[i] - t_dis_extent[i][0]) / (
                    t_dis_extent[i][1] - t_dis_extent[i][0])).values)
                t_dis_test = np.transpose(
                    ((t_dis_frame_test[i] - t_dis_extent[i][0]) / (t_dis_extent[i][1] - t_dis_extent[i][0])).values)

                if i == 0:
                    t_dis_frame_train_norm = t_dis_train
                    t_dis_frame_validation_norm = t_dis_val
                    t_dis_frame_test_norm = t_dis_test
                else:
                    t_dis_frame_train_norm = np.dstack((t_dis_frame_train_norm, t_dis_train))
                    t_dis_frame_validation_norm = np.dstack((t_dis_frame_validation_norm, t_dis_val))
                    t_dis_frame_test_norm = np.dstack((t_dis_frame_test_norm, t_dis_test))

        train_dataset = DataSet(data_varxs_all_norm[train_index], data_vary_all_norm[train_index],
                                s_dis_frame_train_norm, t_dis_frame_train_norm, s_dis_frame_train_thres)

        validation_dataset = DataSet(data_varxs_all_norm[validation_index],
                                     data_vary_all_norm[validation_index],
                                     s_dis_frame_validation_norm,
                                     t_dis_frame_validation_norm, s_dis_frame_validation_thres)

        test_dataset = DataSet(data_varxs_all_norm[test_index],
                               data_vary_all_norm[test_index],
                               s_dis_frame_test_norm,
                               t_dis_frame_test_norm, s_dis_frame_test_thres)

        save_dataset(file_save_path, [train_dataset, validation_dataset, test_dataset, miny, maxy, dataname])

        file_save_path = 'Data/dataset/' + dataname + '.xls'
        # 将数据保存到excel中
        if os.path.exists(file_save_path):
            os.remove(file_save_path)
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('data',cell_overwrite_ok=True)

        workbook = add_data_excel(workbook, name='train', data_coordx=np.array(data_coordx_train), data_coordy=np.array(data_coordy_train),
                                  data_date=np.array(data_date_train), data_varxs_all=np.array(data_varxs_all)[train_index],
                                  data_varxs_all_norm=data_varxs_all_norm[train_index],
                                  data_vary_all=np.array(data_vary_all)[train_index],
                                  data_vary_all_norm=data_vary_all_norm[train_index], all_data=np.array(all_data)[train_index], index=0)
        workbook = add_data_excel(workbook, name='validation', data_coordx=np.array(data_coordx_validation), data_coordy=np.array(data_coordy_validation),
                                  data_date=np.array(data_date_validation), data_varxs_all=np.array(data_varxs_all)[validation_index],
                                  data_varxs_all_norm=data_varxs_all_norm[validation_index],
                                  data_vary_all=np.array(data_vary_all)[validation_index],
                                  data_vary_all_norm=data_vary_all_norm[validation_index], all_data=np.array(all_data)[validation_index], index=len(train_index))
        workbook = add_data_excel(workbook, name='test', data_coordx=np.array(data_coordx_test), data_coordy=np.array(data_coordy_test),
                                  data_date=np.array(data_date_test), data_varxs_all=np.array(data_varxs_all)[test_index],
                                  data_varxs_all_norm=data_varxs_all_norm[test_index],
                                  data_vary_all=np.array(data_vary_all)[test_index],
                                  data_vary_all_norm=data_vary_all_norm[test_index], all_data=np.array(all_data)[test_index], index=len(train_index) + len(validation_index))

        workbook.add_sheet('result',cell_overwrite_ok=True)


        workbook.save(file_save_path)

        print(train_dataset.space_dis)
        return train_dataset, validation_dataset, test_dataset, miny, maxy, dataname

#时空数据、自变量因变量保存到excel中去
def add_data_excel(workbook,name,data_coordx,data_coordy,data_date,data_varxs_all,data_varxs_all_norm,data_vary_all,data_vary_all_norm,all_data,index):
    worksheet = workbook.get_sheet('data')

    worksheet.write(0, 0, 'dataset')
    worksheet.write(0, 1, 'UNIQUESTATION')
    worksheet.write(0, 2, 'Lon')
    worksheet.write(0, 3, 'Lat')
    worksheet.write(0, 4, 'Monitor_Date')
    worksheet.write(0, 5, 'DO')
    worksheet.write(0, 6, 'PH')
    worksheet.write(0, 7, 'COD')
    worksheet.write(0, 8, 'PO4')
    worksheet.write(0, 9, 'TN')
    worksheet.write(0, 10, 'CHLA')
    worksheet.write(0, 11, 'ProjX')
    worksheet.write(0, 12, 'ProjY')
    worksheet.write(0, 13, 'Time')

    # worksheet.write(0,0,'ProjX')
    # worksheet.write(0,1,'ProjY')
    # worksheet.write(0,2,'date')
    # worksheet.write(0,3,'DO')
    worksheet.write(0,14,'DO_norm')
    # worksheet.write(0,5,'PH')
    worksheet.write(0,15,'PH_norm')
    # worksheet.write(0,7,'COD')
    worksheet.write(0,16,'COD_norm')
    # worksheet.write(0,9,'PO4')
    worksheet.write(0,17,'PO4_norm')
    # worksheet.write(0,11,'TN')
    worksheet.write(0,18,'TN_norm')
    # worksheet.write(0,13,'CHLA')
    worksheet.write(0,19,'CHLA_norm')

    start = index

    for i in range(data_coordx.shape[0]):
        worksheet.write(i + 1 + start, 0, name)
        worksheet.write(i + 1 + start, 1, all_data[i, 0])
        worksheet.write(i + 1 + start, 2, all_data[i, 1])
        worksheet.write(i + 1 + start, 3, all_data[i, 2])
        worksheet.write(i + 1 + start, 4, pd.to_datetime(data_date[i, 0]).strftime('%Y-%m-%d'))
        worksheet.write(i + 1 + start, 5, all_data[i, 4])
        worksheet.write(i + 1 + start, 6, all_data[i, 5])
        worksheet.write(i + 1 + start, 7, all_data[i, 6])
        worksheet.write(i + 1 + start, 8, all_data[i, 7])
        worksheet.write(i + 1 + start, 9, all_data[i, 8])
        worksheet.write(i + 1 + start, 10, all_data[i, 9])
        worksheet.write(i + 1 + start, 11, all_data[i, 10])
        worksheet.write(i + 1 + start, 12, all_data[i, 11])
        worksheet.write(i + 1 + start, 13, all_data[i, 12])

        # worksheet.write(i + 1, 0, data_coordx[i, 0])
        # worksheet.write(i + 1, 1, data_coordy[i, 0])
        # worksheet.write(i + 1, 2, pd.to_datetime(data_date[i, 0]).strftime('%Y-%m-%d'))
        # worksheet.write(i + 1, 3, data_varxs_all[i,0])
        # worksheet.write(i + 1, 5, data_varxs_all[i,1])
        # worksheet.write(i + 1, 7, data_varxs_all[i,2])
        # worksheet.write(i + 1, 9, data_varxs_all[i,3])
        # worksheet.write(i + 1, 11, data_varxs_all[i,4])
        worksheet.write(i + 1 + start, 14, data_varxs_all_norm[i, 0])
        worksheet.write(i + 1 + start, 15, data_varxs_all_norm[i, 1])
        worksheet.write(i + 1 + start, 16, data_varxs_all_norm[i, 2])
        worksheet.write(i + 1 + start, 17, data_varxs_all_norm[i, 3])
        worksheet.write(i + 1 + start, 18, data_varxs_all_norm[i, 4])
        # worksheet.write(i + 1, 13, data_vary_all[i, 0])
        worksheet.write(i + 1 + start, 19, data_vary_all_norm[i, 0])

    return workbook

def time_distance_extent_cal(time_frame, cycle=False):
    if not cycle:
        t_dis_min = 1000000
        t_dis_max = 0
        for i in range(len(time_frame)):
            t_dis_temp = (np.abs((time_frame.values[i] - time_frame.values) / np.timedelta64(1, 'D'))).astype(int)
            t_dis_temp = np.reshape(t_dis_temp, len(t_dis_temp))
            if (np.min(t_dis_temp) < t_dis_min):
                t_dis_min = np.min(t_dis_temp)
            if (np.max(t_dis_temp) > t_dis_max):
                t_dis_max = np.max(t_dis_temp)

        return t_dis_min, t_dis_max
    else:
        t_dis_nc_min = 1000000
        t_dis_nc_max = 0
        t_dis_c_min = 1000000
        t_dis_c_max = 0

        years = time_frame.values.astype('datetime64[Y]').astype(int) + 1970
        # months = time_frame.values.astype('datetime64[M]').astype(int) % 12 + 1
        # days = (time_frame.values.astype('datetime64[D]') - time_frame.values.astype('datetime64[M]')).astype(int) + 1
        month_days = (time_frame.values.astype('datetime64[D]') - time_frame.values.astype('datetime64[Y]')).astype(
            int) + 1

        for i in range(len(time_frame)):
            t_dis_nc = np.abs(years[i] - years)
            t_dis_c = np.abs(month_days[i] - month_days)
            t_dis_nc = np.reshape(t_dis_nc, len(t_dis_nc))
            t_dis_c = np.reshape(t_dis_c, len(t_dis_c))

            if (np.min(t_dis_nc) < t_dis_nc_min):
                t_dis_nc_min = np.min(t_dis_nc)
            if (np.max(t_dis_nc) > t_dis_nc_max):
                t_dis_nc_max = np.max(t_dis_nc)

            if (np.min(t_dis_c) < t_dis_c_min):
                t_dis_c_min = np.min(t_dis_c)
            if (np.max(t_dis_c) > t_dis_c_max):
                t_dis_c_max = np.max(t_dis_c)

        return [t_dis_nc_min, t_dis_nc_max], [t_dis_c_min, t_dis_c_max]


def space_distance_extent_cal(coordx_frame, coordy_frame, each_dir=False):
    if not each_dir:
        s_dis_min = 1000000
        s_dis_max = 0
        for i in range(len(coordx_frame)):
            s_dis_x = coordx_frame.values[i] - coordx_frame.values
            s_dis_y = coordy_frame.values[i] - coordy_frame.values
            s_dis = np.sqrt(s_dis_x * s_dis_x + s_dis_y * s_dis_y)
            s_dis = np.reshape(s_dis, len(s_dis))
            if (np.min(s_dis) < s_dis_min):
                s_dis_min = np.min(s_dis)
            if (np.max(s_dis) > s_dis_max):
                s_dis_max = np.max(s_dis)

        return s_dis_min, s_dis_max
    else:
        s_dis_x_min = 1000000
        s_dis_x_max = 0
        s_dis_y_min = 1000000
        s_dis_y_max = 0

        for i in range(len(coordx_frame)):
            s_dis_x = np.abs(coordx_frame.values[i] - coordx_frame.values)
            s_dis_y = np.abs(coordy_frame.values[i] - coordy_frame.values)
            s_dis_x = np.reshape(s_dis_x, len(s_dis_x))
            s_dis_y = np.reshape(s_dis_y, len(s_dis_y))
            if (np.min(s_dis_x) < s_dis_x_min):
                s_dis_x_min = np.min(s_dis_x)
            if (np.max(s_dis_x) > s_dis_x_max):
                s_dis_x_max = np.max(s_dis_x)

            if (np.min(s_dis_y) < s_dis_y_min):
                s_dis_y_min = np.min(s_dis_y)
            if (np.max(s_dis_y) > s_dis_y_max):
                s_dis_y_max = np.max(s_dis_y)

        return [s_dis_x_min, s_dis_x_max], [s_dis_y_min, s_dis_y_max]


def time_distance_cal(time_frame, train_time_frame, cycle=False):
    if not cycle:
        t_dis_frame = pd.DataFrame()
        for i in range(len(time_frame)):
            t_dis_temp = (np.abs((time_frame.values[i] - train_time_frame.values) / np.timedelta64(1, 'D'))).astype(int)
            t_dis_temp = np.reshape(t_dis_temp, len(t_dis_temp))
            t_dis_frame[str(i + 1)] = pd.Series(t_dis_temp)

        # return t_dis_frame, t_dis_frame.max().max(), t_dis_frame.min().min()
        return t_dis_frame
    else:
        t_dis_nc_frame = pd.DataFrame()
        t_dis_c_frame = pd.DataFrame()

        years = time_frame.values.astype('datetime64[Y]').astype(int) + 1970
        month_days = (time_frame.values.astype('datetime64[D]') - time_frame.values.astype('datetime64[Y]')).astype(
            int) + 1

        train_years = train_time_frame.values.astype('datetime64[Y]').astype(int) + 1970
        train_month_days = (train_time_frame.values.astype('datetime64[D]') - train_time_frame.values.astype(
            'datetime64[Y]')).astype(
            int) + 1

        for i in range(len(time_frame)):
            t_dis_nc = np.abs(years[i] - train_years)
            t_dis_c = np.abs(month_days[i] - train_month_days)
            t_dis_nc = np.reshape(t_dis_nc, len(t_dis_nc))
            t_dis_c = np.reshape(t_dis_c, len(t_dis_c))

            t_dis_nc_frame[str(i + 1)] = pd.Series(t_dis_nc)
            t_dis_c_frame[str(i + 1)] = pd.Series(t_dis_c)

        return t_dis_nc_frame, t_dis_c_frame


def space_distance_cal(coordx_frame, coordy_frame, train_coordx_frame, train_coordy_frame, each_dir=False):
    if not each_dir:
        s_dis_frame = pd.DataFrame()
        for i in range(len(coordx_frame)):
            s_dis_x = coordx_frame.values[i] - train_coordx_frame.values
            s_dis_y = coordy_frame.values[i] - train_coordy_frame.values
            s_dis = np.sqrt(s_dis_x * s_dis_x + s_dis_y * s_dis_y)
            s_dis = np.reshape(s_dis, len(s_dis))
            s_dis_frame[str(i + 1)] = pd.Series(s_dis)

        return s_dis_frame
    else:
        s_dis_x_frame = pd.DataFrame()
        s_dis_y_frame = pd.DataFrame()
        for i in range(len(coordx_frame)):
            s_dis_x = np.abs(coordx_frame.values[i] - train_coordx_frame.values)
            s_dis_y = np.abs(coordy_frame.values[i] - train_coordy_frame.values)
            s_dis_x = np.reshape(s_dis_x, len(s_dis_x))
            s_dis_y = np.reshape(s_dis_y, len(s_dis_y))
            s_dis_x_frame[str(i + 1)] = pd.Series(s_dis_x)
            s_dis_y_frame[str(i + 1)] = pd.Series(s_dis_y)

        return s_dis_x_frame, s_dis_y_frame


def save_dataset(dataname, dataset_list):
    f = open(dataname, 'wb')
    pickle.dump(dataset_list, f)
    f.close()


def open_dataset(dataname):
    f = open(dataname, 'rb')
    train_dataset, validation_dataset, test_dataset, miny, maxy, dataname = pickle.load(f)
    f.close()
    return train_dataset, validation_dataset, test_dataset, miny, maxy, dataname

# 根据距离阈值判断等于0或者1
def transWeightByDis(dis_frame, threshold):
    dis_frame = dis_frame.values
    for i in range(dis_frame.shape[0]):
        for j in range(dis_frame.shape[1]):
            if dis_frame[i,j] <= threshold and dis_frame[i,j] > 0:
                dis_frame[i,j] = 1
            else:
                dis_frame[i, j] = 0
    return dis_frame

# 反权重距离(平方)
def space_inverse_distance_cal(dis_frame):
    dis_frame = dis_frame.values
    for i in range(dis_frame.shape[0]):
        for j in range(dis_frame.shape[1]):
            if dis_frame[i,j] == 0:
                dis_frame[i,j] = 0
            else:
                dis_frame[i, j] = 1/np.power(dis_frame[i, j],2)
    return dis_frame

# 行标准化
def rowNorm(dis_frame):
    for i in range(dis_frame.shape[0]):
        all = np.sum(dis_frame[i,:])
        # 判断一下没有邻居的情况（应该怎么处理呢？？）
        if all == 0:
            dis_frame[i,i] = 1
        else:
            dis_frame[i,:] = dis_frame[i,:] / all
    return dis_frame

if __name__ == '__main__':
    init_dataset('zjexport/HSZLJC_SZ_2016_5_new.csv', s_each_dir=False, t_cycle=False, create_force=True, random_fixed=False)
