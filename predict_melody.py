# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import glob

from model import *
from featureExtraction import *
from load_testDataSet import *

import csv
import pandas as pd


class Options(object):
    def __init__(self):
        self.num_spec = 513
        self.window_size = 31  # 115
        self.batch_size = 64  # 64
        self.note_res = 8
        self.figureON = False


options = Options()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def SaveFeatures(filepath, output_dir, gpu_index):

    pitch_range = np.arange(
        40, 95 + 1.0/options.note_res, 1.0/options.note_res)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    '''  Features extraction'''
    X_test, X_spec = spec_extraction(
        file_name=filepath, win_size=options.window_size)
    ''' save results '''

    PATH_spec = '/home/keums/project/dataset/YouTube/'
    if not os.path.exists(os.path.dirname(PATH_spec)):
        os.makedirs(os.path.dirname(PATH_spec))
    fileName = filepath.split('/')[-1].split('.')[-2]

    np.save(PATH_spec + fileName + '.npy', X_spec)
    return


def predict_melody(model_ME, filepath, dataset, output_dir, gpu_index):

    pitch_range = np.arange(
        40, 95 + 1.0/options.note_res, 1.0/options.note_res)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    '''  Features extraction'''
    X_test, X_spec = spec_extraction(
        file_name=filepath, win_size=options.window_size)
    # X_test = load_feature_label(file_name=filepath, dataset=dataset, pitch_res=8)

    '''  melody predict'''
    y_predict = model_ME.predict(
        X_test, batch_size=options.batch_size, verbose=1)

    # if model = JDC
    y_predict = y_predict[0]

    # if model = resNet
    y_shape = y_predict.shape
    num_total = y_shape[0] * y_shape[1]
    y_predict = np.reshape(y_predict, (num_total, y_shape[2]))  # origin

    savePitch = True  # False
    if savePitch == True:
        est_pitch = np.zeros(num_total)
        for i in range(num_total):
            index_predict = np.argmax(y_predict[i, :])
            pitch_MIDI = pitch_range[np.int32(index_predict)]
            if pitch_MIDI >= 40 and pitch_MIDI <= 95:
                est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440

        est_pitch = medfilt(est_pitch, 5)

        ''' save results '''
        PATH_est_pitch = output_dir+'/pitch/' + \
            filepath.split('/')[-1].split('.')[-2]+'.txt'

        print(PATH_est_pitch)

        if not os.path.exists(os.path.dirname(PATH_est_pitch)):
            os.makedirs(os.path.dirname(PATH_est_pitch))
        f = open(PATH_est_pitch, 'w')
        for j in range(len(est_pitch)):
            est = "%.2f %.4f\n" % (0.01 * j, est_pitch[j])
            f.write(est)
        f.close()

    savePrediction = False  # True False
    if savePrediction == True:
        PATH_est_prediction = output_dir+'/predict2/' + \
            filepath.split('/')[-1].split('.')[-2]+'.npy'
        if not os.path.exists(os.path.dirname(PATH_est_prediction)):
            os.makedirs(os.path.dirname(PATH_est_prediction))

        np.save(PATH_est_prediction, y_predict)

    ''' Plot '''
    if options.figureON == True:
        start = 2000
        end = 7000
        fig = plt.figure()
        plt.imshow(X_spec[:, start:end], origin='lower')
        plt.plot(est_pitch[start:end], 'r', linewidth=0.5)
        fig.tight_layout()
        plt.show()

    # return est_pitch, y_predict
    return y_predict


def load_spec(file_name, win_size=31):
    x_test = []
    path_feature = '/home/keums/project/dataset/YouTube/original/features/'
    # path_feature = '/home/keums/project/dataset/YouTube/randAugment/features/'
    # x_spec = np.load(path_feature+file_name+'npy')
    x_spec = np.load(path_feature+file_name)
    num_frames = x_spec.shape[1]

    # for padding
    padNum = num_frames % win_size
    if padNum != 0:
        len_pad = win_size - padNum
        padding_feature = np.zeros(shape=(513, len_pad))
        x_spec = np.concatenate((x_spec, padding_feature), axis=1)
        num_frames = num_frames + len_pad

    for j in range(0, num_frames, win_size):
        x_test_tmp = x_spec[:, range(j, j + win_size)].T
        x_test.append(x_test_tmp)
    x_test = np.array(x_test)

    # for normalization

    x_train_mean = np.load('./x_data_mean_total_31.npy')
    x_train_std = np.load('./x_data_std_total_31.npy')
    x_test = (x_test-x_train_mean)/(x_train_std+0.0001)
    x_test = x_test[:, :, :, np.newaxis]
    return x_test


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('-w', '--weight',
                   help='Path to input audio (default: %(default)s',
                   type=str, default='./weights/55_ResNet_L(CE_G)_r8_singleGPU')
    p.add_argument('-p', '--filepath',
                   help='Path to input audio (default: %(default)s',
                   type=str, default='./test_audio_file.mp4')
    p.add_argument('-gpu', '--gpu_index',
                   help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s',
                   type=int, default=0)
    p.add_argument('-o', '--output_dir',
                   help='Path to output folder (default: %(default)s',
                   #    type=str, default='/home/keums/project/melodyExtraction/training_melody/output/test')
                   type=str, default='/home/keums/project/dataset/YouTube/')
    return p.parse_args()


def predict_melody2(model_ME, filename, filepath, output_dir, gpu_index):

    # filename = filename.rstrip('wav')
    filename = filename.replace('wav', 'npy')
    pitch_range = np.arange(
        40, 95 + 1.0/options.note_res, 1.0/options.note_res)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    '''  Features extraction'''
    # X_test, X_spec = spec_extraction(file_name=filepath, win_size=options.input_size)
    X_test = load_spec(filename)

    # X_test = load_feature_label(file_name=filepath, dataset='ADC04', pitch_res=8)

    '''  melody predict'''
    y_predict = model_ME.predict(
        X_test, batch_size=options.batch_size, verbose=1)

    # if model = JDC
    # y_predict = y_predict[0]

    # if model = resNet
    y_shape = y_predict.shape
    num_total = y_shape[0] * y_shape[1]
    y_predict = np.reshape(y_predict, (num_total, y_shape[2]))  # origin

    ''' TF // savePitch '''
    savePitch = False  # False
    if savePitch == True:
        est_pitch = np.zeros(num_total)
        for i in range(num_total):
            index_predict = np.argmax(y_predict[i, :])
            pitch_MIDI = pitch_range[np.int32(index_predict)]
            if pitch_MIDI >= 40 and pitch_MIDI <= 95:
                est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440

        est_pitch = medfilt(est_pitch, 5)

        ''' save results '''
        PATH_est_pitch = output_dir+'/' + \
            filepath.split('/')[-1].split('.')[-2]+'.txt'
        print(PATH_est_pitch)

        if not os.path.exists(os.path.dirname(PATH_est_pitch)):
            os.makedirs(os.path.dirname(PATH_est_pitch))
        f = open(PATH_est_pitch, 'w')
        for j in range(len(est_pitch)):
            est = "%.2f %.4f\n" % (0.01 * j, est_pitch[j])
            f.write(est)
        f.close()

    ''' TF // savePrediction '''
    savePrediction = True  # True False
    # output_dir = '/home/keums/project/dataset/YouTube/randAugment/predict/'
    # output_dir = '/home/keums/project/dataset/YouTube/original/smoothing/predict_B1/'
    output_dir = '/home/keums/project/dataset/YouTube/original/predict/CS3/'

    if savePrediction == True:
        PATH_est_prediction = output_dir + filename
        if not os.path.exists(os.path.dirname(PATH_est_prediction)):
            os.makedirs(os.path.dirname(PATH_est_prediction))

        np.save(PATH_est_prediction, y_predict)

    return y_predict


if __name__ == '__main__':
    options = Options()
    args = parser()
    path_weight = '/home/keums/project/melodyExtraction/new_train/weights/'
    # args.weight = path_weight+'ResNet_L(CE_G)_B.hdf5'
    # args.weight = path_weight+'ResNet_S_CT_2_singleGPU.hdf5'
    args.weight = path_weight+'ResNet_B1_singleGPU.hdf5'
    args.weight = path_weight + 'ResNet_S_CT_10/ResNet_S_CT_2_singleGPU.hdf5'
    args.weight = path_weight + \
        'ResNet_S_CT_10_2_step2/ResNet_S_CT_10_2_step2_v13_singleGPU.hdf5'
    args.weight = path_weight + \
        'ResNet_JDC_S_CT_10_it3_YG/ResNet_JDC_S_CT_10_it3_YG_v20_singleGPU.hdf5'
    args.weight = path_weight + \
        '/ResNet_CS_wYFm_f_iter2/ResNet_CS_wYFm_f_iter2_v14_singleGPU.hdf5'

    model_ME = melody_ResNet(options)
    # model_ME = melody_ResNet_JDC(options)
    # model_ME = melody_ResNet_T(options)
    model_ME.load_weights(args.weight)
    # predict_melody(model_ME, args.filepath,args.output_dir, args.gpu_index)

    wavpath = '/home/keums/project/downloadYoutube/'
    path = '/home/keums/project/melodyExtraction/training_melody/'

    print(args.weight)
    f = open(path + 'YouTube_Cover_C.txt')
    file_list = f.readlines()
    len_filelist = len(file_list)
    for fileName in file_list:
        fileName = fileName.rstrip('\n')+'.wav'
        print(fileName)
        args.filepath = wavpath+'download_C/'+fileName
        # SaveFeatures(args.filepath, args.output_dir, args.gpu_index)
        # predict_melody(model_ME, args.filepath,args.output_dir, args.gpu_index)
        predict_melody2(model_ME, fileName, args.filepath,
                        args.output_dir, args.gpu_index)
        # break

    f = open(path + 'YouTube_Cover_K.txt')
    file_list = f.readlines()
    len_filelist = len(file_list)
    for fileName in file_list:
        fileName = fileName.rstrip('\n')+'.wav'
        print(fileName)
        args.filepath = wavpath+'download_K/'+fileName
        # SaveFeatures(args.filepath, args.output_dir, args.gpu_index)
        # predict_melody(model_ME, args.filepath,args.output_dir, args.gpu_index)
        predict_melody2(model_ME, fileName, args.filepath,
                        args.output_dir, args.gpu_index)
        # break


# ----
    # pathSave = '/home/keums/project/melodyExtraction/training_melody/output/youtube/predict'
    # file_list = os.listdir(pathSave)
    # file_list_py = [file for file in file_list if file.endswith(".npy")]
    # # print("file_list_py: {}".format(file_list_py))

    # path = '/home/keums/project/downloadYoutube/'
    # csvRead = pd.read_csv(path+'songList_YT_Cover.csv')
    # # csvRead = pd.read_csv(path+'songList_YT_KAIST.csv')
    # lenList = csvRead.shape[0]
    # print(lenList)
    # songList = []
    # for idx in range(lenList):
    #     i =0
    #     youtubeURL = csvRead.loc[idx, 'youtubeLink']
    #     youtubeURL = youtubeURL.split('=')[-1]
    #     for songFolder in file_list_py:
    #         if youtubeURL == songFolder.rstrip('.npy'):
    #             i += 1
    #     if i ==0:
    #         print(youtubeURL)

# ----
    # csvRead = pd.read_csv(path+'songList_YT_Cover.csv')
    # # csvRead = pd.read_csv(path+'songList_YT_KAIST.csv')
    # lenList = csvRead.shape[0]
    # print(lenList)
    # songList = []
    # for idx in range(lenList):
    #     youtubeURL = csvRead.loc[idx, 'youtubeLink']
    #     youtubeURL = youtubeURL.split('=')[-1]
    #     print(youtubeURL)
    #     songList.append(youtubeURL)

    # print(len(songList))

    # with open('./YouTube_Cover_C.txt', 'w') as f:
    #     for item in songList:
    #         f.write("%s\n" % item)
