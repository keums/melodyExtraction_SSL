
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import matplotlib.pyplot as plt
import glob
import numpy as np

from model import *
from featureExtraction import *

def melodyExtraction_NS(file_name):
    note_res = 8
    pitch_range = np.arange(40, 95 + 1.0/note_res, 1.0/note_res)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    '''  Features extraction'''
    X_test, X_spec = spec_extraction(
        file_name=file_name, win_size=31)

    '''  melody predict'''
    model = melody_ResNet()
    model.load_weights('./weights/ResNet_NS.hdf5')
    y_predict = model.predict(X_test, batch_size=64, verbose=1)

    y_shape = y_predict.shape
    num_total_frame = y_shape[0]*y_shape[1]
    est_pitch = np.zeros(num_total_frame)
    index_predict = np.zeros(num_total_frame)

    y_predict = np.reshape(y_predict, (num_total_frame, y_shape[2]))

    for i in range(num_total_frame):
        index_predict[i] = np.argmax(y_predict[i, :])
        pitch_MIDI = pitch_range[np.int32(index_predict[i])]
        if pitch_MIDI >= 45 and pitch_MIDI <= 95:
            est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440

    ''' save results '''
    PATH_est_pitch = './results/pitch_'+file_name.split('/')[-1]+'.txt'
    if not os.path.exists(os.path.dirname(PATH_est_pitch)):
        os.makedirs(os.path.dirname(PATH_est_pitch))
    f = open(PATH_est_pitch, 'w')
    for j in range(len(est_pitch)):
        est = "%.2f %.4f\n" % (0.01 * j, est_pitch[j])
        f.write(est)
    f.close()

    est_arr = np.loadtxt(PATH_est_pitch)
    return est_arr


if __name__ == '__main__':
    file_name = sys.argv[1]
    melodyExtraction_NS(file_name=file_name)

# def melodyExtraction_NS():
#     AudioPATH = './'  # ex) AudioPATH = './dataset/*.mp3'
#     filePath = glob.glob(AudioPATH)
#     for fileName in filePath:
#         string = "python melodyExtraction_JDC.py "
#         string += fileName
#         os.system(string)
