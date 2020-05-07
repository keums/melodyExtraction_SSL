
# -*- coding: utf-8 -*-

import os
import sys
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from model import *
from featureExtraction import *
import glob


class Options(object):
    def __init__(self):
        self.num_spec = 513
        self.input_size = 31  # 115
        self.batch_size = 64  # 64
        self.note_res = 8
        self.figureON = False


options = Options()


def main_CS(file_name):

    pitch_range = np.arange(
        45, 95 + 1.0/options.note_res, 1.0/options.note_res)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    '''  Features extraction'''
    X_test, X_spec = spec_extraction(
        file_name=file_name, win_size=options.input_size)

    '''  melody predict'''
    model = melody_ResNet(options)
    model.load_weights('./weights/ResNet_CS.hdf5')
    y_predict = model.predict(X_test, batch_size=options.batch_size, verbose=1)

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

    est_pitch = medfilt(est_pitch, 5)

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
    
    ''' Plot '''
    if options.figureON == True:
        start = 2000
        end = 7000
        fig = plt.figure()
        plt.imshow(X_spec[:, start:end], origin='lower')
        plt.plot(est_pitch[start:end], 'r', linewidth=0.5)
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    options = Options()
    file_name = sys.argv[1]
    main_CS(file_name=file_name)

# def JDC():
#     AudioPATH = './'  # ex) AudioPATH = './dataset/*.mp3'
#     filePath = glob.glob(AudioPATH)
#     for fileName in filePath:
#         string = "python melodyExtraction_JDC.py "
#         string += fileName
#         os.system(string)

