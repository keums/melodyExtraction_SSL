import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import medfilt
from predict_melody import *
from getlist import *
# from evaluation import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluation_main(weight, sub_modelName, savePath, output_dir, dataset, model, gpu_index):
    if dataset == 'ADC04':
        dataset_path = '/home/keums/project/dataset/adc2004_full_set/file/'
        test_songlist = getlist_ADC2004()
    elif args.dataset == 'MIREX05':
        dataset_path = '/home/keums/project/dataset/mirex05TrainFiles/file/'
        test_songlist = getlist_MIREX05()
    elif 'MDB' in dataset:
        dataset_path = '/home/keums/project/dataset/MedleyDB/MDB_MIX/'
        _, _, test_songlist = getlist_mdb_vocal()
    # elif 'Mdb_melody2' in dataset:
    #     dataset_path = '/home/keums/project/dataset/MedleyDB/MDB_MIX/'
    #     _, _, test_songlist = getlist_mdb()
    else:
        print('Error: Wrong type of dataset, Must be ADC2004 or MIREX05 or Mdb_vocal or Mdb_melody2')

    with open(savePath, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(['Songname', 'VR', 'VFA', 'RPA', 'RCA', 'OA'])
        avg_arr = [0, 0, 0, 0, 0]

        for song in test_songlist:
            filepath = dataset_path+song+'.wav'

            if dataset == 'MDB':
                filepath = dataset_path+song+'_MIX.wav'
            else:
                filepath = dataset_path+song+'.wav'
            print(filepath)
            # filepath = filepath.replace('.wav','_MIX.wav')
            # _ = predict_melody(weight, filepath.replace('.wav', '_MIX.wav'), dataset, output_dir, gpu_index)
            # else:
            _ = predict_melody(weight, filepath, dataset,
                               output_dir, gpu_index)
            eval_arr = evaluation_one_song(dataset, filepath, model)
            avg_arr += eval_arr
            writer.writerow([song, eval_arr[0], eval_arr[1],
                             eval_arr[2], eval_arr[3], eval_arr[4]])
        avg_arr /= len(test_songlist)
        writer.writerow(['Avg', avg_arr[0], avg_arr[1],
                         avg_arr[2], avg_arr[3], avg_arr[4]])


def evaluation_one_song(dataset, filepath, model):
    dataset_dir = '/home/keums/project/dataset/'
    filename = filepath.split('/')[-1]
    songname = filename.split('.')[0]

    if 'MDB' in dataset:
        songname = songname.replace('_MIX', '')
        # ypath = '/home/keums/project/dataset/MedleyDB/pitch/Melody2/'+songname+'_MELODY2.csv'
        # ref_path = dataset_dir+'MedleyDB/Annotation/melody1/' + songname+'_MELODY1.csv'
        ref_path = dataset_dir+'MedleyDB/Annotation/melody2/' + songname+'_MELODY2.csv'
        ref_arr = csv2ref(ref_path)

    elif 'ADC04' in dataset:
        ref_path = '/home/keums/project/dataset/adc2004_full_set/pitch/'+songname+'REF.txt'
        ref_arr = np.loadtxt(ref_path)

    elif 'MIREX05' in dataset:
        ref_path = '/home/keums/project/dataset/mirex05TrainFiles/pitch/'+songname+'REF.txt'
        ref_arr = np.loadtxt(ref_path)

    result_path = './output/pitch/'
    if dataset == 'MDB':
        est_arr = np.loadtxt(result_path+songname+'_MIX.txt')
    else:
        est_arr = np.loadtxt(result_path+songname+'.txt')
    eval_arr = melody_eval(ref_arr, est_arr, dataset)

    return eval_arr


def parser_eval():
    p = argparse.ArgumentParser()
    p.add_argument('-w', '--weight',
                   help='Path to input audio (default: %(default)s',
                   #    type=str, default='./weights/ResNet_L(CE_G)_r8_3_MN1_singleGPU.hdf5')
                   type=str, default='./weights/55_ResNet_L(CE_G)_r8_TS2_singleGPU.hdf5')
    p.add_argument('-gpu', '--gpu_index',
                   help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s',
                   type=int, default=0)
    p.add_argument('-o', '--output_dir',
                   help='Path to output folder (default: %(default)s',
                   type=str, default='./output')
    p.add_argument('-m', '--model',
                   help='Assign a model for melody extraction (JDC,CREPE) (default: %(default)s',
                   type=str, default='resNet')
    p.add_argument('-d', '--dataset',
                   help='Assign a dataset (ADC04, MIREX05) (default: %(default)s',
                   type=str, default='ADC04')

    return p.parse_args()


if __name__ == '__main__':
    options = Options()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    weight_path = '/home/keums/project/melodyExtraction/new_train/weights/'

    args = parser_eval()

    args.model = 'ResNet_JDC'  # ResNet ResNet_JDC
    sub_modelName = 'CS'  # S NS SS_C S_CT
    # sub_modelName = 'S_CT'  # S NS SS_C S_CT
    # sub_modelName = 'B'  # S NS SS_C S_CT

    # args.weight = weight_path + '55_'+args.model+'_L(CE_G)_r8_'+sub_modelName+'_singleGPU.hdf5'
    # args.weight = weight_path + args.model+'_'+sub_modelName+'_2_v8_singleGPU.hdf5'
    # args.weight = weight_path + args.model+'_' + sub_modelName+'(0.5)_1_v9_singleGPU.hdf5'
    # args.weight = weight_path + args.model+'_' + sub_modelName+'2(0.5)_v19_singleGPU.hdf5'

    if args.model == 'ResNet':
        model_ME = melody_ResNet(options)
    elif args.model == 'ResNet_JDC':
        model_ME = melody_ResNet_JDC(options)

    # Temp = '10'
    for vers in [18, 19, 20, 21, 22]:

        # args.weight = './weights/ResNet_S_T_'+str(Temp)+'/ResNet_S_T_'+str(Temp)+'_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = './weights/ResNet_S_CT_' + str(Temp)+'_2_step2/ResNet_S_CT_'+str(Temp) + '_step2_v'+str(vers)+'_singleGPU.hdf5'

        # path_defalt = './weights/ResNet_CS'
        path_model = 'ResNet_CS'
        path_model = 'ResNet_JDC_CS'

        # path_defalt = './weights/ResNet_S_CT_' + str(Temp)
        # path_defalt2 = 'ResNet_S_CT_' + str(Temp)

        # path_defalt = './weights/ResNet_B_' + str(Temp)
        # path_defalt2 = 'ResNet_B_' + str(Temp)

        # path_defalt = './weights/ResNet_JDC' + str(Temp)
        # path_defalt2 = 'ResNet_JDC' + str(Temp)

        # path_defalt = './weights/ResNet_JDC_S_CT'
        # path_defalt2 = 'ResNet_JDC_S_CT'

        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_B1_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/training_melody/weights/ResNet_L(CE_G)_SC.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/training_melody/weights/ResNet_S1_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/training_melody/weights/ResNet_NS1_singleGPU.hdf5'

        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_S_CT_10_2_step2/ResNet_S_CT_10_2_step2_v12_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_JDC/ResNet_JDC_B_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_JDC/ResNet_JDC_S1_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_JDC/ResNet_JDC_NS_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_JDC/ResNet_JDC_S_C_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_JDC/ResNet_JDC_S_CT_10_newload_v14_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_JDC/ResNet_JDC_S_CT_10_it2_n_v9_singleGPU.hdf5'
        # args.weight = '/home/keums/project/melodyExtraction/new_train/weights/ResNet_JDC/ResNet_JDC_S_CT_12_it2_n_v14_singleGPU.hdf5'
        # args.weight = path_defalt + '_step3/' + path_defalt2 + '_step3_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_wJMD/' + path_defalt2 + '_wJMD_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_wJMD2/' + path_defalt2 + '_wJMD2_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_musdb/' + path_defalt2 + '_musdb_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_wDSD/' + path_defalt2 + '_wDSD_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_UL_only/'+ path_defalt2 + '_UL_only_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_UL_TR/' + path_defalt2 + '_UL_TR_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_TR2/' + path_defalt2 + '_TR2_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_gtzan/' + path_defalt2 + '_gtzan_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_gtzan2/' + path_defalt2 + '_gtzan2_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_it3/' + path_defalt2 + '_it3_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_it3_2/' + path_defalt2 + '_it3_2_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_it3_YG/' + path_defalt2 + '_it3_YG_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_it4_YG/' + path_defalt2 + '_it4_YG_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_wYJG_SV/' + path_defalt2 + '_wYJG_SV_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_wYJG/' + path_defalt2 + '_wYJG_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_wJG/' + path_defalt2 + '_wJG_v'+str(vers)+'_singleGPU.hdf5'
        # args.weight = path_defalt + '_wY/' + path_defalt2 + '_wY_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFs/' + \
            path_model + '_wYFs_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFs_f/' + \
            path_model + '_wYFs_f_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFm/' + \
            path_model + '_wYFm_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFm_v/' + \
            path_model + '_wYFm_v_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFm_f_iter2/' + \
            path_model + '_wYFm_f_iter2_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFm_f_iter3/' + \
            path_model + '_wYFm_f_iter3_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFm_fn_iter3/' + \
            path_model + '_wYFm_fn_iter3_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFm_f_iter3_sgd/' + \
            path_model + '_wYFm_f_iter3_sgd_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFl_f2/' + \
            path_model + '_wYFl_f2_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFl/' + \
            path_model + '_wYFl_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFl_f_iter2/' + \
            path_model + '_wYFl_f_iter2_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFl_f_iter3_4/' + \
            path_model + '_wYFl_f_iter3_4_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFl_f_iter3_b1/' + \
            path_model + '_wYFl_f_iter3_b1_v'+str(vers)+'_singleGPU.hdf5'
        args.weight = './weights/'+path_model + '_wYFl_f_iter3/' + \
            path_model + '_wYFl_f_iter3_v'+str(vers)+'_singleGPU.hdf5'

        # args.weight = './weights/ResNet_S_T_13_v9_singleGPU.hdf5'
        model_ME.load_weights(args.weight)

        for args.dataset in ['ADC04', 'MIREX05', 'MDB']:  # , 'ADC04', 'MIREX05','MDB'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_B/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_B_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_S/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_S_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_NS/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_NS_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_S_C/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_S_C_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_S_CT_n/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_S_CT_n_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_S_CT_it2_n/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_S_CT_it2_v'+str(vers)+'.csv'
            # savePath = '/home/keums/project/melodyExtraction/new_train/output/ResNet_S_CT_10_step2/' + str(args.dataset) + '_ResNet_S_CT_10_2_step2_v12_2.csv'
            # savePath = '/home/keums/project/melodyExtraction/new_train/output/ResNet_S_CT_10_step3/' + str(args.dataset) + '_ResNet_S_CT_10_2_step3_v18_2.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_wJMD2/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_wJMD2_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_wJMD/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp)+'_wJMD_v'+str(vers)+'.csv'
                # str(sub_modelName)+'_' + str(Temp)+'_2_step2_v'+str(vers)+'_2.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_musdb/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_musdb_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_wDSD/' + str(args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_wDSD_v'+str(vers)+'.csv'

            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_UL_only/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_UL_only_v'+str(vers)+'.csv'

            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_UL_TR/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_UL_TR_v'+str(vers)+'.csv'

            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_TR2/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_TR2_v'+str(vers)+'.csv'

            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_gtzan/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_gtzan_v'+str(vers)+'.csv'

            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_it3_2/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_it3_2_v'+str(vers)+'.csv'

            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_it3_YG/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_it3_YG_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_it4_YG/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_it4_YG_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_wYJG_SV/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_wYJG_SV_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_wYJG/' + str(
                # args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_wYJG_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_wJG/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_wJG_v'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+'_'+str(Temp) + '_wY/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+'_' + str(Temp) + '_wY'+str(vers)+'.csv'
            # savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName)+ '_wYFs/' + str(
            #     args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName)+ '_wYFs_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFs_f/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFs_f_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFm/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFm_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFm_v/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFm_v_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFm_f_iter2/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFm_f_iter2_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFm_f_iter3/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFm_f_iter3_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFm_fn_iter3/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFm_fn_iter3_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFm_f_iter3_sgd/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFl_f_iter3_sgd_'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFl_f2/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFl_f2_v'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFl/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFl_v'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFl_f_iter2/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFl_f_iter2_v'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFl_f_iter3_4/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFl_f_iter3_4_v'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFl_f_iter3_b1/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFl_f_iter3_b1_v'+str(vers)+'.csv'
            savePath = args.output_dir+'/'+str(args.model)+'_'+str(sub_modelName) + '_wYFl_f_iter3/' + str(
                args.dataset) + '_'+str(args.model)+'_' + str(sub_modelName) + '_wYFl_f_iter3_v'+str(vers)+'.csv'

            print(savePath)

            if not os.path.exists(os.path.dirname(savePath)):
                os.makedirs(os.path.dirname(savePath))

            # savePath = args.output_dir+'/ResNet_S_T_01/' + str(args.dataset) + '_'+str(args.model) + \
            #     '_result_'+str(sub_modelName)+'01_v'+str(vers)+'.csv'

            evaluation_main(model_ME, sub_modelName, savePath, args.output_dir,
                            args.dataset, args.model, args.gpu_index)
