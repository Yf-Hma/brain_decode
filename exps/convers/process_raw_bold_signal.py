import pandas as pd
import numpy as np
from glob import glob
import os, sys
import argparse
from nilearn.image import load_img
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
main = os.path.dirname(parent)
sys.path.append(main)

import src.configs.convers.configs as configs

#============================================================
def normalize_vect (x):
    max = np.max (x)
    min = np.min (x)

    if min < max:
        for i in range (len (x)):
            x[i]= (x[i] - min) / (max - min)


#============================================================
def nearestPoint(vect, value):
    dist = abs(value - vect[0])
    pos = 0

    for i in range(1, len(vect)):
        if abs(value - vect[i]) < dist and value <= vect[i]:
            dist = abs(value - vect[i])
            pos = i
    return pos


#======================================================
def bold_image_to_rois (file, atlas):
    fmri_img = load_img(file)
    masked_data = masker.fit_transform(fmri_img)
    print (masked_data.shape)
    return masked_data


#======================================================

def add_duration(df):
    duration = []
    for i in range(df.shape[0] - 1):
        duration.append ([df.iloc[i, 3] / 1000.0, df.iloc[i + 1, 3] / 1000.0, df.iloc[i + 1, 3] / 1000.0 - df.iloc[i, 3] / 1000.0])
    df = pd.concat([df, pd.DataFrame(duration, columns=['begin', 'end', 'Interval'])], axis=1)

    return df

#======================================================

def conversation_to_dataframe (data_path, folder_name, data, colnames, index, begin, end, type_conv, num_conv, concat):

    index_normalized = index[begin:end]
    start_pt = index_normalized [0]

    for j in range(0, end - begin):
        index_normalized[j] -= start_pt

    convers_data = pd.DataFrame ()
    convers_data ["Time (s)"] = index_normalized

    convers_data_discr = pd.DataFrame ()
    convers_data_discr ["Time (s)"] = index_normalized

    convers_data. loc [:, colnames] = data [begin : end, :]

    filename = data_path + "/" + folder_name + "/" + subject + "/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".csv"

    if os.path.exists (filename) and concat:
        existed_data = pd. read_pickle (filename). iloc[:,1:]
        convers_data = pd. concat ([convers_data, existed_data], axis = 1)

    # convers_data.to_pickle (filename)
    convers_data.to_csv(filename, index=False)

#======================================================

if __name__ == '__main__':

    parser = argparse. ArgumentParser ()
    parser.add_argument("--n_rois", type = int, default = 200, choices=[100,200,400])
    args = parser.parse_args()

    n_rois = args.n_rois
    data_path = configs.RAW_FMRI_DATA_PATH

    # Output directory of the preprocessed data from configs file
    args.output_data_path = configs.PROCESSED_FMRI_DATA_PATH 
    out_data_path = args.output_data_path


    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, verbose=0)

    atlas_img = load_img(atlas['maps'])

    masker = NiftiLabelsMasker(
        atlas.maps,
        labels=atlas.labels,
        standardize = True,
        resampling_target = "labels",
    )

    #region_names = ["atlas_%d"%(i+1) for i in range (0, len (atlas.labels))]
    region_names = ["atlas_%d"%(i+1) for i in range (0, n_rois)]
    colnames = region_names



    folder_name = "fMRI_data_%d"%n_rois
    if not os.path.exists ("%s/fMRI_data_%d"%(data_path, n_rois)):
        os.makedirs("%s/fMRI_data_%d"%(data_path, n_rois))

    index = [0]
    for i in range (1, 4 * 385):
        index. append (1.205 + index [i - 1])

    # loop over subjects
    for s in range (0,25):
        subject = "sub-%02d"%(s + 1)
        print (subject)

        if not os.path.exists ("%s/%s/%s"%(out_data_path,folder_name, subject)):
            os.makedirs("%s/%s/%s"%(out_data_path,folder_name, subject))

        bold_signal = pd. DataFrame ()

        files = []
        for i in range (4):
            file = "%s/%s/func/%s_task-convers_run-%02d_bold.nii.gz"%(data_path, subject, subject, (i+1))

            bold_one_block = bold_image_to_rois (file, atlas)
            if i == 0:
                bold_signal = bold_one_block
            else:
                bold_signal = np. concatenate ((bold_signal, bold_one_block), axis = 0)

        normalize_vect (bold_signal)

        if bold_signal.shape[0] == 0:
            continue

        testBlocks = ["TestBlocks" + str (i + 1) for i in range (4)]

        #--------------------------------------------------------#
        indice_block = 0
        for testBlock in testBlocks:
            logfile = glob (data_path + "/sourcedata/" + subject + "/" + subject + "_task-convers-" + testBlock + "*.txt")
            if len (logfile) == 0:
                print ("Some logfiles do not exist for subject %s"%subject)
                continue
            else:
                logfile = logfile [0]

            df = pd.read_csv (logfile, sep='\t', header=0)
            df = df [['condition', 'image', 'duration', 'ONSETS_MS']]
            df = add_duration (df)

            hh_convers = df [df.condition.str.contains("CONV1")] [['condition', 'begin', 'end']]
            hr_convers = df [df.condition.str.contains("CONV2")] [['condition', 'begin', 'end']]

            nb_hh_convers = hh_convers. shape [0]
            nb_hr_convers = hr_convers. shape [0]

            hh = 1
            hr = 2

            for i in range(nb_hh_convers):
                begin = nearestPoint (index, hh_convers.values[i][1]) + (385 * indice_block)
                end = nearestPoint (index, hh_convers.values[i][2]) + (385 * indice_block)  + 3 # add two observatiosn after the end of the conversation
                conversation_to_dataframe (out_data_path, folder_name, bold_signal, colnames, index, begin, end, "CONV1", hh,  False)
                hh += 2

            for i in range(nb_hr_convers):
                begin = nearestPoint (index, hr_convers.values[i][1]) + (385 * indice_block)
                end = nearestPoint (index, hr_convers.values[i][2]) + (385 * indice_block) + 3 # add two observatiosn after the end of the conversation
                conversation_to_dataframe (out_data_path, folder_name, bold_signal, colnames, index, begin, end, "CONV2", hr, False)

                hr += 2
            indice_block += 1
