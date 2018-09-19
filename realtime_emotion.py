from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import facenet
import detect_face
import os
from os.path import join as pjoin
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib

from fastai.fastai.imports import *
from fastai.fastai.plots import *
from fastai.fastai.io import get_data

from fastai.fastai.conv_learner import *

import pdb

import argparse

'''
    python realtime_emotion.py --dataset "s570_AU9.csv" --model "s570_AU9_check_overfitting_ps"
'''

parser = argparse.ArgumentParser(description='Detect expressions in real time', argument_default=argparse.SUPPRESS)
parser.add_argument('--dataset', default='s2000_au6_au12.csv', help='dataset name - in order to create the model')
parser.add_argument('--model', default='au6_12_s2000_au6_au12_check_overfitting_ps3', help='model name - in order to load the trained model')
args = parser.parse_args()

dataset_name = args.dataset
model_name = args.model

abpath = './'
det_path = f'{abpath}data/facerec'

print('Creating networks and loading parameters')

# default dataset is for detecting happiness - not happiness 's2000_au6_au12.csv' and the default model au6_12_s2000_au6_au12
PATH = 'data/emotionet/'
MC_CSV = f'{PATH}{dataset_name}'
JPEGS = 'imgs'

arch = resnet34
bs = 20
sz = 224

ps = [0.4, 0.8, 0.9]

aug_tfms = [RandomRotateZoom(20, 1.1, 0.15, ps=[0.4, 0.3, 0.1, 0.2]), RandomLighting(0.1, 0.1), RandomDihedral()]
tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms)
data = ImageClassifierData.from_csv(PATH, JPEGS, MC_CSV, bs=bs, tfms=tfms)

learn = ConvLearner.pretrained(arch, data, precompute=False, xtra_fc=[1024, 512], opt_fn=lambda *args, **kwargs: optim.Adam(*args, **kwargs), ps=ps)

learn.load(f'{model_name}')

#prediction
trn_tfms, val_tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms) # get transformations


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, det_path)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        video_capture = cv2.VideoCapture(0)
        c = 0

        # #video writer
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('3F_0726.avi', fourcc, fps=30, frameSize=(640,480))

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    # resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if c % timeF == 0:
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]

                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                # print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue
                        
                        #take a crop bigger with 50% of face size
                        face_perc = 0.5
                        sz1 = int(round((bb[i][3]-bb[i][1])*face_perc))
                        sz2 = int(round((bb[i][2]-bb[i][0])*face_perc))

                        newc1 = bb[i][1]-sz1 if (bb[i][1]-sz1)>0 else 0
                        newc3 = bb[i][3]+sz1 if (bb[i][3]+sz1<frame.shape[0]) else frame.shape[0]
                        newc0 = bb[i][0]-sz2 if (bb[i][0]-sz2)>0 else 0
                        newc2 = bb[i][2]+sz2 if (bb[i][2]+sz2<frame.shape[1]) else frame.shape[1]

                        # frame shape 240 320 3
                        cropped.append(frame[newc1:newc3, newc0:newc2, :])
                        z = len(cropped) - 1
                        cropped[z] = facenet.flip(cropped[z], False) # shape 110 84 3

                        scaled.append(misc.imresize(cropped[z], (image_size, image_size), interp='bilinear'))
                        scaled[z] = cv2.resize(scaled[z], (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                        scaled[z] = facenet.prewhiten(scaled[z])
                        scaled_reshape.append(scaled[z].reshape(-1, input_image_size, input_image_size, 3))
                        # image_scaled_reshaped.shape = 160 160 3
                        image_scaled_reshaped = scaled_reshape[z][0,:,:,:]

                        # emotion prediction
                        face_image = image_scaled_reshaped
                        im = val_tfms(face_image)
                        log_pred_path = learn.predict_array(im[None])

                        prob_path = np.exp(log_pred_path)
                        pred_path = np.argmax(prob_path, axis=1)

                        probab = round(prob_path[0][pred_path[0]],2)

                        classname = '\n'.join(data.classes[o] for o in [pred_path[0]])   
                        classname = classname + str(probab)

                        send_data = {'probab': float(probab), 'class': data.classes[pred_path[0]]}
                        print('send_data',send_data)

                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    # boxing face

                        # plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20

                        cv2.putText(frame, classname, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            strr = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, strr, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        # #video writer
        # out.release()
        cv2.destroyAllWindows()
