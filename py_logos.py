#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings

#gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')

from keras import backend as K
K.set_image_data_format('channels_last')
print(K.backend())
if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth=True
    session_conf.intra_op_parallelism_threads = 1  # For reproducibility
    session_conf.inter_op_parallelism_threads = 1  # For reproducibility
    sess = tf.Session(config=session_conf, graph=tf.get_default_graph())
    #sess = tf.Session(config=session_conf)
    K.set_session(sess)

#import util


import argparse
import random, math, time
import cv2
import traceback
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import utilLoadImages, utilModel

utilLoadImages.init()

#----------------------------------------------------------------
def plot_train_curves(hist):
    plt.style.use("ggplot")
    plt.figure()
    for k in hist.keys():
        plt.plot(np.arange(0, len(hist[k])), hist[k], label=k)
    plt.title("Training and validation results")
    plt.xlabel("# epoch")
    plt.ylabel("Value")
    plt.legend(loc="upper left")
    #plt.savefig(args["plot"])
    plt.show()


#------------------------------------------------------------------------------
def run_ae_visual_validation(model, features_pred, Y_pred, Y_test, array_filenames):
    for i in range(len(Y_pred)):
        input = utilLoadImages.load_one_image(array_filenames[i], resize=512)

        #print(np.min(Y_pred[i]), np.max(Y_pred[i]))
        #
        output = np.array(Y_pred[i] * 255., dtype = np.uint8)
        output = cv2.resize(output, (512, 512), interpolation = cv2.INTER_CUBIC)

        aux = cv2.normalize(features_pred[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        aux = np.array(aux * 255., dtype = np.uint8)
        aux = cv2.resize(aux, (512, 512), interpolation = cv2.INTER_CUBIC)

        cv2.putText(input,"Input", (0,0),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('Input',input)
        cv2.putText(output,"Prediction", (0,0),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('Prediction',output)
        cv2.putText(aux,"Features", (0,0),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('Features',aux)
        cv2.waitKey(0)


#------------------------------------------------------------------------------
def run_visual_validation(model, Y_pred, Y_test, array_filenames, array_labels):
    for i in range(len(Y_pred)):
        proba = Y_pred[i]                           
        proba_idxs = np.argsort(proba)[::-1]                                    #probabilidad de cada característica [0,1,1,0...] ordenada  
#        print(array_filenames[i],'-',proba,'-', proba_idxs)

        img = utilLoadImages.load_one_image(array_filenames[i], resize=512)

        for (idx, j) in enumerate(proba_idxs):
            # build the label and draw it on the image if probabilit y > 0
            if proba[j] >= 0.0001:                                             #valor de la probabilidad de cada característica  ordenada
                label = "{}: {:.2f}%".format(array_labels[j], proba[j] * 100)
                cv2.putText(img, label, (10, (idx * 30) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        str = ''   # Add text with the GT
        cv2.putText(img, array_filenames[i], (5,250), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        for idx, c in enumerate(Y_test[i]):         #Y_test con etiquetas reales
            if c == 1:
                str += array_labels[idx] + '-'
        cv2.putText(img, str, (10, (13 * 30) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        cv2.imshow('Prediction', img)
        cv2.waitKey(0)


#-----------------------------------------------------------------------------
def save_NC(model, X, array_filenames, csv_filename):
    model_features = utilModel.create_features_model(model)
    features = model_features.predict(X, batch_size=args.batch)
    print(' - Output shape:', features.shape)
    #print( array_filenames[0] )

    if len(features[0].shape) > 1:
        new_shape = features[0].shape[0] * features[0].shape[1] * features[0].shape[2]
        features = np.reshape(features, (len(X), new_shape))
        print('  -> After reshape:', features.shape)

    array_filenames = map(os.path.basename, array_filenames)
    utilLoadImages.add_to_csv(csv_filename, array_filenames, features)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Logos')
parser.add_argument('-csv',   required=True,   type=str,   help='Path to CSV file.')
parser.add_argument('-img',   required=True,   type=str,   help='Path to images.')
parser.add_argument('-resize', default=256,     type=int,   help='Resize to this size.')
parser.add_argument('--aug',   action='store_false',            help='Use data augmentation.')
parser.add_argument('-type',   default='maincat',  type=str, help='Classification type',
                                              choices=['mainsec', 'subsec', 'maincat', 'subcat', 'color', 'shape', 'text', 'ae'])
parser.add_argument('-e',      default=100,    type=int,   dest='epochs',         help='Number of epochs')
parser.add_argument('-b',      default=16,     type=int,   dest='batch',            help='Batch size')
parser.add_argument('--v',     action='store_true',        dest='verbose',     help='Activate verbose')
parser.add_argument('-truncate', default=-1,     type=int,   help='Truncate at this size. -1 to deactivate')
parser.add_argument('--load',  action='store_true',            help='Only test.')
parser.add_argument('--save',  action='store_true',            help='Save Neural Codes.')
parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')
parser.add_argument('-p',      default=1000,    type=int,   dest='page_size',         help='Page size')
args = parser.parse_args()

print('Loading data...')
csv_data= utilLoadImages.load_csv(args.csv)
assert len(csv_data) > 0
assert len(csv_data.shape) == 2
assert csv_data.shape[1] == 6       

if not os.path.isdir('WEIGHTS'):
  os.makedirs('WEIGHTS')

args.WEIGHTS_FILENAME = 'WEIGHTS/weights' \
                                                    + '_type_' + args.type \
                                                    + '_size'+ str(args.resize) \
                                                    + ('_aug' if args.aug else '') \
                                                    + '_e' + str(args.epochs) \
                                                    + '_b' + str(args.batch) \
                                                    + ('_TRUNCATE'+str(args.truncate) if args.truncate > 0 else '') \
                                                    + '.h5'
early_stopping = EarlyStopping(monitor='loss', patience=15)

train_files, test_files = train_test_split(csv_data, test_size=0.2, random_state=42)

print(' - WEIGHTS_FILENAME:', args.WEIGHTS_FILENAME)
print(' - Classification type:', args.type)
print(' - Img size:', args.resize)
print(' - Total train images:', len(train_files))
print(' - Total test images:', len(test_files))
print(' - Epochs:', args.epochs)
print(' - Batch:', args.batch)
print(' - Aug:', args.aug)
print(' - Page size:', args.page_size)

if args.load == False:
   pos = 0
   while True:
     from_file = pos * args.page_size
     to_file = (pos + 1) * args.page_size
     pos += 1

     if from_file >= len(train_files):
       break

     if to_file >= len(train_files):
       to_file = len(train_files) - 1

     print(80*'-')
     print("# Loading page from %d to %d..." % ((from_file, to_file)))
    
#    X_train = load_images( train_files[from_file:to_file] )
#    mapear data de train_files según el valor de 'type'
     array_filenames, X, Y_mainsec, Y_subsec, Y_maincat, Y_subcat, Y_color, Y_shape, Y_text = \
         utilLoadImages.read_dataset( train_files[from_file:to_file], args.img, args.resize, args.truncate, args.type)
    
     if args.type == 'mainsec':
       array_labels = utilLoadImages.MAINSEC
       Y_train = Y_mainsec
       if pos == 1:
         model = utilModel.get_model(X.shape[1:], classes=Y_mainsec.shape[1])
         print(model.summary())
     elif args.type == 'subsec':
       array_labels = utilLoadImages.SUBSEC
       Y_train = Y_subsec
       if pos == 1:
         model = utilModel.get_model(X.shape[1:], classes=Y_subsec.shape[1])
         print(model.summary())
     elif args.type == 'maincat':
       array_labels = utilLoadImages.MAINCAT
       Y_train = Y_maincat
       if pos == 1:
         model = utilModel.get_model(X.shape[1:], classes=Y_maincat.shape[1])
         print(model.summary())
     elif args.type == 'subcat':
       array_labels = utilLoadImages.SUBCAT
       Y_train = Y_subcat
       if pos == 1:
         model = utilModel.get_model(X.shape[1:], classes=Y_subcat.shape[1])
         print(model.summary())
     elif args.type == 'color':
       array_labels = utilLoadImages.COLORS
       Y_train = Y_color
       if pos == 1:
         model = utilModel.get_model(X.shape[1:], classes=Y_color.shape[1])
         print(model.summary())
     elif args.type == 'shape':
       array_labels = utilLoadImages.SHAPE
       Y_train = Y_shape
       if pos == 1:
         model = utilModel.get_model(X.shape[1:], classes=Y_shape.shape[1])
         print(model.summary())
     elif args.type == 'text':
       array_labels = utilLoadImages.TEXT
       Y_train = Y_text
       if pos == 1:
         model = utilModel.get_model(X.shape[1:], classes=Y_text.shape[1], categorical=True)
         print(model.summary())

     elif args.type == 'ae':
       array_labels = None
       Y_train = X
       if pos == 1:
         model = utilModel.get_autoencoder( X.shape[1:] )
         print(model.summary())

     else:
       raise Exception('Unknown type:' + args.type)

     print(' - Total Y_train:', len(Y_train))


     print('Fit network...')
     model.fit(X, Y_train, batch_size= args.batch,
                     epochs= args.epochs,
                     validation_split= 0.1,
                     shuffle=True,
                     verbose= args.verbose,
                     callbacks=[early_stopping])

     model.save_weights(args.WEIGHTS_FILENAME, overwrite=True)


#Save NC train
     if args.save == True:
       if not os.path.isdir('NC'):
         os.makedirs('NC')
       csv_filename = 'NC/features_' + args.WEIGHTS_FILENAME.replace('.h5', '').replace('WEIGHTS/weights_', '')
       print('Save NC for train data...')
       save_NC(model, X, array_filenames, csv_filename + '_train.csv')


# mapear data de test
print('Loading data test...')
array_filenames, X_test, Y_mainsec, Y_subsec, Y_maincat, Y_subcat, Y_color, Y_shape, Y_text = \
    utilLoadImages.read_dataset( test_files, args.img, args.resize, args.truncate, args.type)


if args.type == 'mainsec':
  Y_test = Y_mainsec
  if args.load == True:
    model = utilModel.get_model(X_test.shape[1:], classes=Y_mainsec.shape[1])
    array_labels = utilLoadImages.MAINSEC 
elif args.type == 'subsec':
  Y_test = Y_subsec
  if args.load == True:
    model = utilModel.get_model(X_test.shape[1:], classes=Y_subsec.shape[1]) 
    array_labels = utilLoadImages.SUBSEC 
elif args.type == 'maincat':
  Y_test = Y_maincat
  if args.load == True:
    model = utilModel.get_model(X_test.shape[1:], classes=Y_maincat.shape[1]) 
    array_labels = utilLoadImages.MAINCAT
elif args.type == 'subcat':
  Y_test = Y_subcat
  if args.load == True:
    model = utilModel.get_model(X_test.shape[1:], classes=Y_subcat.shape[1]) 
    array_labels = utilLoadImages.SUBCAT
elif args.type == 'color':
  Y_test = Y_color
  if args.load == True:
    model = utilModel.get_model(X_test.shape[1:], classes=Y_color.shape[1]) 
    array_labels = utilLoadImages.COLORS
elif args.type == 'shape':
  Y_test = Y_shape
  if args.load == True:
    model = utilModel.get_model(X_test.shape[1:], classes=Y_shape.shape[1]) 
    array_labels = utilLoadImages.SHAPE
elif args.type == 'text':
  Y_test = Y_text
  if args.load == True:
    model = utilModel.get_model(X_test.shape[1:], classes=Y_text.shape[1], categorical=True) 
    array_labels = utilLoadImages.TEXT
elif args.type == 'ae':
  Y_test = X_test
  if args.load == True:
    model = utilModel.get_autoencoder(X_test.shape[1:] )

# Save NC test...
if args.save == True:
    #csv_filename = 'NC/features_' + args.WEIGHTS_FILENAME.replace('.h5', '').replace('WEIGHTS/weights_', '')
    print('Save NC for test data...')
    save_NC(model, X_test, array_filenames, csv_filename + '_test.csv')


print('Loading weights...')
model.load_weights(args.WEIGHTS_FILENAME)


# Evaluation...
print('Evaluate...')
score = model.evaluate(X_test, Y_test, batch_size= args.batch, verbose=0)
print(' - Test loss:', score[0])
print(' - Test metric:', score[1])

Y_pred = model.predict(X_test, batch_size= args.batch)

if args.type == 'ae':
    model_features = utilModel.create_features_model(model)
    features_pred = model_features.predict(X_test, batch_size= args.batch)
    print(' - Features shape:', features_pred.shape)
#    run_ae_visual_validation(model, features_pred, Y_pred, Y_test, array_filenames)
elif args.type == 'text':
    Y_pred=np.argmax(Y_pred, axis=1) 
    Y_test=np.argmax(Y_test, axis=1)     
    print(' - Accuracy:', accuracy_score(Y_test, Y_pred))
    print(' - F1 score macro:', f1_score(Y_test, Y_pred, average='macro') )
    print(' - F1 score micro:', f1_score(Y_test, Y_pred, average='micro') )

else:
    # https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
    print(' - Coverage error:', coverage_error(Y_test, Y_pred))
    print(' - Label ranking average precision score:', label_ranking_average_precision_score(Y_test, Y_pred))
    print(' - Label ranking loss:', label_ranking_loss(Y_test, Y_pred))

    run_visual_validation(model, Y_pred, Y_test, array_filenames, array_labels)
                                                    
