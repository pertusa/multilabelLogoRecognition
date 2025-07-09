#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings
import argparse
import random, math, time
import cv2
import traceback
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import utilLoadImages
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

utilLoadImages.init()


#------------------------------------------------------------------------------
def get_labels(array_ids, array_labels, t):
    Y = {'mainsec': [], 'subsec':[], 'maincat':[], 'subcat':[], 'color':[], 'shape':[], 'text':[], 'ae':[]}

    for id in array_ids:
        mainsec, subsec = utilLoadImages.get_label_sector( array_labels[id][utilLoadImages.COL_SECTOR] )

        Y['mainsec'].append( mainsec )
        Y['subsec'].append( subsec )
        Y['maincat'].append( utilLoadImages.get_labels_maincat( array_labels[id][utilLoadImages.COL_CATEGORIES],t ) )

        Y['subcat'].append( utilLoadImages.get_labels_subcat( array_labels[id][utilLoadImages.COL_CATEGORIES],t ) )
        Y['color'].append( utilLoadImages.get_labels_colors( array_labels[id][utilLoadImages.COL_COLOR],t ) )

        Y['shape'].append( utilLoadImages.get_labels_shape( array_labels[id][utilLoadImages.COL_CATEGORIES],t ) )
        Y['text'].append( utilLoadImages.get_labels_text( array_labels[id][utilLoadImages.COL_CATEGORIES], array_labels[id][utilLoadImages.COL_TEXT]) )
        Y['ae'].append(array_labels[id])

    for key in Y.keys():
        Y[key] = np.asarray(Y[key])
        #print(key, Y[key].shape)

    return Y

"""
#------------------------------------------------------------------------------
def read_csv_file(filename):
    content = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            content.append(line.split(','))

    #with open(filename) as f:
    #    content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    #content = [x.strip() for x in content]
    content = np.asarray(content)
    #for idx in range(len(content)):
    #    content[idx] =

    #content = np.asarray(content)
    print(content.shape)
    quit()
    return content"""


#------------------------------------------------------------------------------
def load_data(train_path, test_path, csv_path, tipo):
    #data_train = read_csv_file(args.train)

    data_train = utilLoadImages.load_csv( train_path )
    data_test = utilLoadImages.load_csv( test_path )
    csv_data = utilLoadImages.load_csv( csv_path )
    assert len(data_train) > 0
    assert len(data_test) > 0
    assert len(csv_data) > 0
    assert len(data_train.shape) == 2
    assert len(data_test.shape) == 2
    assert len(csv_data.shape) == 2
    assert data_train.shape[1] > 1
    assert data_test.shape[1] > 1
    assert csv_data.shape[1] == 6      

    X_train = data_train[:, 1:].astype('float32')       #datos de train.csv
    ID_train = data_train[:, 0].astype('str')           #nombre de imagen train.csv
    X_test = data_test[:, 1:].astype('float32')
    ID_test = data_test[:, 0].astype('str')

    csv_dict = {str(row[1]):row for row in csv_data}    #datos en salida.csv

    Y_train = get_labels(ID_train, csv_dict, tipo)      #característica de cada tipo Y_train.Y[mainsec], Y_train.Y[subsec], Y_train.Y[manicat]... 
    Y_test = get_labels(ID_test, csv_dict, tipo)

#    print ('Y_train', Y_train['mainsec'][0])            
    
    for i in range(len(X_train)):
        utilLoadImages.l2norm( X_train[i] )

    for i in range(len(X_test)):
        utilLoadImages.l2norm( X_test[i] )

    return ID_train, X_train, Y_train, ID_test, X_test, Y_test


#------------------------------------------------------------------------------
def labels_to_str(array_labels, Y_test, idx):
    str = ''
    for key in array_labels.keys():
        aux = []
#        print('*---------',key,'-',idx,'-',Y_test[key][0])  #maincat - 0 - None
        for label_idx in range(len(Y_test[key][idx])):
            if Y_test[key][idx][label_idx] == 1:
                aux.append(array_labels[key][label_idx])
        str += 'Labels for {}: {}\n'.format(key, ' | '.join(aux))
    return str


#------------------------------------------------------------------------------
def show_or_save_neighbors(clf, ID_train, Y_train, ID_test, Y_test, array_labels, config):
    img_size = 128
    nb_results = 10

    #array_labels = { config.type: array_labels[config.type] }              # TODO: remove (modified to show only that class)

    neighbors =  clf.kneighbors(X_test, n_neighbors=nb_results, return_distance=False)

    for idx, neig_idxs in enumerate(neighbors):

        ###   # TODO: remove
        """num_good_results = 0
        for neig in neig_idxs:
            if np.array_equal(Y_test[config.type][idx], Y_train[config.type][neig]) == True \
                and np.sum(Y_test[config.type][idx]) > 0:
                num_good_results += 1

        if num_good_results < 3:
            print('Not used...')
            continue"""
        ###

        print(80*'-')
        #print(idx, neig_idxs)
        #img_result = np.zeros((img_size, img_size, ))

        img_result = utilLoadImages.load_one_image(os.path.join(config.img, ID_test[idx]), img_size)
        #cv2.imshow("Query", img_result)
        #cv2.waitKey(0)

        print( labels_to_str(array_labels, Y_test, idx) )
        print( '--\nLabels of the first neighbor:' )
        print( labels_to_str(array_labels, Y_train, neig_idxs[0]) )

        for neig in neig_idxs:
            img_neig = utilLoadImages.load_one_image(os.path.join(config.img, ID_train[neig]), img_size)
            img_result = np.hstack((img_result, img_neig))

        if config.v:
            cv2.imshow("Result", img_result)
            cv2.waitKey(0)

        if config.save:
            filename = os.path.join('OUT_NEIGHBORS', config.type, str(idx))
            cv2.imwrite( filename + '.jpg', img_result )
            with open(filename + '.txt', 'w') as f:
                f.write( labels_to_str(array_labels, Y_test, idx) )
                f.write( '--\nLabels of the first neighbor:' )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[0]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[1]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[2]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[3]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[4]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[5]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[6]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[7]) )
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[8]) )     
                f.write( labels_to_str(array_labels, Y_train, neig_idxs[9]) )

# -----------------------------------------------------------------------------
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


#------------------------------------------------------------------------------
def evaluate_one_knn(X_train, Y_train, X_test, Y_test, k, from_type, to_type):
        clf = KNeighborsClassifier(n_neighbors = k)  #, weights='uniform', algorithm = 'kd_tree')
        clf.fit(X_train, Y_train)                                                      

        """from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=500,random_state=0)
        clf.fit(X_train, Y_train)
        """

        Y_pred = clf.predict(X_test)                                           

        #neighbors =  clf.kneighbors(X_test, n_neighbors=1, return_distance=False)    
        #print(neighbors)
        #print(neighbors.shape, Y_train.shape)
        #Y_pred = np.take(Y_train, neighbors, axis=0).squeeze()
        #print(Y_pred.shape, Y_test.shape)
        #print(Y_pred[0], Y_test[0])
        #quit()

        #print(' - Test score:\t', clf.score(X_test, Y_test))
        #print(' - Coverage error:\t', coverage_error(Y_test, Y_pred))
        #print(' - Label ranking average precision score:\t', label_ranking_average_precision_score(Y_test, Y_pred))
        #print(' - Label ranking loss:\t', label_ranking_loss(Y_test, Y_pred))

        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(from_type, to_type, k,
                                                    clf.score(X_test, Y_test),
                                                    coverage_error(Y_test, Y_pred),
                                                    label_ranking_average_precision_score(Y_test, Y_pred),
                                                    label_ranking_loss(Y_test, Y_pred),
                                                    hamming_score(Y_test, Y_pred)))


#------------------------------------------------------------------------------
def evaluate_array_knn(X_train, Y_train, X_test, Y_test, array_k_values, array_labels_to_evaluate, config):
    print('Evaluate kNN...')

    for label in array_labels_to_evaluate:
        #print(80*'-')
        print('{}\t->\t{}'.format(config.type, label))

        X_train_aux, Y_train_aux = filter_set(X_train, Y_train[label])
        X_test_aux, Y_test_aux = filter_set(X_test, Y_test[label])

        for k in array_k_values:
            evaluate_one_knn(X_train_aux, Y_train_aux, X_test_aux, Y_test_aux,
                                                   k, config.type, label)


#------------------------------------------------------------------------------
def filter_set(X, Y):
#    print('*---------------------------------------------Filter set X-Y-X ',X[0],'-',Y[0],'-',X[1])
#    Y_sum = np.sum(Y, axis=1)
#    index = np.argwhere(Y_sum > 1)        
#    print(index,'-',X[index])
#    X = np.delete( X, index, axis=0)      
#    Y = np.delete( Y, index, axis=0)      
#    print(index,'-',X[index],'-',Y[index])
    return X, Y


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logos kNN')
    parser.add_argument('-train',   required=True,   type=str,   help='Path to CSV file with the train data.')
    parser.add_argument('-test',   required=True,   type=str,   help='Path to CSV file with the test data.')
    parser.add_argument('-csv',   required=True,   type=str,   help='Path to CSV file.')
    parser.add_argument('-img',   required=True,   type=str,   help='Path to images.')
    parser.add_argument('-type',   default='color',  type=str, help='Classification type',
                                                choices=['mainsec', 'subsec', 'maincat', 'subcat', 'color', 'shape', 'text', 'ae'])
    parser.add_argument('-values',      default='1,3',       type=str,   help='List of k values to test')
    parser.add_argument('--v',  action='store_true',            help='Show images.')
    parser.add_argument('--save',  action='store_true',            help='Save images with the nearest neighbors.')
    args = parser.parse_args()

    if args.save:
            if not os.path.isdir(os.path.join('OUT_NEIGHBORS', args.type)):
                os.makedirs(os.path.join('OUT_NEIGHBORS', args.type))

    print('Loading data...')
    ID_train, X_train, Y_train, ID_test, X_test, Y_test = \
                            load_data(args.train, args.test, args.csv, args.type)            

    print(' - Type:', args.type)
    print(' - Train features:', args.train)
    print(' - Test features:', args.test)
    print(' - X_train:', X_train.shape)
    print(' - Y_train:', ','.join([str(Y_train[key].shape) for key in Y_train.keys()]))
    print(' - X_test:', X_test.shape)
    print(' - Y_test:', ','.join([str(Y_test[key].shape) for key in Y_test.keys()]))

    #for key in Y_train.keys():
    #    print(key,'-',Y_train[key].shape) ,'-',Y_train[key][0])  #-'maincat' (14724, 45) [o,1,o...] 
        
    array_labels = {}                         
#    array_labels['mainsec'] = utilLoadImages.MAINSEC
#    array_labels['subsec'] = utilLoadImages.SUBSEC
#    array_labels['maincat'] = utilLoadImages.MAINCAT
#    array_labels['subcat'] = utilLoadImages.SUBCAT
#    array_labels['color'] = utilLoadImages.COLORS
#    array_labels['shape'] = utilLoadImages.SHAPE
#    array_labels['text'] = utilLoadImages.TEXT
    array_labels['ae'] = None


    # https://scikit-learn.org/stable/modules/multiclass.html

    # kNN classifier 
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    array_k_values = map(int, args.values.split(','))

    array_labels_to_evaluate = [args.type] if args.type != 'ae' else array_labels.keys()

    print('array_labels_to_evaluate', array_labels_to_evaluate)

    evaluate_array_knn(X_train, Y_train, X_test, Y_test, array_k_values, array_labels_to_evaluate, args)


    # Show images with the k nearest neighbors...
    if args.v or args.save:
        clf = KNeighborsClassifier(n_neighbors = 1)  #, weights='uniform', algorithm = 'kd_tree')
        clf.fit(X_train, Y_train[args.type])
        show_or_save_neighbors(clf, ID_train, Y_train, ID_test, Y_test, array_labels, args)

