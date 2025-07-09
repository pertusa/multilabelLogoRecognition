#coding= utf-8
#util
from __future__ import print_function

import random
import math
import numpy as np
import sys, os
import tensorflow as tf

def init():
    np.set_printoptions(threshold=sys.maxsize)
    sys.setrecursionlimit(40000)
    random.seed(42)                             # For reproducibility
    np.random.seed(42)
    tf.set_random_seed(42)

#------------------------------------------------------------------------------
def add_to_csv(filename, labels, features):
    #n_dim = len(features[0])
    with open(filename, 'a') as f:
        for i in range(len(features)):
            str_features = ','.join(map(str, features[i]))
            f.write('{},{}\n'.format(labels[i], str_features))
#------------------------------------------------------------------------------            
#utilLoadimages
#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np
import pandas as pd
#import util

COL_MARK_FEATURE = 0
COL_IMG_FILENAME = 1
COL_CATEGORIES = 2
COL_COLOR = 3
COL_SECTOR = 4
COL_TEXT = 5

COLORS = ['Red','Yellow','Green','Blue','Violet','White','Brown','Black','Silver','Grey','Gold','Orange','Pink']
MAINSEC = ['Goods', 'Services']
MAINCAT = [ 'Celestial Bodies, Natural Phenomena, Geographical Maps',
                        'Human Beings',
                        'Animals',
                        'Supernatural, Fabulous, Fantastic Or Unidentifiable Beings',
                        'Plants',
                        'Landscapes',
                        'Constructions, Structures For Advertisements, Gates Or Barriers',
                        'Foodstuffs',
                        'Textiles, Clothing, Sewing Accessories, Headwear, Footwear',
                        'Tobacco, Smokers Requisites, Matches, Travel Goods, Fans, Toilet Articles',
                        'Household Utensils',
                        'Furniture, Sanitary Installations',
                        'Lighting, Wireless Valves, Heating, Cooking Or Refrigerating Equipment, Washing Machines, Drying Equipment',
                        'Ironmongery, Tools, Ladders',
                        'Machinery, Motors, Engines',
                        'Telecommunications, Sound Recording Or Reproduction, Computers, Photography, Cinematography, Optics',
                        'Horological Instruments, Jewelry, Weights And Measures',
                        'Transport, Equipment For Animals',
                        'Containers And Packing, Representations Of Miscellaneous Products',
                        'Writing, Drawing Or Painting Materials, Office Requisites, Stationery And Booksellers Goods',
                        'Games, Toys, Sporting Articles, Roundabouts',
                        'Musical Instruments And Their Accessories, Music Accessories, Bells, Pictures, Sculptures',
                        'Arms, Ammunition, Armour',
                        'Heraldry, Coins, Emblems, Symbols',
                        'Ornamental motifs, Surfaces or Backgrounds with Ornaments'
                    ]
SUBCAT = [  'Stars, Comets',
                        'Sun',
                        'Earth, Terrestrial Globes, Planets',
                        'Moon',
                        'Constellations, Groups Of Stars, Starry Sky, Celestial Globes, Celestial Maps',
                        'Armillary Spheres, Planetaria, Astronomic Orbits, Atomic Models, Molecular Models',
                        'Natural Phenomena',
                        'Geographical Maps, Planispheres',
                        'Men',
                        'Women',
                        'Children',
                        'Mixed Groups, Scenes',
                        'Parts Of The Human Body, Skeletons, Skulls',
                        'Quadrupeds (Series I)',
                        'Quadrupeds (Series Ii)',
                        'Quadrupeds (Series Iii)',
                        'Quadrupeds (Series Iv)',
                        'Quadrupeds (Series V), Quadrumana',
                        'Parts Of The Bodies, Skeletons, Skulls Of Quadrupeds Or Of Quadrumana',
                        'Birds, Bats',
                        'Aquatic Animals, Scorpions',
                        'Reptiles, Amphibia, Snails, Seals, Sea Lions',
                        'Insects, Spiders, Micro-Organisms',
                        'Other Animals; Large Prehistoric Animals',
                        'Groups Of Animals Classified In Different Divisions Of Category 3',
                        'Winged Or Horned Personages',
                        'Beings Partly Human And Partly Animal',
                        'Fabulous Animals',
                        'Plants, Objects Or Geometrical Figures Representing A Personage Or An Animal; Masks Or Fantastic Or Unidentifiable Heads',
                        'Groups Of Figurative Elements Classified In Different Divisions Of Category 4',
                        'Trees, Bushes',
                        'Leaves, Needles, Branches With Leaves Or Needles',
                        'Flowers, Blossoms',
                        'Grain, Seeds, Fruits',
                        'Vegetables',
                        'Other Plants',
                        'Decorations Made Of Plants',
                        'Mountains, Rocks, Grottoes',
                        'Landscapes With Water, River Or Stream',
                        'Desert Or Tropical-Type Landscapes',
                        'Urban Landscapes Or Village Scenes',
                        'Other Landscapes',
                        'Dwellings, Buildings, Advertisement Hoardings Or Pillars, Cages Or Kennels For Animals',
                        'Parts Of Dwellings Or Of Buildings, Interiors',
                        'Monuments, Stadiums, Fountains',
                        'Structural Works',
                        'Building Materials, Walls, Gates Or Barriers, Scaffolding',
                        'Bakers Products, Pastry, Confectionery, Chocolate',
                        'Milk, Dairy Products, Cheeses',
                        'Butchers Meat, Pork Products, Fishmongers Products',
                        'Other Foodstuffs',
                        'Textiles Other Than Clothing, Shuttles',
                        'Clothing',
                        'Sewing Accessories, Patterns For Dressmaking',
                        'Headwear',
                        'Footwear',
                        'Tobacco, Smokers Requisites, Matches',
                        'Travel Goods, Fans, Bags',
                        'Toilet Articles, Mirrors',
                        'Knives, Forks And Spoons, Kitchen Utensils And Machines',
                        'Containers For Beverages, Plates And Dishes, Kitchen Utensils For Serving, Preparing Or Cooking Food Or Drink',
                        'Other Household Utensils',
                        'Furniture',
                        'Sanitary Installations',
                        'Lighting, Wireless Valves',
                        'Heating, Cooking Or Refrigerating Equipment, Washing Machines, Drying Equipment',
                        'Tubes, Cables, Heavy Ironmongery Articles',
                        'Small Ironmongery Articles, Springs',
                        'Keys For Locks, Locks, Padlocks',
                        'Tools',
                        'Agricultural Or Horticultural Implements, Ice Axes',
                        'Ladders',
                        'Machines For Industry Or Agriculture, Industrial Installations, Motors, Engines, Various Mechanical Appliances',
                        'Machines For Household Use, Spinning Wheels',
                        'Office Machines',
                        'Wheels, Bearings',
                        'Electrical Equipment',
                        'Telecommunications, Sound Recording Or Reproduction, Computers',
                        'Photography, Cinematography, Optics',
                        'Horological And Other Time-Measuring Instruments',
                        'Jewelry',
                        'Balances, Weights',
                        'Measures',
                        'Land Vehicles',
                        'Equipment For Animals',
                        'Vehicles For Use On Water And Amphibious Vehicles',
                        'Anchors; Buoys Or Lifebelts',
                        'Aerial Or Space Vehicles',
                        'Traffic Signs And Indicator Boards',
                        'Large Containers',
                        'Small Containers',
                        'Bottles, Flasks',
                        'Parts Or Accessories Of Bottles',
                        'Amphorae, Pitchers, Vases, Flower Pots, Flower Stands',
                        'Coffins, Funerary Urns',
                        'Receptacles For Laboratory Use And For Pharmacy',
                        'Medical Or Surgical Apparatus, Instruments Or Utensils, Prostheses, Medicines',
                        'Representations Of Miscellaneous Products',
                        'Writing, Drawing Or Painting Materials, Small Office Requisites',
                        'Papers, Documents',
                        'Books, Bookbindings, Newspapers',
                        'Games, Toys',
                        'Sporting Articles, Roundabouts',
                        'Musical Instruments, Musical Instrument Accessories, Music Accessories',
                        'Bells',
                        'Pictures, Sculptures',
                        'Side Arms, Other Weapons Not Being Firearms',
                        'Firearms, Ammunition, Explosives',
                        'Armour',
                        'Shields',
                        'Seals, Stamps',
                        'Medals, Coins, Decorations, Orders',
                        'Flags',
                        'Crowns, Diadems',
                        'Emblems, Insignia',
                        'Crosses',
                        'Arrows',
                        'Signs, Notations, Symbols',
                        'ORNAMENTAL MOTIFS',
                        'HORIZONTALLY ELONGATED ORNAMENTAL SURFACES',
                        'BACKGROUNDS DIVIDED INTO TWO OR FOUR',
                        'SURFACES OR BACKGROUNDS COVERED WITH REPEATED FIGURATIVE ELEMENTS OR INSCRIPTIONS',
                        'SURFACES OR BACKGROUNDS COVERED WITH OTHER ORNAMENTS'
                    ]
SUBSEC = ['Chemicals',
                    'Paints, varnishes, lacquers',
                    'Non-medicated cosmetics and toiletry preparations',
                    'Industrial oils and greases, wax; lubricants',
                    'Pharmaceuticals, medical and veterinary preparations',
                    'Common metals and their alloys, ores',
                    'Machines, machine tools, power-operated tools',
                    'Hand tools and implements, hand-operated',
                    'Scientific, nautical, surveying, photographic, cinematographic, optical, weighing, measuring, signalling, checking (supervision), life-saving and teaching apparatus and instruments',
                    'Surgical, medical, dental and veterinary apparatus and instruments; artificial limbs, eyes and teeth',
                    'Apparatus for lighting, heating, steam generating, cooking, refrigerating, drying, ventilating, water supply and sanitary purposes',
                    'Vehicles',
                    'Firearms',
                    'Precious metals and their alloys',
                    'Musical instruments',
                    'Paper and cardboard',
                    'Unprocessed and semi-processed rubber, gutta-percha, gum, asbestos, mica and substitutes for all these materials',
                    'Leather and imitations of leather',
                    'Building materials (non-metallic)',
                    'Furniture, mirrors, picture frames',
                    'Household or kitchen utensils and containers',
                    'Ropes and string',
                    'Yarns and threads, for textile use',
                    'Textiles and substitutes for textiles',
                    'Clothing, footwear, headgear',
                    'Lace and embroidery, ribbons and braid',
                    'Carpets, rugs, mats and matting, linoleum and other materials for covering existing floors',
                    'Games, toys and playthings',
                    'Meat, fish, poultry and game',
                    'Coffee, tea, cocoa and artificial coffee',
                    'Raw and unprocessed agricultural, aquacultural, horticultural and forestry products',
                    'Beers',
                    'Alcoholic beverages (except beers)',
                    'Tobacco',
                    'Advertising',
                    'Insurance',
                    'Building construction; repair; installation services',
                    'Telecommunications',
                    'Transport',
                    'Treatment of materials',
                    'Education',
                    'Scientific and technological services and research and design relating thereto',
                    'Services for providing food and drink',
                    'Medical services',
                    'Legal services'
                    ]
SHAPE = [  'CIRCLES, ELLIPSES',
                    'SEGMENTS OR SECTORS OF CIRCLES OR ELLIPSES',
                    'TRIANGLES, LINES FORMING AN ANGLE',
                    'QUADRILATERALS',
                    'OTHER POLYGONS',
#                    'DIFFERENT GEOMETRICAL FIGURES, JUXTAPOSED, JOINED OR INTERSECTING',
                    'LINES, BANDS',
#                    'OTHER GEOMETRICAL FIGURES, INDEFINABLE DESIGNS',
                    'GEOMETRICAL SOLIDS'
                ]
TEXT = ['Yes', 'No']


#------------------------------------------------------------------------------
def crop_borders(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY_INV)

    rows, cols = thresh.shape
    non_empty_columns = np.where(thresh.max(axis=0) > 0)[0]
    non_empty_rows = np.where(thresh.max(axis=1) > 0)[0]
    cropBox = ( min(non_empty_rows),
                            min(max(non_empty_rows), rows),
                            min(non_empty_columns),
                            min(max(non_empty_columns), cols))

    return img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

# ----------------------------------------------------------------------------
def l2norm(X):
    norm = 0
    for i in range(len(X)):
        if X[i] < 0:
            X[i] = 0
        else:
            norm += X[i] * X[i]
    if norm != 0:
        norm = math.sqrt(norm)
        X /= norm


#------------------------------------------------------------------------------
def load_one_image(path, resize):
#    assert os.path.isfile(path), path

    img = None

    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        #cv2.imshow("Img", img)
        #cv2.waitKey(0)

        img = crop_borders(img)
        #cv2.imshow("Img", img)
        #cv2.waitKey(0)

        if resize != 0: #256:
            img = cv2.resize(img, (resize, resize), interpolation = cv2.INTER_CUBIC)

    except:
        print('ERROR - IMG:', path)

    return img


#------------------------------------------------------------------------------
# https://www.wipo.int/classifications/nice/nclpub/en/fr/20180101/classheadings/?class_heading=2&explanatory_notes=show&lang=en&menulang=en
# # El sector es un unico numero y esta comprendido entre el 1 y el 45.
# # Del 1 al 34 son goods (bienes o mercancias) y del 35 al 45 servicios.
# # Ejemplo: 35;38;39
def get_label_sector(data):
    data = data.replace('.', ';')           # TODO - review
    csv_sectors = map(int, data.split(';'))
    array_mainsec = np.zeros((2))
    array_subsec = np.zeros((45))

    for s in csv_sectors:
        assert s >= 1 and s <= 45
        #print(s)

        mainsec = int(s>34)    #productos <=34 > Servicios 
        #assert array_mainsec[mainsec] == 0, 'This logo is in both main sectors'

        array_mainsec[mainsec] = 1
        array_subsec[s-1] = 1

    #print(array_mainsec, array_subsec)
    #util.pause()

    return array_mainsec, array_subsec


#------------------------------------------------------------------------------
def get_labels_maincat(csv_data,tipo=None): 
    # CSV example: 02.09.04;02.09.25;25.05.99;26.02.07;26.07.25;26.11.02;26.11.12;27.05.12
    array_maincat = np.zeros((25))
    has_cat = False

    for c in csv_data.split(';'):
        cat = int(c.split('.')[0])
        if cat > 25:
            continue
        array_maincat[ cat -1 ] = 1
        has_cat = True
    
    if tipo == 'ae':                   #MB para py_logos_knn(key,'ae')
       return array_maincat
    else:
        return array_maincat if has_cat == True else None


#------------------------------------------------------------------------------
def get_labels_subcat(csv_data, tipo=None):
    # CSV example: 02.09.04;02.09.25;25.05.99;26.02.07;26.07.25;26.11.02;26.11.12;27.05.12
    # 118 subcats
    index_map = {'1.1':0,'1.3':1,'1.5':2,'1.7':3,'1.11':4,'1.13':5,'1.15':6,'1.17':7,'2.1':8,'2.3':9,'2.5':10,'2.7':11,'2.9':12,'3.1':13,'3.2':14,'3.3':15,'3.4':16,'3.5':17,'3.6':18,'3.7':19,'3.9':20,'3.11':21,'3.13':22,'3.15':23,'3.17':24,'4.1':25,'4.2':26,'4.3':27,'4.5':28,'4.7':29,'5.1':30,'5.3':31,'5.5':32,'5.7':33,'5.9':34,'5.11':35,'5.13':36,'6.1':37,'6.3':38,'6.6':39,'6.7':40,'6.19':41,'7.1':42,'7.3':43,'7.5':44,'7.11':45,'7.15':46,'8.1':47,'8.3':48,'8.5':49,'8.7':50,'9.1':51,'9.3':52,'9.5':53,'9.7':54,'9.9':55,'10.1':56,'10.3':57,'10.5':58,'11.1':59,'11.3':60,'11.7':61,'12.1':62,'12.3':63,'13.1':64,'13.3':65,'14.1':66,'14.3':67,'14.5':68,'14.7':69,'14.9':70,'14.11':71,'15.1':72,'15.3':73,'15.5':74,'15.7':75,'15.9':76,'16.1':77,'16.3':78,'17.1':79,'17.2':80,'17.3':81,'17.5':82,'18.1':83,'18.2':84,'18.3':85,'18.4':86,'18.5':87,'18.7':88,'19.1':89,'19.3':90,'19.7':91,'19.8':92,'19.9':93,'19.10':94,'19.11':95,'19.13':96,'19.19':97,'20.1':98,'20.5':99,'20.7':100,'21.1':101,'21.3':102,'22.1':103,'22.3':104,'22.5':105,'23.1':106,'23.3':107,'23.5':108,'24.1':109,'24.3':110,'24.5':111,'24.7':112,'24.9':113,'24.11':114,'24.13':115,'24.15':116,'24.17':117,'25.1':118,'25.3':119,'25.5':120,'25.7':121,'25.12':122}
    array_subcat = np.zeros((123))
    has_cat = False

    for c in csv_data.split(';'):
        cats = list(map(int, c.split('.')))
        if cats[0] > 25:
            continue
        str_subcat = str(cats[0]) + '.' + str(cats[1])
        assert str_subcat in index_map.keys()

        array_subcat[ index_map[str_subcat] ] = 1
        has_cat = True

    if tipo == 'ae':                   #MB para py_logos_knn(key,'ae')
       return array_subcat
    else:
        return array_subcat if has_cat == True else None


#------------------------------------------------------------------------------
def get_labels_colors(csv_data, tipo=None):
    array_colors = np.zeros((13))
    has_color = False

    for c in csv_data.split(';'):
        if c.startswith('29') == False:
            continue

        if c in ['29', '29.01', '29.01.11', '29.01.12', '29.01.13', '29.01.14', '29.01.15' ]:
            continue        # Not used

        assert c.startswith('29.01.')

        c = int(c.replace('29.01.', ''))
        assert (c >= 1 and c <= 8) or (c >= 95 and c <= 99)

        if c > 90:
            c -= 86

        #print(c)
        array_colors[c-1] = 1
        has_color = True

    #print(array_colors)
    #util.pause()
    if tipo == 'ae':                   #MB para py_logos_knn(key,'ae')
       return array_colors
    else:       
       return array_colors if has_color == True else None


#------------------------------------------------------------------------------
# Quitar la categoria 25 ???
def get_labels_shape(csv_data, tipo=None):
    array_shapes = np.zeros((9))
    has_shape = False
#    shape_map = {1:0, 2:1, 3:2, 4:3, 5:4, 7:5, 11:6, 13:7, 15:8}
    shape_map = {1:0, 2:1, 3:2, 4:3, 5:4, 11:5, 15:6}
    
    for c in csv_data.split(';'):
        if c.startswith('26') == False:
            continue

        c = c.replace('26.99', '26.1')  # La categoría 26.1.5 indica "when not sure about the spirals put also code 26.99.5"
        c = c.replace('26.07', '26.5')  # La categoría 26.7 se agrupa con la 26.5
        c = c.replace('26.13', '26.5')  # La categoría 26.13 se agrupa con la 26.5
        c = c.replace('26.', '')
        c = re.sub(r'\.[0-9]+', '', c)
        c = int(c)
        assert c in [1, 2, 3, 4, 5, 11, 15]

        array_shapes[ shape_map[c] ] = 1
        has_shape = True

    if tipo == 'ae':                   #MB para py_logos_knn(key,'ae')
        return array_shapes
    else:  
        return array_shapes if has_shape == True else None


#------------------------------------------------------------------------------
# Categories 27 (NO 28)
def get_labels_text(csv_data1, csv_data2):
#si longitud CRAFT > 0 o categoria == 27 (excepto 27.05.22 y 27.99.xx)     
    if int(csv_data2) > 0:             #MB longitud del archivo CRAFT > 0
#        print('CRAFT ',int(csv_data2))
        return np.array([1, 0])
    else:
        for c in csv_data1.split(';'):
#            print('NO CRAFT ',c)
            cats = map(int, c.split('.'))
            if cats[0] <> 27: 
                continue
            if cats[0] == 27 and cats[1] == 99:      #letras del alfabeto (complementaria)
                continue
            if cats[0] == 27 and cats[1] <> 5:
                    return np.array([1, 0])
            elif cats[0] == 27 and cats[1] == 5 and cats[2] not in (21,22):   #Text1 cats[2] <> 22
                return np.array([1, 0])

    return np.array([0, 1])

# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
import io
def load_csv(path, sep=',', header=None):
  df = pd.read_csv(path, sep=sep, header=header) #, engine='python')
  #df2 = pd.read_csv(io.BytesIO(uploaded[path]), sep=sep, header=header)
  return df.values

#------------------------------------------------------------------------------
# csv_data[i, j]
# i - row - features of one logo
# j - column [0, 3]
def read_dataset(csv_data, img_path, resize, truncate, type):
    array_filenames = []
    X = []
    Y_mainsec = []
    Y_subsec = []
    Y_maincat = []
    Y_subcat = []
    Y_color = []
    Y_shape = []
    Y_text = []

    print("1- longitud: ", len(csv_data)) 
#    csv_data= load_csv( csv_path )
#    assert len(csv_data) > 0
#    assert len(csv_data.shape) == 2
#    assert csv_data.shape[1] == 5

    for i in range(len(csv_data)):
        #print(i, csv_array[i])

        if csv_data[i, COL_MARK_FEATURE] != 'Figurative':
            print('NO Figurative -> DISCARDED', csv_data[i, COL_IMG_FILENAME])
            continue

        # Load image
        filename = os.path.join(img_path, csv_data[i, COL_IMG_FILENAME])
        img = load_one_image(filename, resize)
        if img is None:
          print('NO IMG -> DISCARDED', filename)
          continue

        # Load sector
        label_mainsec = None
        label_subsec = None
        if type in ('mainsec','subsec'):
          label_mainsec, label_subsec = get_label_sector( csv_data[i, COL_SECTOR] )
          if label_mainsec is None or label_subsec is None:
            print('NO SECTOR -> DISCARDED', filename)
            continue
          Y_mainsec.append( label_mainsec )
          Y_subsec.append( label_subsec )

        # Load colors
        label_colors = None
        if type == 'color':
          label_colors = get_labels_colors( csv_data[i, COL_COLOR] )
          if label_colors is None:
            print('NO COLOR -> DISCARDED', filename)
            continue
          Y_color.append( label_colors )

        # Load shape
        label_shape = None
        if type == 'shape':
          label_shape = get_labels_shape( csv_data[i, COL_CATEGORIES] )
          if label_shape is None:
            print('NO SHAPE -> DISCARDED', filename)
            continue
          Y_shape.append( label_shape )

        # Load maincat
        label_maincat = None
        if type == 'maincat':
          label_maincat = get_labels_maincat(csv_data[i, COL_CATEGORIES])
          if label_maincat is None:
            print('NO CATEGORY25 -> DISCARDED', filename)
            continue          
          Y_maincat.append( label_maincat )   

        # Load subcat
        label_subcat = None
        if type == 'subcat':
          label_subcat = get_labels_subcat(csv_data[i, COL_CATEGORIES])
          if label_subcat is None:
            print('NO CATEGORY25 -> DISCARDED', filename)
            continue          
          Y_subcat.append( label_subcat )

        #Load text (has text? only yes or no)
        label_text = None
        if type == 'text':
          label_text = get_labels_text( csv_data[i, COL_CATEGORIES], csv_data[i, COL_TEXT] )
          print('imagen-texto', filename,'-',label_text)
          Y_text.append( label_text )  

        array_filenames.append( filename )
        X.append( img )


        if truncate > 0 and len(X) >= truncate:
            print('TRUNCATED AT', truncate, '!')
            print(80*'-')
            break

    print("2") 
    X = np.asarray(X).astype('float32')
    #X = (X - np.mean(X)) / (np.std(X) + 0.00001)
    X /= 255.


    return np.asarray(array_filenames), X, \
                     np.asarray(Y_mainsec), np.asarray(Y_subsec), \
                     np.asarray(Y_maincat), np.asarray(Y_subcat), \
                     np.asarray(Y_color), np.asarray(Y_shape), np.asarray(Y_text)

