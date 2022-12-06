#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:55:33 2019

@author: dwi-lab
"""
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from random import shuffle
from keras import layers
import threading
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import PIL
from keras.models import load_model
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import argparse
from tensorflow.keras.optimizers import RMSprop
import glob
import os
from dipy.io.image import load_nifti, save_nifti
from os.path import isfile, join
import shutil
import random
from scipy.ndimage.morphology import binary_erosion
import pandas as pd
import csv
from PIL import Image



def creport (yt, y_pred):
    cm1 = confusion_matrix(yt, y_pred)
    print('Confusion Matrix : \n', cm1)
    print (cm1[0, 0] + cm1[1, 1])
    total = sum(sum(cm1))
    accuracy = float(cm1[0, 0] + cm1[1, 1]) / total
    sensitivity = float(cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1])
    specificity = float(cm1[1, 1]) / (cm1[1, 0] + cm1[1, 1])
    precision = float(cm1[0, 0]) / (cm1[0, 0] + cm1[1, 0])
    f = 2 * ((precision * sensitivity) / (precision + sensitivity))
    f = float(cm1[0, 0]) / (cm1[0, 0] + cm1[1, 0])
    return accuracy,sensitivity,precision,f,specificity


##choosing a  patch in different directions
def getRandomPatch(i,j,k,d, f_FW,f_tumor,f_edema):
    lr_patch=int(patch_size/2)
    f_edema_out=np.ones(np.shape(f_edema))-f_edema
    if (d==0):
    
        Patch=f_FW[i,j-lr_patch:j+lr_patch, k-lr_patch:k+lr_patch]
        Patch2=f_tumor[i,j-lr_patch:j+lr_patch, k-lr_patch:k+lr_patch] 
        Patch3=f_edema_out[i,j-lr_patch:j+lr_patch, k-lr_patch:k+lr_patch]    
    if (d==1):
        Patch = f_FW[i-lr_patch:i+lr_patch, j, k-lr_patch:k+lr_patch]
        Patch2 = f_tumor[i-lr_patch:i+lr_patch, j, k-lr_patch:k+lr_patch]  
        Patch3=f_edema_out[i-lr_patch:i+lr_patch, j, k-lr_patch:k+lr_patch]      
    if (d==2):
        Patch = f_FW[i-lr_patch:i+lr_patch, j-lr_patch:j+lr_patch, k]
        Patch2 = f_tumor[i-lr_patch:i+lr_patch, j, k-lr_patch:k+lr_patch]
        Patch3 = f_edema_out[i-lr_patch:i+lr_patch, j, k-lr_patch:k+lr_patch]
         
    ##check for overlaps with tumor core or healthy brain 
    if (np.shape(np.where(Patch2.flatten()==True))[1]>0):
        return [-1]   
    if (np.shape(np.where(Patch3.flatten()==True))[1]>(0.2*(patch_size**2))):
        return [-1]
    return Patch



#################################################
##reading list of subjects, should have a column of Ids and a column for diagnosis(either "Met" or "GBM")
patch_size=16
L=['Met','GBM']
with pd.ExcelFile('TrainSamples/list.xlsx') as xls:
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
       # print(df.head())        
labels=df['Diagnose']  
FName=df['Id']

## making a folder for patches
if(not os.path.exists('TrainSamples/Patches')):
  os.mkdir('TrainSamples/Patches')
  os.mkdir('TrainSamples/Patches/Met/')
  os.mkdir('TrainSamples/Patches/GBM/')

##generating training patches
print('Extracting patches')
for root, dirs, files in os.walk('TrainSamples'): 
  if root.count(os.sep) < 1: 
    for i in range (len(dirs)):     
        f=dirs[i]
        if (f=='Patches'):
          break
        print(f)
        f_edema,affine=load_nifti(os.path.join('TrainSamples',f,'edema.nii.gz'))
        f_FW,affine=load_nifti(os.path.join('TrainSamples',f,'fw_volume_fraction.nii.gz'))
        f_tumor, affine = load_nifti(os.path.join('TrainSamples',f,'tumor.nii.gz'))
        [l1 ,l2 ,l3]=np.shape(f_edema)
        A=np.array(list(range(l1*l2*l3)))
        A2=np.reshape(A,[l1,l2,l3])        
        Edema_Mask_f=np.reshape(f_edema,[l1*l2*l3])
        Mask_Indexes,=np.where(Edema_Mask_f==True)
        patch_number=int(len(Mask_Indexes)/(patch_size))
        for R in random.sample(list(Mask_Indexes), patch_number):
            d = random.randint(0, 2)
            p1, p2,p3 = np.where(A2 == R)
            I=getRandomPatch(p1[0],p2[0],p3[0],d,f_FW,f_tumor,f_edema)           
            if (np.asarray(I).size<256):
                continue               
            ITemp0=I * 255
            im = Image.fromarray(I*255)           
            if im.mode != 'RGB':
                im = im.convert('RGB')
            cl=L.index(labels[(FName.tolist().index(f))])   
            if(cl==0):
                 im.save(str('TrainSamples/Patches/Met/' + f + '_pacth_' + str(p1[0]) + '_' + str(p2[0]) + '_' + str(p3[0]) + '_' + str(d) + '.jpg'))            
            elif(cl==1):
                 im.save(str('TrainSamples/Patches/GBM/' + f + '_pacth_' + str(p1[0]) + '_' + str(p2[0]) + '_' + str(p3[0]) + '_' + str(d) + '.jpg'))  
 ##########################
l1=patch_size
l2=patch_size
l3=3
nClasses=2
#####################
def createModel():
    model = Sequential()  
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(l1,l2,l3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
   # model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4,momentum=0.9, decay=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
###################################################
b_s = 32
train_dir='TrainSamples/Patches/'
print('Training CNN')
nTrain = sum([len(files) for r, d, files in os.walk(train_dir)])
train_features = np.zeros(shape=(nTrain, l1, l2, l3))
train_labels = np.zeros(shape=(nTrain, 2))

datagen = keras.preprocessing.image.ImageDataGenerator( rescale=1.0/255, featurewise_center=True,featurewise_std_normalization=True,    )
train_generator = datagen.flow_from_directory(train_dir,target_size=(l1, l2),batch_size=b_s,class_mode='categorical')
file_name=train_generator.filenames

i=0
for inputs_batch, labels_batch in train_generator: 
    features_batch=inputs_batch
    train_features[i * b_s: (i + 1) * b_s] = features_batch
    train_labels[i * b_s: (i + 1) * b_s] = labels_batch
    i += 1
    #print (i * b_s)
    if i * b_s >= nTrain:
        break
    #print (np.shape(file_name))

kf = KFold( n_splits=5, shuffle=True, random_state=100)
Acc=[]
TrainAc=[]
GroundY=-np.ones([nTrain,1])
bestY=-np.ones([nTrain,1])
bestYprob=-np.ones([nTrain,1])
fn=0
epochs = 100

for train, test in kf.split(train_features):
    model = createModel()
    fn=fn+1
    checkpoint = ModelCheckpoint('Models/best_model_FoldNo'+str(fn)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)   
    X_train, X_test, y_train, y_test =train_features[train], train_features[test], train_labels[train], train_labels[test]
    history = model.fit(X_train, y_train,epochs=epochs,batch_size=b_s,validation_data=(X_test,y_test),callbacks=[checkpoint])   
    Acc.append(history.history['val_accuracy'][epochs-1])
    TrainAc.append(history.history['accuracy'][epochs-1])   
    model=load_model('Models/best_model_FoldNo'+str(fn)+'.h5') 
    predict_x=model.predict(X_test) 
    predictions=np.argmax(predict_x,axis=1)
    prob = model.predict(X_test)
    print(classification_report(y_test[:,1], predictions))
    bestY[test,0]=predictions
    bestYprob[test,0]=prob[:,1]
    GroundY[test,0]=y_test[:,1]
    accuracy, recall, precision, f1, specificity = creport(y_test[:,1], predictions)

   
    
    
    