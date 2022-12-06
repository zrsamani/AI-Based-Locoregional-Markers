# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from random import shuffle
import threading
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.models import load_model
import os
import glob
import argparse
import csv
import glob
import os
from dipy.io.image import load_nifti, save_nifti
from os.path import isfile, join
import shutil
import numpy as np
import random
from PIL import Image
from scipy.ndimage.morphology import binary_erosion
import pandas as pd
from scipy.ndimage.morphology import binary_dilation

##generaing patches
def getRandomPatch(i,j,k,d, f_FW):
    lr_patch=int(patch_size/2)
    if (d==0):
        Patch=f_FW[i,j-lr_patch:j+lr_patch, k-lr_patch:k+lr_patch]
    if (d==1):
        Patch = f_FW[i-lr_patch:i+lr_patch, j, k-lr_patch:k+lr_patch]
    if (d==2):
        Patch = f_FW[i-lr_patch:i+lr_patch, j-lr_patch:j+lr_patch, k]
    return Patch

patch_size=16
for root, dirs, files in os.walk('TestSamples'):
  if root.count(os.sep) < 1:
    for i in range (len(dirs)):   
        f=dirs[i]  
        print('getting the results for subject', f)          
        f_edema,affine=load_nifti(os.path.join('TestSamples',f,'edema.nii.gz'))
        f_FW,affine=load_nifti(os.path.join('TestSamples',f,'fw_volume_fraction.nii.gz'))
        [l1 ,l2 ,l3]=np.shape(f_edema)
        A=np.array(list(range(l1*l2*l3)))
        A2=np.reshape(A,[l1,l2,l3])       
        Edema_Mask_f=np.reshape(f_edema,[l1*l2*l3])
        Mask_Indexes,=np.where(Edema_Mask_f==True)       
        if (not os.path.exists('TestSamples/' + f+'/Patches')):
            os.mkdir(str(  'TestSamples/' + f+'/Patches' ))
            os.mkdir(str(  'TestSamples/' + f+'/Patches/GBM1' ))
            os.mkdir(str(  'TestSamples/' + f+'/Patches/GBM2' ))        
        din=1
        for R in Mask_Indexes:
            if (din==1):
              din=din+1
              sp='GBM1/'
            else:
              sp='GBM2/'                
           # d = random.randint(0, 2)
            p1, p2,p3 = np.where(A2 == R)
            I=getRandomPatch(p1[0],p2[0],p3[0],0,f_FW)
            if (I.size==0):
              continue        
            im = Image.fromarray(I*255)           
            if im.mode != 'RGB':
                im = im.convert('RGB')
            #Classes
            im.save(str('TestSamples' +'/' + f +  '/Patches/'+sp + f + 'pacth=' + str(p1[0]) + '_' + str(p2[0]) + '_' + str(p3[0]) + '_' + str(0) + '.jpg'))            
            I=getRandomPatch(p1[0],p2[0],p3[0],1,f_FW)
            if (I.size==0):
              continue
            im = Image.fromarray(I*255)            
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(str('TestSamples' +'/' + f +  '/Patches/'+sp + f + 'pacth=' + str(p1[0]) + '_' + str(p2[0]) + '_' + str(p3[0]) + '_' + str(1) + '.jpg'))
            I=getRandomPatch(p1[0],p2[0],p3[0],2,f_FW)
            if (I.size==0):
              continue
            im = Image.fromarray(I*255)            
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(str('TestSamples' +'/' + f +  '/Patches/'+sp + f + 'pacth=' + str(p1[0]) + '_' + str(p2[0]) + '_' + str(p3[0]) + '_' + str(2) + '.jpg'))
            
model_dir='Models/'
datagen = keras.preprocessing.image.ImageDataGenerator( rescale=1.0/255, featurewise_center=True, featurewise_std_normalization=True
              )
patch_size=8
nClasses=2
l1=patch_size*2
l2=patch_size*2
l3=3

for root, dirs, files in os.walk('TestSamples'):
   if root.count(os.sep) < 1:
     for idir in range (len(dirs)):  
      subId=dirs[idir]
      Res_dir=str('Results/'+subId)
      val_dir=str('TestSamples/'+subId+'/Patches')
      if not(os.path.exists(Res_dir)):
        os.mkdir(Res_dir)
      nval = sum([len(files) for r, d, files in os.walk(val_dir)])   
      val_features = np.zeros(shape=(nval, l1, l2, l3))
      val_labels = np.zeros(shape=(nval, 2))
      batch_size=32               
      val_generator = datagen.flow_from_directory(val_dir,target_size=(l1, l2),batch_size=batch_size, class_mode='categorical',shuffle=False)
    
      file_name=val_generator.filenames
      i=0
      for inputs_batch, labels_batch in val_generator:
             features_batch =inputs_batch
             val_features[i * batch_size: (i + 1) * batch_size] = features_batch
             val_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
             i += 1
             if i * batch_size >= nval:
                  break
      GroundY=-np.ones([nval,1])
      bestY=np.zeros([nval,1])
      bestYprob=np.zeros([nval,2])
      i=0;
      nfo=5
      for i in range (nfo):        
            model=load_model(model_dir+'best_model_FoldNo'+str(i+1)+'.h5')
            model=load_model(model_dir+'New-CNN-patches-8-'+str(i)+'.h5')
            predict_x=model.predict(val_features) 
            predictions=np.argmax(predict_x,axis=1)
            prob = model.predict(val_features)
            bestY=np.reshape(predictions,[nval,1])+bestY   
            bestYprob[:,0]=bestYprob[:,0]+prob[:,0]
            bestYprob[:,1]=bestYprob[:,1]+prob[:,1]
            GroundY=val_labels[:,1]
      CF = open(Res_dir+"/Probabilities.csv", "w")
      Cwriter = csv.writer(CF, delimiter=',')
      for i in range(len(bestY)):
            Cwriter.writerow([file_name[i],float(bestYprob[i,0]),float(bestYprob[i,1])])        
      CF.flush()
      CF.close()
    
    
    
    
