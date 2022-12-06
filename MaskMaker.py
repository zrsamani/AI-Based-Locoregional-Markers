
import csv
import numpy as np

import os
import pandas as pd
import argparse
from dipy.io.image import load_nifti, save_nifti
nof=5
##majority voting and generating masks
for root, dirs, files in os.walk('Results'):   
  if root.count(os.sep) < 1:
    for i in range (len(dirs)):  
    
        outcomef='Results/'+dirs[i]+'/Probabilities.csv'

        N0,L0,NC,N0En=[],[],[],[]
        f=0
        L=0
        try:
          reader = csv.reader(open(outcomef, "r"), delimiter=',')
        except:
          print (outcomef)
        
          continue
        vn=0
        for rows in reader:
            Sample_Name = rows[0][rows[0].rindex('/')+1:rows[0].rindex('_')]
            NC.append(rows[0])
            OS=Sample_Name
            Flag=0
            try:
                N0.index(Sample_Name)
                Flag = 1
            except ValueError:
                N0.append(Sample_Name)
                if (f == 1):
                    L0.append(float(L) /( vn*nof))
                f = 1
                L = 0;
                vn=0
            U=rows[1]
            L=L+float(rows[1])
            vn=vn+1
        L0.append(float(L) / (vn*nof))
        N1= list(set(N0))
        L1=L0
        f_edema, affine = load_nifti('TestSamples/'+dirs[i]+'/tumor.nii.gz')
        GBM_Mask = np.zeros(np.shape(f_edema))
        for n1 in N1:
        
          indices = [i for i, xx in enumerate(N1) if xx == n1]
          for ind in indices:
            nc = N1[ind]
            Slahses = [i for i, ltr in enumerate(nc) if ltr == '_']
            x = int(nc[nc.index('=') + 1:Slahses[0]])
            y = int(nc[Slahses[0] + 1:Slahses[1]])
            z = int(nc[Slahses[1] + 1:len(nc)])
            GBM_Mask[x, y, z] = float(L1[ind])      
        #os.mkdir('Results/'  +dirs[i])
        save_nifti('Results/'  +dirs[i]+'/' +'PMI.nii.gz',GBM_Mask,affine)
     
