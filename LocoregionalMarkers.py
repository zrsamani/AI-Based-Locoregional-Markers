import os
import numpy as np
from scipy.ndimage.morphology import binary_erosion
import pandas as pd
import csv
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_erosion
import scipy.ndimage
import networkx as nx
from scipy.spatial.ckdtree import cKDTree
from dipy.io.image import load_nifti,save_nifti
from scipy.spatial import distance
from scipy.spatial.distance import pdist
Thre=0.9
kdtr=1

CF2 = open('Results/LocoregionalMarkers.csv', "w", newline='')
Cwriter = csv.writer(CF2, delimiter=',')

headerList = ['Id', 'Num','Size','Shape Heterogeneity','Directional Heterogeneity', 'Spatial Heterogeneity']
Cwriter.writerow(headerList)

for root, dirs, files in os.walk('Results'):
  if root.count(os.sep) < 1:
   for i in range (len(dirs)):  
            #Reading data            
            f=dirs[i]          
            try: 
              GBM_Mask_r, affine =load_nifti( 'Results/' +f+'/' 'PMI.nii.gz')
              f_tumor, affine = load_nifti( 'TestSamples/'  +f+'/' +'tumor.nii.gz')
              f_edema, affine = load_nifti('TestSamples/'  +f+'/' +'edema.nii.gz')
            except:
              print(f+' mask not found\n')
              continue          
            Edema_size=np.count_nonzero(f_edema)         
            GBM_Mask=np.zeros(np.shape(GBM_Mask_r))         
            GBM_Mask[f_edema == 1] =GBM_Mask_r[f_edema == 1]
            ####making connected components
            cs = np.argwhere(GBM_Mask > Thre)           
           # build k-d tree
            kdt = cKDTree(cs)
            edges = kdt.query_pairs(kdtr)
            # create graph
            G = nx.from_edgelist(edges)          
            # find connected components of PMI as locoregional hubs
            ccs = nx.connected_components(G)
            node_component = {v: k for k, vs in enumerate(ccs) for v in vs}
            # make mask and visualize
            df = pd.DataFrame(cs, columns=['x', 'y', 'z'])
            df['c'] = pd.Series(node_component)
            u=list(df['c'])
            GBM_Mask2=-1*np.ones(np.shape(GBM_Mask))
            GBM_Mask2[df['x'], df['y'], df['z']] = df['c']
           # save_nifti(  'Results/' + f + '/' + f +'CC.ni.gz', GBM_Mask2,affine)
            ### 
            ###################################################
            #computing size and number of hubs and shape heterogeneity
            ccs = nx.connected_components(G)
            number_of_hubs=0         
            shape_feature, size_feature, spatial_feature, hubs_sd = [],[],[],[]            
            for k, vs in enumerate(ccs):#k is the number of CC and vs is its elements               
                 size_feature.append(len(vs))
                 number_of_hubs=number_of_hubs+1                
                 xx, yy, zz = np.where(GBM_Mask2 == k) 
                 sdvec=[np.std(xx),np.std(yy),np.std(zz)]       
                   
                 hubs_sd.append(sdvec)
                 sdvec_sorted_ind = sorted(range(3), key=lambda k: sdvec[k], reverse=True)
                 shape_feature.append ((sdvec [sdvec_sorted_ind[0]]- sdvec[sdvec_sorted_ind [1]])/ sdvec[sdvec_sorted_ind [0]]) 
                 
                 xx_mean, yy_mean, zz_mean = np.mean(xx), np.mean(yy), np.mean(zz)
                 spatial_feature.append([xx_mean, yy_mean, zz_mean])                
               
                 
            size_of_hubs=np.mean(size_feature)
            shape_heterogeneity=np.mean (shape_feature)
            #############################################################
            #diretional heterogeneity
            min_Hausdorff =[]       
            for j in range (number_of_hubs-1):
               pair_cos_dis=[]
               for k in range (j+1,number_of_hubs):
                   pair_cos_dis.append (distance.cosine(np.asarray(hubs_sd[j]),np.asarray(hubs_sd[k])))                               
               min_Hausdorff.append(np.min( pair_cos_dis))
            directional_heterogeneity=np.max(min_Hausdorff)
            ################################################################
            #spatial Heterogeneity
            x_edema, y_edema, z_edema = np.where(f_edema == 1)
            Ed_cord =  [ x_edema, y_edema, z_edema]
            Diss=pdist(np.asarray(Ed_cord).transpose())  
            Ed_diam=Diss.max()            
            pair_euclidean_dis=[]
            for i in range (number_of_hubs):
               for j in range (i):
                   pair_euclidean_dis.append(np.linalg.norm((np.asarray(spatial_feature[i]) - np.asarray(spatial_feature[j])) / Ed_diam))              
            spatial_heterogeneity=np.mean(pair_euclidean_dis)
            
            Cwriter.writerow([f,number_of_hubs/Edema_size,size_of_hubs/Edema_size,shape_heterogeneity,  directional_heterogeneity,spatial_heterogeneity])
           

CF2.flush()
CF2.close()

