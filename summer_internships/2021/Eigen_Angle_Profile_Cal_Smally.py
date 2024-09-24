import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import illustris_python as il
import pdb
import h5py
import glob
import math
import illustris_python.snapshot as snapp
from numpy import linalg as LA
import os
import pandas as pd    # Pandas library to find the shape!!
from scipy import stats
from scipy.stats import norm
import sys
from scipy.optimize import fsolve
import random
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-galN', action='store', type=int, choices=range(0, 25))

args = my_parser.parse_args()

"""
Code: August 28: 2020
Goal:

"""

Halo_ID = np.genfromtxt('./SubHalo_disky_index.txt')
Halo_ID.astype(int)


#colors = [cm.Reds(x) for x in evenly_spaced_interval]


os.chdir("/n/holyscratch01/loeb_lab/remamimeibody/illustris_python/TNG50/")


virialR = np.genfromtxt('./virialR50/virialR_Gal.txt')


## The indices  below refers to the cases with a real disk which has epsilon > 0.7 as well as the fraction of bigger than 40% of the stars in the disk.
Real_DISK_Gal = np.array([ 2,  4,  5,  6,  7, 10, 15, 17, 20, 21, 23, 24, 27, 29, 31, 34, 35, 39, 48, 52, 54, 57, 58, 63, 65])


### To guide us here the way that we associated them with each other:
   
#alignL = np.array([np.arccos(vvp0 @ ltot_hat)*180/np.pi, np.arccos(vvp1 @ ltot_hat)*180/np.pi, np.arccos(vvp2 @ ltot_hat)*180/np.pi])
# E_Vec00 = np.stack([rsph, vvp0[0], vvp0[1], vvp0[2]], axis=-1)
# E_Vec11 = np.stack([rsph, vvp1[0], vvp1[1], vvp1[2]], axis=-1)
# E_Vec22 = np.stack([rsph, vvp2[0], vvp2[1], vvp2[2]], axis=-1)
   

Dirr='/n/home00/remamimeibody/Gaia_Project/illustris_python/Stellar_Halo/TNG50/Approach3/Parallel_Computation/Shell_Based_Ananlysis/NEW_Approach/Stellar_Halo/Enclosed_Volume_Fixed/Weighting_Factor_r2/Shell_epsilon_whole/Plots/eigen_angle_cal2_NEW/'

for uu in range(25):

   Angles_ijk_min  = []
   Angles_ijk_inter  = []
   Angles_ijk_max  = []
   
   Vector_align_X = []
   Vector_align_Y = []
   Vector_align_Z = []
   
   
   Eigen_value0p = []
   Eigen_value1p = []
   Eigen_value2p = []
   Eigen_valueMIM = []
   
   
   Min_Eigen_Vector = []
   Inter_Eigen_Vector = []
   Max_Eigen_Vector = []
   
   
   max_ii = []
   eigens = []
   
   mm = Real_DISK_Gal[uu]
   Ltot = np.genfromtxt('/n/home00/remamimeibody/Gaia_Project/illustris_python/Data_Results_TNG50/DISK/LDisk_Eps_Less1/LTOT_DISK_Eps_Less1'+str(mm).zfill(3)+'.txt')
   
   ltot_hat = Ltot/(np.sqrt(Ltot[0]*Ltot[0] + Ltot[1]*Ltot[1] + Ltot[2]*Ltot[2]))
   
   EigenVector1 = np.genfromtxt('./Shape_SH_Shell_Enclosed_Volume_r2_Epsilon_Whole_NEW/Star_Approach3N/EigenMW-principal3/Eigen_prin_'+ str(mm).zfill(3)+'.txt')
   AngleVector1 = np.genfromtxt('./Shape_SH_Shell_Enclosed_Volume_r2_Epsilon_Whole_NEW/Star_Approach3N/align-principal3/align_prin_'+ str(mm).zfill(3)+'.txt')
   
   rrStar_Con = np.genfromtxt('./Shape_SH_Shell_Enclosed_Volume_r2_Epsilon_Whole_NEW/Star_Approach3N/shapeMW-principal3/shape_prin_'+str(mm).zfill(3)+'.txt')

   EigenVector_Reshape_00 = np.genfromtxt('./Shape_SH_Shell_Enclosed_Volume_r2_Epsilon_Whole_NEW/Star_Approach3N/EigenVecrots00-principal3/EigenVecrots00_'+ str(mm).zfill(3)+'.txt')
   
   EigenVector_Reshape_11 = np.genfromtxt('./Shape_SH_Shell_Enclosed_Volume_r2_Epsilon_Whole_NEW/Star_Approach3N/EigenVecrots11-principal3/EigenVecrots11_'+ str(mm).zfill(3)+'.txt')
   
   EigenVector_Reshape_22 = np.genfromtxt('./Shape_SH_Shell_Enclosed_Volume_r2_Epsilon_Whole_NEW/Star_Approach3N/EigenVecrots22-principal3/EigenVecrots22_'+ str(mm).zfill(3)+'.txt')
   
   #x_hat = np.array([1,0,0])
   #y_hat = np.array([0,1,0])
   #z_hat = np.array([0,0,1])
   
   """
   ## To reduce the degeneracy, we made some rotations!!
   thetax = 45*np.pi/180
   thetaz = 45*np.pi/180

   MMx = np.array([[1,0,0], [0,np.cos(thetax), -np.sin(thetax)], [0, np.sin(thetax), np.cos(thetax)]])
   MMz = np.array([[np.cos(thetaz), -np.sin(thetaz), 0], [np.sin(thetaz), np.cos(thetaz), 0], [0, 0, 1]])

   xx = np.array([1,0,0])
   yy = np.array([0,1,0])
   zz = np.array([0,0,1])

   x_hat = MMx @ MMz @ xx
   y_hat = MMx @ MMz @ yy
   z_hat = MMx @ MMz @ zz
   """
   
   x_hat = np.array([1,0,0])
   y_hat = np.array([0,1,0])
   z_hat = np.array([0,0,1])
   
   
   ### Below we compute the radial profile of the angles between different eigen_values of inertia tensor and the total angular momemntum. To get a clean pucture of what is going on, in the same plot we show both of the aligned and perpendicular angles. We repeat this for the largest, intermediate and smallest eigen-values.
   
   ## So here are the steps of the analysis:
   
   ## First, we shall sort out the eigen-values
   
   Eigen_valuess = EigenVector1[:,1:4]   ## First, we eliminate the zero index: The radii.

   #E_Vec00_3 = EigenVector_Reshape_00[:, 1:4]
   #E_Vec11_3 = EigenVector_Reshape_11[:, 1:4]
   #E_Vec22_3 = EigenVector_Reshape_22[:, 1:4]
   
   
   ## This is the angle at the  smallest radii.
   #Angles_ijk_00 = np.array([EigenVector1[0, 0], np.arccos(E_Vec00_3[0,:] @ ltot_hat)*180/np.pi, np.arccos(E_Vec00_3[0,:] @ x_hat)*180/np.pi, np.arccos(E_Vec00_3[0,:] @ y_hat)*180/np.pi,  np.arccos(E_Vec00_3[0,:] @ z_hat)*180/np.pi])
   
   
   #Angles_ijk_11 = np.array([EigenVector1[0, 0], np.arccos(E_Vec11_3[0,:] @ ltot_hat)*180/np.pi,np.arccos(E_Vec11_3[0,:] @ x_hat)*180/np.pi, np.arccos(E_Vec11_3[0,:] @ y_hat)*180/np.pi,  np.arccos(E_Vec11_3[0,:] @ z_hat)*180/np.pi])
   
   #Angles_ijk_22 = np.array([EigenVector1[0, 0], np.arccos(E_Vec22_3[0,:] @ ltot_hat)*180/np.pi, np.arccos(E_Vec22_3[0,:] @ x_hat)*180/np.pi, np.arccos(E_Vec22_3[0,:] @ y_hat)*180/np.pi,  np.arccos(E_Vec22_3[0,:] @ z_hat)*180/np.pi])

   E_Vec00_3 = EigenVector_Reshape_00[0, 1:4]
   E_Vec11_3 = EigenVector_Reshape_11[0, 1:4]
   E_Vec22_3 = EigenVector_Reshape_22[0, 1:4]
   
   Sorted_eigen_values = np.array([EigenVector1[0,1], EigenVector1[0,2], EigenVector1[0,3]])
   
   Sorted_Eigen_Vector = np.stack([E_Vec00_3, E_Vec11_3, E_Vec22_3])
   
   sorted_index = np.argsort(Sorted_eigen_values)
   
   min_ind = sorted_index[0]
   inter_ind = sorted_index[1]
   max_ind = sorted_index[2]
   
   
   Min_Eigen_Vector = Sorted_Eigen_Vector[min_ind]
   Inter_Eigen_Vector = Sorted_Eigen_Vector[inter_ind]
   Max_Eigen_Vector = Sorted_Eigen_Vector[max_ind]
   
   
   ## This is the angle at the  smallest radii.
   Angles_ijk_min = np.array([EigenVector1[0, 0], np.arccos(Sorted_Eigen_Vector[min_ind] @ ltot_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[min_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[min_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vector[min_ind] @ z_hat)*180/np.pi])
      
   Angles_ijk_inter = np.array([EigenVector1[0, 0], np.arccos(Sorted_Eigen_Vector[inter_ind] @ ltot_hat)*180/np.pi,np.arccos(Sorted_Eigen_Vector[inter_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[inter_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vector[inter_ind] @ z_hat)*180/np.pi])
   
   Angles_ijk_max = np.array([EigenVector1[0, 0], np.arccos(Sorted_Eigen_Vector[max_ind] @ ltot_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[max_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[max_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vector[max_ind] @ z_hat)*180/np.pi])
  
   """
   ## Firstly we propose that initial angle should be less than 90 degree:
   Angles_ijk_min_3 = np.array([np.arccos(Sorted_Eigen_Vector[min_ind] @ ltot_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[min_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[min_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vector[min_ind] @ z_hat)*180/np.pi])
      
   mask_angle_x = Angles_ijk_min_3 >90
   Angles_ijk_min_3[mask_angle_x] = 180 - Angles_ijk_min_3[mask_angle_x]
   #print(Angles_ijk_min_3)
   Angles_ijk_min = np.array([EigenVector1[0, 0], Angles_ijk_min_3[0], Angles_ijk_min_3[1], Angles_ijk_min_3[2], Angles_ijk_min_3[3]])
    
   Angles_ijk_inter_3 = np.array([np.arccos(Sorted_Eigen_Vector[inter_ind] @ ltot_hat)*180/np.pi,np.arccos(Sorted_Eigen_Vector[inter_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[inter_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vector[inter_ind] @ z_hat)*180/np.pi])
      
   mask_angle_X_Inter = Angles_ijk_inter_3 >90
   Angles_ijk_inter_3[mask_angle_X_Inter] = 180 - Angles_ijk_inter_3[mask_angle_X_Inter]
   
   Angles_ijk_inter = np.array([EigenVector1[0, 0], Angles_ijk_inter_3[0], Angles_ijk_inter_3[1], Angles_ijk_inter_3[2], Angles_ijk_inter_3[3]])
   
   
   Angles_ijk_max_3 = np.array([np.arccos(Sorted_Eigen_Vector[max_ind] @ ltot_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[max_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vector[max_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vector[max_ind] @ z_hat)*180/np.pi])
      
   mask_angle_X_Max = Angles_ijk_max_3 >90
   Angles_ijk_max_3[mask_angle_X_Max] = 180 - Angles_ijk_max_3[mask_angle_X_Max]
   
   Angles_ijk_max = np.array([EigenVector1[0, 0], Angles_ijk_max_3[0], Angles_ijk_max_3[1], Angles_ijk_max_3[2], Angles_ijk_max_3[3]])
   """
   
   Eigen_value0p = np.array([EigenVector1[0,0], Sorted_eigen_values[min_ind]])
   Eigen_value1p = np.array([EigenVector1[0,0], Sorted_eigen_values[inter_ind]])
   Eigen_value2p = np.array([EigenVector1[0,0], Sorted_eigen_values[max_ind]])
   Eigen_valueMIM = np.append(Eigen_valueMIM, [EigenVector1[0,0], Sorted_eigen_values[min_ind],Sorted_eigen_values[inter_ind], Sorted_eigen_values[max_ind]])
   
   ## This is the angle at the  smallest radii.
   #Angles_ijk_00 = np.array([EigenVector1[0, 0], np.arccos(E_Vec00_3[0,:] @ ltot_hat)*180/np.pi, np.arccos(E_Vec00_3[0,:] @ x_hat)*180/np.pi, np.arccos(E_Vec00_3[0,:] @ y_hat)*180/np.pi,  np.arccos(E_Vec00_3[0,:] @ z_hat)*180/np.pi])
   
   #Angles_ijk_11 = np.array([EigenVector1[0, 0], np.arccos(E_Vec11_3[0,:] @ ltot_hat)*180/np.pi,np.arccos(E_Vec11_3[0,:] @ x_hat)*180/np.pi, np.arccos(E_Vec11_3[0,:] @ y_hat)*180/np.pi,  np.arccos(E_Vec11_3[0,:] @ z_hat)*180/np.pi])
   
   #Angles_ijk_22 = np.array([EigenVector1[0, 0], np.arccos(E_Vec22_3[0,:] @ ltot_hat)*180/np.pi, np.arccos(E_Vec22_3[0,:] @ x_hat)*180/np.pi, np.arccos(E_Vec22_3[0,:] @ y_hat)*180/np.pi,  np.arccos(E_Vec22_3[0,:] @ z_hat)*180/np.pi])
   
   #pdb.set_trace()
   ## First construst the 3*3 matrix of the cosines between different vectors and then based on that we shall find the closest one (taking into account the absolute value) and apply the positivity as a calibration to it!!!
       
   for i in range(len(EigenVector_Reshape_00[:,0])-1):
       
       
       ## We should be careful that for any radii above r_min we should use the already assigned vectors instead of reading them off again from the original vectors!!!
       if i==0:
           Eigen0_cur  = EigenVector_Reshape_00[i,1:]
           Eigen0_next = EigenVector_Reshape_00[i+1,1:]
           
           Eigen1_cur  = EigenVector_Reshape_11[i,1:]
           Eigen1_next = EigenVector_Reshape_11[i+1,1:]
           
           Eigen2_cur  = EigenVector_Reshape_22[i,1:]
           Eigen2_next = EigenVector_Reshape_22[i+1,1:]
       
       
       if i >= 1:
           Eigen0_cur = Eigen_Vector_Arrays[v0_ind]
           Eigen0_next = EigenVector_Reshape_00[i+1,1:]
           
           Eigen1_cur = Eigen_Vector_Arrays[v1_ind]
           Eigen1_next = EigenVector_Reshape_11[i+1,1:]
           
           Eigen2_cur = Eigen_Vector_Arrays[v2_ind]
           Eigen2_next = EigenVector_Reshape_22[i+1,1:]
       
       
       
       cos00 = np.array([Eigen0_cur @ Eigen0_next, Eigen0_cur @ Eigen1_next, Eigen0_cur @ Eigen2_next])
       
       cos11 = np.array([Eigen1_cur @ Eigen0_next, Eigen1_cur @ Eigen1_next, Eigen1_cur @ Eigen2_next])
       
       cos22 = np.array([Eigen2_cur @ Eigen0_next, Eigen2_cur @ Eigen1_next, Eigen2_cur @ Eigen2_next])
       

       ## bsolute values of the angle to find the closest one!!!
       cos0_abs = np.array([abs(Eigen0_cur @ Eigen0_next), abs(Eigen0_cur @ Eigen1_next), abs(Eigen0_cur @ Eigen2_next)])
       
       cos1_abs = np.array([abs(Eigen1_cur @ Eigen0_next), abs(Eigen1_cur @ Eigen1_next), abs(Eigen1_cur @ Eigen2_next)])
       
       cos2_abs = np.array([abs(Eigen2_cur @ Eigen0_next), abs(Eigen2_cur @ Eigen1_next), abs(Eigen2_cur @ Eigen2_next)])
       
       
       ### We shall find the maximum value of the cosine of their inner product.
       v0_ind = np.argmax(cos0_abs)
       v1_ind = np.argmax(cos1_abs)
       v2_ind = np.argmax(cos2_abs)
       
       #print(v0_ind, v1_ind, v2_ind)
       
       if v0_ind == v1_ind or v0_ind == v2_ind or v1_ind == v2_ind:
       
          print("Degenerate:", uu, v0_ind, v1_ind, v2_ind)
       
       Eigen_Vector_Arrays = np.stack([EigenVector_Reshape_00[i+1,1:4], EigenVector_Reshape_11[i+1,1:4], EigenVector_Reshape_22[i+1,1:4]])
       
       ### We need to first associate the eigen-values to the deformed eigen-vectors and then sort them out!!! BE Careful here!!
       Eigen_value_Arrays = np.array([EigenVector1[i+1,1:4]])[0]
       
       #pdb.set_trace()
       Eigen_value_Arrays_Associated_EV = np.array([Eigen_value_Arrays[v0_ind], Eigen_value_Arrays[v1_ind], Eigen_value_Arrays[v2_ind]])
       
       #Eigen_value0pi = np.array([EigenVector1[i+1,0], Eigen_value_Arrays_Associated_EV[0]])
       #Eigen_value1pi = np.array([EigenVector1[i+1,0], Eigen_value_Arrays_Associated_EV[1]])
       #Eigen_value2pi = np.array([EigenVector1[i+1,0], Eigen_value_Arrays_Associated_EV[2]])
              
       
       #E_Vec00_3 = EigenVector_Reshape_00[i+1, 1:4]
       #E_Vec11_3 = EigenVector_Reshape_11[i+1, 1:4]
       #E_Vec22_3 = EigenVector_Reshape_22[i+1, 1:4]
              
       ## Now check the sign: Here we flip the vactor if this has a negative cosine!
       
       #pdb.set_trace()
       if cos00[v0_ind]<0:
          #print("Zero", cos00[v0_ind])
          Eigen_Vector_Arrays[v0_ind] = - Eigen_Vector_Arrays[v0_ind]
          #print(Eigen_Vector_Arrays[v0_ind])
       
       if cos11[v1_ind]<0:
           #print("One", cos11[v1_ind])
           Eigen_Vector_Arrays[v1_ind] = - Eigen_Vector_Arrays[v1_ind]
           #print(Eigen_Vector_Arrays[v1_ind])

       if cos22[v2_ind]<0:
          #print("Two", cos22[v2_ind])
          Eigen_Vector_Arrays[v2_ind] = - Eigen_Vector_Arrays[v2_ind]
          #print(Eigen_Vector_Arrays[v2_ind])

       
       Sorted_Eigen_Vectori = np.stack([Eigen_Vector_Arrays[v0_ind], Eigen_Vector_Arrays[v1_ind], Eigen_Vector_Arrays[v2_ind]])
       
       sorted_indexi = np.argsort(Eigen_value_Arrays_Associated_EV)
       
       min_indi = sorted_indexi[0]
       inter_indi = sorted_indexi[1]
       max_indi = sorted_indexi[2]
       
       Min_Eigen_Vector = np.vstack((Min_Eigen_Vector, Sorted_Eigen_Vectori[min_ind]))
       Inter_Eigen_Vector = np.vstack((Inter_Eigen_Vector, Sorted_Eigen_Vectori[inter_ind]))
       Max_Eigen_Vector = np.vstack((Max_Eigen_Vector, Sorted_Eigen_Vectori[max_ind]))
       
       ## This is the angle at the  smallest radii.
              
       Angles_ijk_min = np.append(Angles_ijk_min, [EigenVector1[0, 0], np.arccos(Sorted_Eigen_Vectori[min_ind] @ ltot_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vectori[min_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vectori[min_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vectori[min_ind] @ z_hat)*180/np.pi])
              
       Angles_ijk_inter = np.append(Angles_ijk_inter, [EigenVector1[0, 0], np.arccos(Sorted_Eigen_Vectori[inter_ind] @ ltot_hat)*180/np.pi,np.arccos(Sorted_Eigen_Vectori[inter_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vectori[inter_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vectori[inter_ind] @ z_hat)*180/np.pi])
       
       Angles_ijk_max = np.append(Angles_ijk_max, [EigenVector1[0, 0], np.arccos(Sorted_Eigen_Vectori[max_ind] @ ltot_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vectori[max_ind] @ x_hat)*180/np.pi, np.arccos(Sorted_Eigen_Vectori[max_ind] @ y_hat)*180/np.pi,  np.arccos(Sorted_Eigen_Vectori[max_ind] @ z_hat)*180/np.pi])
       
       Eigen_value0p = np.append(Eigen_value0p, [EigenVector1[i+1,0], Eigen_value_Arrays_Associated_EV[min_ind]])
       Eigen_value1p = np.append(Eigen_value1p, [EigenVector1[i+1, 0], Eigen_value_Arrays_Associated_EV[inter_ind]])
       Eigen_value2p = np.append(Eigen_value2p, [EigenVector1[i+1, 0], Eigen_value_Arrays_Associated_EV[max_ind]])
       Eigen_valueMIM = np.append(Eigen_valueMIM, [EigenVector1[i+1, 0], Eigen_value_Arrays_Associated_EV[min_ind],Eigen_value_Arrays_Associated_EV[inter_ind], Eigen_value_Arrays_Associated_EV[max_ind]])

       
   ## Next, we shall compute the angle between the total angular momentum and different eigen_vectors. And we should also compute angle between the perp. direction to L_tot and each of eigenvectors. WE use one of the eigen-vectors for that calibration. Since in almost all of the cases we consider one V_is is almost aligned with L_tot, we just need to look for the one that has larger angle and then compute their cross-product to begin with!
   
   #print("here  we have 00", Angles_ijk_00)
   #print("here  we have 11", Angles_ijk_11)

   
   len55 = int(len(Angles_ijk_min)/5.0)
   Vector_align_00 = Angles_ijk_min.reshape(len55, 5)
   Vector3D_00 = Vector_align_00[:, 1:5]     ## We eliminate the radius from here!!!
   
   ## Here we put the criteria of smaller than 90 degree on the initial point.
   mask0 = Vector3D_00[0,:] > 90
   for i in range(4):
       if mask0[i] == True:
          Vector3D_00[:,i] = 180 - Vector3D_00[:,i]
   
   
   len55 = int(len(Angles_ijk_inter)/5.0)
   Vector_align_11 = Angles_ijk_inter.reshape(len55, 5)
   Vector3D_11 = Vector_align_11[:, 1:5]     ## We eliminate the radius from here!!!
   
   ## Here we put the criteria of smaller than 90 degree on the initial point.
   mask1 = Vector3D_11[0,:] > 90
   for i in range(4):
       if mask1[i] == True:
          Vector3D_11[:,i] = 180 - Vector3D_11[:,i]
   
   len55 = int(len(Angles_ijk_max)/5.0)
   Vector_align_22 = Angles_ijk_max.reshape(len55, 5)
   Vector3D_22 = Vector_align_22[:, 1:5]     ## We eliminate the radius from here!!!
   
   ## Here we put the criteria of smaller than 90 degree on the initial point.
   mask2 = Vector3D_22[0,:] > 90
   for i in range(4):
       if mask2[i] == True:
          Vector3D_22[:,i] = 180 - Vector3D_22[:,i]
      
   len44 = int(len(Eigen_valueMIM)/4.0)
   vec_eigen_sort = Eigen_valueMIM.reshape(len44,4)
   
   path1 = '/n/home00/remamimeibody/Gaia_Project/illustris_python/Stellar_Halo/TNG50/Approach3/Parallel_Computation/Shell_Based_Ananlysis/NEW_Approach/Stellar_Halo/Enclosed_Volume_Fixed/Weighting_Factor_r2/Shell_epsilon_whole/data_angle'
   
   np.savetxt(path1 + '/Vector_Angle00_'+ str(uu).zfill(3) + '.txt', Vector_align_00)
   
   np.savetxt(path1 + '/Vector_Angle11_'+ str(uu).zfill(3) + '.txt', Vector_align_11)
   
   np.savetxt(path1 + '/Vector_Angle22_'+ str(uu).zfill(3) + '.txt', Vector_align_22)
   
   
   np.savetxt(path1 + '/Eigen_Vector_Angle_Min_'+ str(uu).zfill(3) + '.txt', Min_Eigen_Vector)
   
   np.savetxt(path1 + '/Eigen_Vector_Angle_Inter_'+ str(uu).zfill(3) + '.txt', Inter_Eigen_Vector)
   
   np.savetxt(path1 + '/Eigen_Vector_Angle_Max_'+ str(uu).zfill(3) + '.txt', Max_Eigen_Vector)
   
   np.savetxt(path1 + '/Sorted_Eigen_Value_Profile_'+ str(uu).zfill(3) + '.txt', vec_eigen_sort)
   
   
   print("Sorted EigenValues:", vec_eigen_sort)
   fig, big_axes = plt.subplots(figsize=(16.4, 3.5), nrows=1, ncols=1)

   plt.subplots_adjust(top =2.9, bottom=0.3, hspace=0.4, wspace=0.6)
   #for row, big_ax in enumerate(big_axes, start=0):
   #big_axes.set_title(r'Halo ID:' + str(int(Halo_ID[1])), fontsize=20, color = 'purple', x=0.5, y= 1.02)

   big_axes.get_xaxis().set_ticks([])
   big_axes.get_yaxis().set_ticks([])
   big_axes._frameon = False
   
   ax = fig.add_subplot(1, 5, 1)

   rr = EigenVector1[:, 0]
   a = EigenVector1[:, 1]
   b = EigenVector1[:, 2]
   c = EigenVector1[:, 3]
   
   ax.plot(rr, a/rr, color = 'teal', label = r'$a_F$' , linewidth=0, alpha=1, marker = 'o', markersize = 3)
   ax.plot(rr, b/rr, color = 'teal', label = r'$b_F$' , linewidth=0, alpha=1, marker = 'o', markersize = 3)
   ax.plot(rr, c/rr, color = 'teal', label = r'$c_F$' , linewidth=0, alpha=1, marker = 'o', markersize = 3)
   axes = plt.gca()
   plt.xscale('log')
   #plt.yscale('log')
   plt.tick_params(axis='x', which='major', labelsize= 9)
   plt.tick_params(axis='y', which='major', labelsize= 9)
   #legend = plt.legend(loc=4, borderaxespad=1., ncol=1,numpoints=1,fontsize = 14,fancybox=True)
   
   plt.tight_layout()
   ax.set_xlabel(r'r(kpc)', fontsize = 12)
   ax.set_ylabel(r'Axes/r', fontsize = 12)
   axes.set_xlim([1, 100])
   plt.grid(True)
   
   ax = fig.add_subplot(1, 5, 2)

   ax.plot(EigenVector1[:,0], Vector3D_00[:,0], color = 'blue', label = r'$L^{\parallel}_{\rm{tot}}$' , linewidth=1, alpha=1, marker = 'o', markersize = 3)
   
   ax.plot(EigenVector1[:,0], Vector3D_00[:,1], color = 'crimson', label = r'$\hat{i}$' , linewidth=1, alpha=1, marker = 's', markersize = 3)
   
   ax.plot(EigenVector1[:,0], Vector3D_00[:,2], color = 'darkgreen', label = r'$\hat{j}$' , linewidth=1, alpha=1, marker = 'v', markersize = 3)
       
   
   ax.plot(EigenVector1[:,0], Vector3D_00[:,3], color = 'gold', label = r'$\hat{k}$' , linewidth=1, alpha=1, marker = 'h', markersize = 3)
       
   axes = plt.gca()
   plt.xscale('log')
   plt.tick_params(axis='x', which='major', labelsize=9, length = 4)
   plt.tick_params(axis='y', which='major', labelsize=9, length = 4)
   legend = ax.legend(loc=9, borderaxespad=1., ncol=2, numpoints=1, fontsize = 11, fancybox=True)
   plt.tight_layout()
   ax.set_xlabel(r'r(kpc)', fontsize = 12)
   ax.set_ylabel(r'$\Theta_{\rm{min}}(\rm{deg})$', fontsize = 12)
   axes.set_ylim([-2, 220])
   axes.set_xlim([1, 100])
   plt.grid(True)


   ax = fig.add_subplot(1,5,3)
   ax.set_title(r'Halo ID:' + str(int(Halo_ID[uu])), fontsize=13, color = 'purple')
   
   ax.plot(EigenVector1[:,0], Vector3D_11[:,0], color = 'blue', label = r'$L^{\parallel}_{\rm{tot}}$' , linewidth=1, alpha=1, marker = 'o', markersize = 3)
   
   ax.plot(EigenVector1[:,0], Vector3D_11[:,1], color = 'crimson', label = r'$\hat{i}$' , linewidth=1, alpha=1, marker = 's', markersize = 3)
   
   ax.plot(EigenVector1[:,0], Vector3D_11[:,2], color = 'darkgreen', label = r'$\hat{j}$' , linewidth=1, alpha=1, marker = 'v', markersize = 3)
       
   
   ax.plot(EigenVector1[:,0], Vector3D_11[:,3], color = 'gold', label = r'$\hat{k}$' , linewidth=1, alpha=1, marker = 'h', markersize = 3)

   axes = plt.gca()
   plt.xscale('log')
   plt.tick_params(axis='x', which='major', labelsize=9, length = 4)
   plt.tick_params(axis='y', which='major', labelsize=9, length = 4)
   legend = ax.legend(loc=9, borderaxespad=1., ncol=2, numpoints=1, fontsize = 11, fancybox=True)
   plt.tight_layout()
   ax.set_xlabel(r'r(kpc)', fontsize = 12)
   ax.set_ylabel(r'$\Theta_{\rm{inter}}(\rm{deg})$', fontsize = 12)
   axes.set_ylim([-2, 220])
   axes.set_xlim([1, 100])
   plt.grid(True)

   
   ax = fig.add_subplot(1,5,4)
   
   ax.plot(EigenVector1[:,0], Vector3D_22[:,0], color = 'blue', label = r'$L^{\parallel}_{\rm{tot}}$' , linewidth=1, alpha=1, marker = 'o', markersize = 3)
   
   ax.plot(EigenVector1[:,0], Vector3D_22[:,1], color = 'crimson', label = r'$\hat{i}$' , linewidth=1, alpha=1, marker = 's', markersize = 3)
   
   ax.plot(EigenVector1[:,0], Vector3D_22[:,2], color = 'darkgreen', label = r'$\hat{j}$' , linewidth=1, alpha=1, marker = 'v', markersize = 3)
       
   
   ax.plot(EigenVector1[:,0], Vector3D_22[:,3], color = 'gold', label = r'$\hat{k}$' , linewidth=1, alpha=1, marker = 'h', markersize = 3)
   
   axes = plt.gca()
   plt.xscale('log')
   plt.tick_params(axis='x', which='major', labelsize=9, length = 4)
   plt.tick_params(axis='y', which='major', labelsize=9, length = 4)
   legend = ax.legend(loc=9, borderaxespad=1., ncol=2, numpoints=1, fontsize = 11, fancybox=True)
   plt.tight_layout()
   ax.set_xlabel(r'r(kpc)', fontsize = 12)
   ax.set_ylabel(r'$\Theta_{\rm{max}}(\rm{deg})$', fontsize = 12)
   axes.set_ylim([-2, 220])
   axes.set_xlim([1, 100])
   plt.grid(True)


   ax = fig.add_subplot(1,5,5)
   
   
   plt.plot(rrStar_Con[:,0], rrStar_Con[:,1], color = 'navy', label = r'$s$', marker = 'o',linewidth = 2, alpha = 0.8, markersize = 3, linestyle = '-')
   
   plt.plot(rrStar_Con[:,0], rrStar_Con[:,2], color = 'darkred', label = r'$q$', marker = 's',linewidth = 2, alpha = 0.8, markersize = 3, linestyle = '-')

   plt.tight_layout(True)
   ax.set_xlabel(r'r(kpc)', fontsize = 12)
   ax.set_ylabel(r'shape', fontsize = 12)
            
   axes = plt.gca()
    
   axes.set_xlim([1, 100])
   axes.set_ylim([0.0, 1.06])
   plt.tick_params(axis='x', which='major', labelsize=9)
   plt.tick_params(axis='y', which='major', labelsize=9)
    
   plt.xscale('log')
   legend = plt.legend(loc=3, borderaxespad=1., ncol=1,numpoints=1,fontsize = 11,fancybox=True)
   plt.grid(True)
   

   fig.set_facecolor('w')
   plt.tight_layout()
   fig.subplots_adjust(top=0.88)
   
   plt.savefig(Dirr + 'Eigen_AngleN_whole_'+ str(uu).zfill(3) +'.pdf',dpi = 400, transparent = True, bbox_inches='tight')
   
