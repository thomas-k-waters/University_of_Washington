import matplotlib
matplotlib.use('Agg')  # Add this when you want to make a plot in test part of the cluster runs!
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
import csv
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import matplotlib.mlab as mlab
import h5py
import scipy.fftpack
from scipy import interpolate
import matplotlib.font_manager as fm
import pylab as py
import scipy.special as sp
import os
import pdb
from matplotlib.pyplot import cm
from numpy import linalg as LA
import os
import pandas as pd    # Pandas library to find the shape!!
from scipy import stats
from scipy.stats import norm
import sys
#import jscatter as js
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d


gal_identifier = ['476266', '478216', '479938', '480802', '485056', '488530', '494709', '497557', '501208', '501725', '502995', '503437', '505586', '506720', '509091', '510585', '511303', '513845', '519311', '522983', '523889', '529365', '530330', '535410', '538905']


x = []
y = []

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


"""
Code: Oct. 10
Goal: In this script, we shall compute the radial profile of the median and percentiles of the shape for the stellar halo.
"""



#os.chdir("/n/holyscratch01/loeb_lab/remamimeibody/illustris_python/TNG50/")
Real_DISK_Gal = np.array([ 2,  4,  5,  6,  7, 10, 15, 17, 20, 21, 23, 24, 27, 29, 31, 34, 35, 39, 48, 52, 54, 57, 58, 63, 65])
## Here is the dictionary of the shape files, (rsph, s, q), where we have defined s and q as
## s = (1.0*csph)/asph and q = (1.0*bsph)/asph.



"""
Here we compute the shape using LSIM method in which we keep the enclosed volume fixed!!!
"""
####
######
#######
r_space1 = np.arange(2, 400, 2)  ## Please change this based on your numbers 10-200
r_space2 = np.arange(2, 400, 2)
####
######
#######

print(r_space1)

mean_s = []
standard_dev_s1 = []
standard_dev_s2 = []

mean_q = []
standard_dev_q1 = []
standard_dev_q2 = []


mean_T = []
standard_dev_T1 = []
standard_dev_T2 = []


for i in range(len(r_space1)-1):
    q_shape = []
    s_shape = []
    T_shape = []
    for uu in range(len(Real_DISK_Gal)):
        try:
            j = Real_DISK_Gal[uu]
            #rr1 = np.genfromtxt('./DMShape1/Approach3/shapeMW-principal3/shape_prin_'+str(j).zfill(3)+'.txt')
            rr1 = np.genfromtxt('/n/home08/twaters/shape_finder_outputs/Shape_Finder_Output_Hot_Gas/Gas_LSIM/shapeMW-principal3/shape_prin_'+str(j).zfill(3)+'.txt')
            shapee_s = []
            shaS  = []
            
            shapee_q = []
            shaQ  = []
            
            shapee_T = []
            shaT = []
            
            mask1 = rr1[:,0] >= r_space1[i]
            mask2 = rr1[:,0] <  r_space1[i+1]
            mask_tot = np.logical_and(mask1, mask2)
            ## 2 refers to q and 1 refers to s. 
            shaS = rr1[mask_tot, 1]
            shaQ = rr1[mask_tot, 2]
            ShaT = (1.0 - shaQ**2.0)/(1.0 - shaS**2.0)
            
            shapee_s = np.append(shapee_s, shaS)
            shapee_q = np.append(shapee_q, shaQ)
            shapee_T = np.append(shapee_T, ShaT)
            
            
            if len(shapee_s)> 0:
               s_shape = np.append(s_shape, shapee_s)
               q_shape = np.append(q_shape, shapee_q)
               T_shape = np.append(T_shape, shapee_T)
               

            else:
                pass
        except IOError:
              pass
    
    if len(q_shape) >0:
        #print(i,r_space1[i], len(q_shape), np.median(T_shape))
          
        mean_s = np.append(mean_s, [r_space1[i], np.median(s_shape)])
        mean_q = np.append(mean_q, [r_space1[i], np.median(q_shape)])
        mean_T = np.append(mean_T, [r_space1[i], np.median(T_shape)])
          
           
        standard_dev_s1 = np.append(standard_dev_s1, [r_space1[i], np.percentile(s_shape, 16)])
        standard_dev_q1 = np.append(standard_dev_q1, [r_space1[i], np.percentile(q_shape, 16)])
        standard_dev_T1 = np.append(standard_dev_T1, [r_space1[i], np.percentile(T_shape, 16)])
           
        standard_dev_s2 = np.append(standard_dev_s2, [r_space1[i], np.percentile(s_shape, 84)])
        standard_dev_q2 = np.append(standard_dev_q2, [r_space1[i], np.percentile(q_shape, 84)])
        standard_dev_T2 = np.append(standard_dev_T2, [r_space1[i], np.percentile(T_shape, 84)])

                
        
row_Shape_s = int(len(mean_s)/2.0)
Mean_vec_s = mean_s.reshape(row_Shape_s,2)

row_Shape_q = int(len(mean_q)/2.0)
Mean_vec_q = mean_q.reshape(row_Shape_q,2)

row_Shape_T = int(len(mean_T)/2.0)
Mean_vec_T = mean_T.reshape(row_Shape_T,2)

row_Shape_std_s1 = int(len(standard_dev_s1)/2.0)
Std_vec_s1 = standard_dev_s1.reshape(row_Shape_std_s1, 2)

row_Shape_std_s2 = int(len(standard_dev_s2)/2.0)
Std_vec_s2 = standard_dev_s2.reshape(row_Shape_std_s2, 2)

row_Shape_std_q1 = int(len(standard_dev_q1)/2.0)
Std_vec_q1 = standard_dev_q1.reshape(row_Shape_std_q1, 2)

row_Shape_std_q2 = int(len(standard_dev_q2)/2.0)
Std_vec_q2 = standard_dev_q2.reshape(row_Shape_std_q2, 2)

row_Shape_std_T1 = int(len(standard_dev_T1)/2.0)
Std_vec_T1 = standard_dev_T1.reshape(row_Shape_std_T1, 2)

row_Shape_std_T2 = int(len(standard_dev_T2)/2.0)
Std_vec_T2 = standard_dev_T2.reshape(row_Shape_std_T2, 2)

final_mean_s = np.median(Mean_vec_s[:,1])
final_std_s1 = np.median(Std_vec_s1[:,1])
final_std_s2 = np.median(Std_vec_s2[:,1])


final_mean_q = np.median(Mean_vec_q[:,1])
final_std_q1 = np.median(Std_vec_q1[:,1])
final_std_q2 = np.median(Std_vec_q2[:,1])


final_mean_T = np.median(Mean_vec_T[:,1])
final_std_T1 = np.median(Std_vec_T1[:,1])
final_std_T2 = np.median(Std_vec_T2[:,1])


print("s LSIM-Enclosed-Volume:", final_mean_s, final_std_s1, final_std_s2)
print("q LSIM-Enclosed-Volume:", final_mean_q, final_std_q1, final_std_q2)
print("T LSIM-Enclosed-Volume:", final_mean_T, final_std_T1, final_std_T2)


##################################

"""
Here we compute the shape using LSIM method in which we keep the semi-majopr axes fixed!!!
"""

mean_s2 = []
standard_dev_s12 = []
standard_dev_s22 = []

mean_q2 = []
standard_dev_q12 = []
standard_dev_q22 = []


mean_T2 = []
standard_dev_T12 = []
standard_dev_T22 = []


for i in range(len(r_space2)-1):
    q_shape2 = []
    s_shape2 = []
    T_shape2 = []
    for uu in range(len(Real_DISK_Gal)):
        #pdb.set_trace()
        try:
            j = Real_DISK_Gal[uu]
            rr1 = np.genfromtxt('/n/home08/twaters/shape_finder_outputs/Shape_Finder_Output_Hot_Gas/Gas_LSIM/shapeMW-principal3/shape_prin_'+str(j).zfill(3)+'.txt')
            shapee_s2 = []
            shaS2  = []
            
            shapee_q2 = []
            shaQ2  = []
            
            shapee_T2 = []
            shaT2 = []
            
            mask11 = rr1[:,0] >= r_space2[i]
            mask22 = rr1[:,0] <  r_space2[i+1]
            mask_tot2 = np.logical_and(mask11, mask22)
            ## 2 refers to q and 1 refers to s.
            shaS2 = rr1[mask_tot2, 1]
            shaQ2 = rr1[mask_tot2, 2]
            ShaT2 = (1.0 - shaQ2**2.0)/(1.0 - shaS2**2.0)
            
            
            shapee_s2 = np.append(shapee_s2, shaS2)
            shapee_q2 = np.append(shapee_q2, shaQ2)
            shapee_T2 = np.append(shapee_T2, ShaT2)
            
            
            if len(shapee_s2)> 0:
               s_shape2 = np.append(s_shape2, shapee_s2)
               q_shape2 = np.append(q_shape2, shapee_q2)
               T_shape2 = np.append(T_shape2, shapee_T2)
               
            else:
                pass

        except IOError:
              pass
    
    if len(q_shape2) >0:
        #print(i,r_space1[i], len(q_shape4), np.median(T_shape4))
          
        mean_s2 = np.append(mean_s2, [r_space2[i], np.median(s_shape2)])
        mean_q2 = np.append(mean_q2, [r_space2[i], np.median(q_shape2)])
        mean_T2 = np.append(mean_T2, [r_space2[i], np.median(T_shape2)])
          
           
        standard_dev_s12 = np.append(standard_dev_s12, [r_space2[i], np.percentile(s_shape2, 16)])
        standard_dev_q12 = np.append(standard_dev_q12, [r_space2[i], np.percentile(q_shape2, 16)])
        standard_dev_T12 = np.append(standard_dev_T12, [r_space2[i], np.percentile(T_shape2, 16)])
           
        standard_dev_s22 = np.append(standard_dev_s22, [r_space2[i], np.percentile(s_shape2, 84)])
        standard_dev_q22 = np.append(standard_dev_q22, [r_space2[i], np.percentile(q_shape2, 84)])
        standard_dev_T22 = np.append(standard_dev_T22, [r_space2[i], np.percentile(T_shape2, 84)])

 
row_Shape_s2 = int(len(mean_s2)/2.0)
Mean_vec_s2 = mean_s2.reshape(row_Shape_s2,2)

row_Shape_q2 = int(len(mean_q2)/2.0)
Mean_vec_q2 = mean_q2.reshape(row_Shape_q2,2)

row_Shape_T2 = int(len(mean_T2)/2.0)
Mean_vec_T2 = mean_T2.reshape(row_Shape_T2,2)


row_Shape_std_s12 = int(len(standard_dev_s12)/2.0)
Std_vec_s12 = standard_dev_s12.reshape(row_Shape_std_s12, 2)

row_Shape_std_s22 = int(len(standard_dev_s22)/2.0)
Std_vec_s22 = standard_dev_s22.reshape(row_Shape_std_s22, 2)


row_Shape_std_q12 = int(len(standard_dev_q12)/2.0)
Std_vec_q12 = standard_dev_q12.reshape(row_Shape_std_q12, 2)

row_Shape_std_q22 = int(len(standard_dev_q22)/2.0)
Std_vec_q22 = standard_dev_q22.reshape(row_Shape_std_q22, 2)


row_Shape_std_T12 = int(len(standard_dev_T12)/2.0)
Std_vec_T12 = standard_dev_T12.reshape(row_Shape_std_T12, 2)

row_Shape_std_T22 = int(len(standard_dev_T22)/2.0)
Std_vec_T22 = standard_dev_T22.reshape(row_Shape_std_T22, 2)


final_mean_s2 = np.median(Mean_vec_s2[:,1])
final_std_s12 = np.median(Std_vec_s12[:,1])
final_std_s22 = np.median(Std_vec_s22[:,1])


final_mean_q2 = np.median(Mean_vec_q2[:,1])
final_std_q12 = np.median(Std_vec_q12[:,1])
final_std_q22 = np.median(Std_vec_q22[:,1])


final_mean_T2 = np.median(Mean_vec_T2[:,1])
final_std_T12 = np.median(Std_vec_T12[:,1])
final_std_T22 = np.median(Std_vec_T22[:,1])


#print("s LSIM-Semi-Major:", final_mean_s2, final_std_s12, final_std_s22)
#print("q LSIM-Semi-Major:", final_mean_q2, final_std_q12, final_std_q22)
#print("T LSIM-Semi-Major:", final_mean_T2, final_std_T12, final_std_T22)


####################################################


"""
Finally, we compute the shape using an enclosed volume fixed method. EVIM



mean_s4 = []
standard_dev_s14 = []
standard_dev_s24 = []

mean_q4 = []
standard_dev_q14 = []
standard_dev_q24 = []


mean_T4 = []
standard_dev_T14 = []
standard_dev_T24 = []


for i in range(len(r_space2)-1):
    q_shape4 = []
    s_shape4 = []
    T_shape4 = []
    for uu in range(len(Real_DISK_Gal)):
        #pdb.set_trace()
        try:
            j = Real_DISK_Gal[uu]
            rr1 = np.genfromtxt('./Shape_SH_X_Semi_Converge_Whole/Star_Approach3N/shapeMW-principal3/shape_prin_'+str(j).zfill(3)+'.txt')
            shapee_s4 = []
            shaS4  = []
            
            shapee_q4 = []
            shaQ4  = []
            
            shapee_T4 = []
            shaT4 = []
            
            mask1 = rr1[:,0] >= r_space2[i]
            mask2 = rr1[:,0] <  r_space2[i+1]
            mask_tot4 = np.logical_and(mask1, mask2)
            ## 2 refers to q and 1 refers to s.
            shaS4 = rr1[mask_tot4, 1]
            shaQ4 = rr1[mask_tot4, 2]
            ShaT4 = (1.0 - shaQ4**2.0)/(1.0 - shaS4**2.0)
            
            shapee_s4 = np.append(shapee_s4, shaS4)
            shapee_q4 = np.append(shapee_q4, shaQ4)
            shapee_T4 = np.append(shapee_T4, ShaT4)
            
            
            if len(shapee_s4)> 0:
               s_shape4 = np.append(s_shape4, shapee_s4)
               q_shape4 = np.append(q_shape4, shapee_q4)
               T_shape4 = np.append(T_shape4, shapee_T4)
               
            else:
                pass

        except IOError:
              pass
    
    if len(q_shape4) >0:
        #print(i,r_space1[i], len(q_shape4), np.median(T_shape4))
          
        mean_s4 = np.append(mean_s4, [r_space2[i], np.median(s_shape4)])
        mean_q4 = np.append(mean_q4, [r_space2[i], np.median(q_shape4)])
        mean_T4 = np.append(mean_T4, [r_space2[i], np.median(T_shape4)])
          
           
        standard_dev_s14 = np.append(standard_dev_s14, [r_space2[i], np.percentile(s_shape4, 16)])
        standard_dev_q14 = np.append(standard_dev_q14, [r_space2[i], np.percentile(q_shape4, 16)])
        standard_dev_T14 = np.append(standard_dev_T14, [r_space2[i], np.percentile(T_shape4, 16)])
           
        standard_dev_s24 = np.append(standard_dev_s24, [r_space2[i], np.percentile(s_shape4, 84)])
        standard_dev_q24 = np.append(standard_dev_q24, [r_space2[i], np.percentile(q_shape4, 84)])
        standard_dev_T24 = np.append(standard_dev_T24, [r_space2[i], np.percentile(T_shape4, 84)])

 
row_Shape_s4 = int(len(mean_s4)/2.0)
Mean_vec_s4 = mean_s4.reshape(row_Shape_s4,2)

row_Shape_q4 = int(len(mean_q4)/2.0)
Mean_vec_q4 = mean_q4.reshape(row_Shape_q4,2)

row_Shape_T4 = int(len(mean_T4)/2.0)
Mean_vec_T4 = mean_T4.reshape(row_Shape_T4,2)


row_Shape_std_s14 = int(len(standard_dev_s14)/2.0)
Std_vec_s14 = standard_dev_s14.reshape(row_Shape_std_s14, 2)

row_Shape_std_s24 = int(len(standard_dev_s24)/2.0)
Std_vec_s24 = standard_dev_s24.reshape(row_Shape_std_s24, 2)


row_Shape_std_q14 = int(len(standard_dev_q14)/2.0)
Std_vec_q14 = standard_dev_q14.reshape(row_Shape_std_q14, 2)

row_Shape_std_q24 = int(len(standard_dev_q24)/2.0)
Std_vec_q24 = standard_dev_q24.reshape(row_Shape_std_q24, 2)


row_Shape_std_T14 = int(len(standard_dev_T14)/2.0)
Std_vec_T14 = standard_dev_T14.reshape(row_Shape_std_T14, 2)

row_Shape_std_T24 = int(len(standard_dev_T24)/2.0)
Std_vec_T24 = standard_dev_T24.reshape(row_Shape_std_T24, 2)


final_mean_s4 = np.median(Mean_vec_s4[:,1])
final_std_s14 = np.median(Std_vec_s14[:,1])
final_std_s24 = np.median(Std_vec_s24[:,1])


final_mean_q4 = np.median(Mean_vec_q4[:,1])
final_std_q14 = np.median(Std_vec_q14[:,1])
final_std_q24 = np.median(Std_vec_q24[:,1])


final_mean_T4 = np.median(Mean_vec_T4[:,1])
final_std_T14 = np.median(Std_vec_T14[:,1])
final_std_T24 = np.median(Std_vec_T24[:,1])


print("s EVIM:", final_mean_s4, final_std_s14, final_std_s24)
print("q EVIM:", final_mean_q4, final_std_q14, final_std_q24)
print("T EVIM:", final_mean_T4, final_std_T14, final_std_T24)

########################################################
"""

fig=plt.figure(figsize=(19,2))
ax = plt.subplot(1,3,1)
#ax = fig.add_subplot(621)
plt.subplots_adjust(top =1.8, bottom=0.2,hspace=0.3, wspace=0.3)
#ax.set_title(r" Cen Halo", fontsize = 12)

#mask3 = np.in1d(Mean_vec_s[:,0], even_Num, assume_unique=True)

#mask4 = np.in1d(Mean_vec_s4[:,0], even_Num1, assume_unique=True)

#print(np.sum(mask3))

ysmoothed_min_mean = gaussian_filter1d(Mean_vec_s[:, 1], sigma=2)
ysmoothed_min_std1 = gaussian_filter1d(Std_vec_s1[:,1], sigma=2)
ysmoothed_min_std2 = gaussian_filter1d(Std_vec_s2[:,1], sigma=2)

ax.plot(Mean_vec_s[:,0], ysmoothed_min_std1, c='darkred',linestyle='-',alpha=0.3, markeredgewidth=5, linewidth=0, markersize=12)
ax.plot(Mean_vec_s[:,0], ysmoothed_min_std2, c='darkred',linestyle='-',alpha=0.3, markeredgewidth=5, linewidth=0, markersize=12)
ax.plot(Mean_vec_s[:,0], ysmoothed_min_mean, c='darkred',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'LSIM')
ax.fill_between(Mean_vec_s[:, 0], ysmoothed_min_mean, ysmoothed_min_std2, facecolor='darkred', interpolate=True, alpha=0.5)
ax.fill_between(Mean_vec_s[:, 0], ysmoothed_min_mean,ysmoothed_min_std1, facecolor='darkred', interpolate=True, alpha=0.5)



ysmoothed2_min_mean = gaussian_filter1d(Mean_vec_s2[:, 1], sigma=2)
ysmoothed2_min_std1 = gaussian_filter1d(Std_vec_s12[:,1], sigma=2)
ysmoothed2_min_std2 = gaussian_filter1d(Std_vec_s22[:,1], sigma=2)


#ax.plot(Mean_vec_s2[:,0], ysmoothed2_min_std1, c='crimson',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_s2[:,0], ysmoothed2_min_std2, c='crimson',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_s2[:,0], ysmoothed2_min_mean, c='crimson',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'LSIM(Semi-Major)')
#ax.fill_between(Mean_vec_s2[:,0], ysmoothed2_min_mean, ysmoothed2_min_std2, facecolor='crimson', interpolate=True, alpha=0.3)
#ax.fill_between(Mean_vec_s2[:,0], ysmoothed2_min_mean, ysmoothed2_min_std1, facecolor='crimson', interpolate=True, alpha=0.3)
#forceAspect(ax,aspect=1)


#ysmoothed4_min_mean = gaussian_filter1d(Mean_vec_s4[:, 1], sigma=2)
#ysmoothed4_min_std1 = gaussian_filter1d(Std_vec_s14[:,1], sigma=2)
#ysmoothed4_min_std2 = gaussian_filter1d(Std_vec_s24[:,1], sigma=2)


#ax.plot(Mean_vec_s4[:,0], ysmoothed4_min_std1, c='orange',linestyle='-',alpha=0.8, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_s4[:,0], ysmoothed4_min_std2, c='orange',linestyle='-',alpha=0.8, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_s4[:,0], ysmoothed4_min_mean, c='orange',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'EVIM')
#ax.fill_between(Mean_vec_s4[:,0], ysmoothed4_min_mean, ysmoothed4_min_std2, facecolor='orange', interpolate=True, alpha=0.4)
#ax.fill_between(Mean_vec_s4[:,0], ysmoothed4_min_mean, ysmoothed4_min_std1, facecolor='orange', interpolate=True, alpha=0.4)
#forceAspect(ax,aspect=1)

#ax.plot(Mean_vec_s4[:,0], Std_vec_s14[:,1], c='darkgreen',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_s4[:,0], Std_vec_s24[:,1], c='darkgreen',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_s4[:,0], Mean_vec_s4[:,1], c='darkgreen',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= 'App2')
#ax.fill_between(Mean_vec_s4[:,0], Mean_vec_s4[:,1], Std_vec_s24[:,1], facecolor='green', interpolate=True, alpha=0.4)
#ax.fill_between(Mean_vec_s4[:,0], Mean_vec_s4[:,1], Std_vec_s14[:,1], facecolor='green', interpolate=True, alpha=0.4)


plt.tick_params(axis='both', which='major', labelsize=10)
axes = plt.gca()
#legend = plt.legend(loc=2, borderaxespad=1., ncol=1,numpoints=1,fontsize = 10,fancybox=True)
ax.set_aspect('auto')

plt.ylabel(r'$s = a/c$',fontsize=14)
plt.xlabel(r'$ r(kpc)$', fontsize=14)
axes.set_xlim([0.0, 225])
axes.set_ylim([0.2, 0.7])

#plt.xscale('log')

ax = plt.subplot(1,3,2)
plt.subplots_adjust(top =1.8, bottom=0.2,hspace=0.3, wspace=0.3)

#ax.set_title("(DM Halo)", fontsize = 12)

#ax.plot(Mean_vec_q[mask3,0], Std_vec_q1[mask3,1], c='blue',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q[mask3,0], Std_vec_q2[mask3,1], c='blue',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q[mask3,0], Mean_vec_q[mask3,1], c='blue',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= 'App1')
#ax.fill_between(Mean_vec_q[mask3,0], Mean_vec_q[mask3,1], Std_vec_q2[mask3,1], facecolor='blue', interpolate=True, alpha=0.2)
#ax.fill_between(Mean_vec_q[mask3,0], Mean_vec_q[mask3,1], Std_vec_q1[mask3,1], facecolor='blue', interpolate=True, alpha=0.2)

#ax.plot(Mean_vec_q4[mask4,0], Std_vec_q14[mask4,1], c='darkgreen',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q4[mask4,0], Std_vec_q24[mask4,1], c='darkgreen',linestyle='-',alpha=0.4, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q4[mask4,0], Mean_vec_q4[mask4,1], c='darkgreen',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= 'App2')
#ax.fill_between(Mean_vec_q4[mask4,0], Mean_vec_q4[mask4,1], Std_vec_q24[mask4,1], facecolor='green', interpolate=True, alpha=0.4)
#ax.fill_between(Mean_vec_q4[mask4,0], Mean_vec_q4[mask4,1], Std_vec_q14[mask4,1], facecolor='green', interpolate=True, alpha=0.4)

ysmoothed_min_mean = gaussian_filter1d(Mean_vec_q[:, 1], sigma=2)
ysmoothed_min_std1 = gaussian_filter1d(Std_vec_q1[:,1], sigma=2)
ysmoothed_min_std2 = gaussian_filter1d(Std_vec_q2[:,1], sigma=2)

ax.plot(Mean_vec_q[:,0], ysmoothed_min_std1, c='darkred',linestyle='-',alpha=0.3, markeredgewidth=5, linewidth=0, markersize=12)
ax.plot(Mean_vec_q[:,0], ysmoothed_min_std2, c='darkred',linestyle='-',alpha=0.3, markeredgewidth=5, linewidth=0, markersize=12)
ax.plot(Mean_vec_q[:,0], ysmoothed_min_mean, c='darkred',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'LSIM')
ax.fill_between(Mean_vec_q[:, 0], ysmoothed_min_mean, ysmoothed_min_std2, facecolor='darkred', interpolate=True, alpha=0.5)
ax.fill_between(Mean_vec_q[:, 0], ysmoothed_min_mean,ysmoothed_min_std1, facecolor='darkred', interpolate=True, alpha=0.5)

#ax.fill_between(Mean_vec_q[:, 0], Mean_vec_q[:, 1],Std_vec_q1[:, 1], facecolor='blue', interpolate=True, alpha=0.2)


ysmoothed2_min_mean = gaussian_filter1d(Mean_vec_q2[:, 1], sigma=2)
ysmoothed2_min_std1 = gaussian_filter1d(Std_vec_q12[:,1], sigma=2)
ysmoothed2_min_std2 = gaussian_filter1d(Std_vec_q22[:,1], sigma=2)


#ax.plot(Mean_vec_q2[:,0], ysmoothed2_min_std1, c='crimson',linestyle='-',alpha=0.6, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q2[:,0], ysmoothed2_min_std2, c='crimson',linestyle='-',alpha=0.6, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q2[:,0], ysmoothed2_min_mean, c='crimson',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'LSIM(Semi-Major)')
#ax.fill_between(Mean_vec_q2[:,0], ysmoothed2_min_mean, ysmoothed2_min_std2, facecolor='crimson', interpolate=True, alpha=0.4)
#ax.fill_between(Mean_vec_q2[:,0], ysmoothed2_min_mean, ysmoothed2_min_std1, facecolor='crimson', interpolate=True, alpha=0.4)
#forceAspect(ax,aspect=1)


#ysmoothed4_min_mean = gaussian_filter1d(Mean_vec_q4[:, 1], sigma=2)
#ysmoothed4_min_std1 = gaussian_filter1d(Std_vec_q14[:,1], sigma=2)
#ysmoothed4_min_std2 = gaussian_filter1d(Std_vec_q24[:,1], sigma=2)


#ax.plot(Mean_vec_q4[:,0], ysmoothed4_min_std1, c='orange',linestyle='-',alpha=0.8, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q4[:,0], ysmoothed4_min_std2, c='orange',linestyle='-',alpha=0.8, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_q4[:,0], ysmoothed4_min_mean, c='orange',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'EVIM')
#ax.fill_between(Mean_vec_q4[:,0], ysmoothed4_min_mean, ysmoothed4_min_std2, facecolor='orange', interpolate=True, alpha=0.4)
#ax.fill_between(Mean_vec_q4[:,0], ysmoothed4_min_mean, ysmoothed4_min_std1, facecolor='orange', interpolate=True, alpha=0.4)
#ax.set_aspect('auto')



plt.tick_params(axis='both', which='major', labelsize=10)
axes = plt.gca()
#legend = plt.legend(loc=2, borderaxespad=1., ncol=1,numpoints=1,fontsize = 10,fancybox=True)

plt.ylabel(r'$q = b/c$', fontsize=14)
plt.xlabel(r'$ r(kpc)$', fontsize=14)
axes.set_xlim([0.0, 225])
axes.set_ylim([0.4, 1.0])


ax = plt.subplot(1,3,3)
plt.subplots_adjust(top =1.8, bottom=0.2,hspace=0.3, wspace=0.3)
#ax.set_title("(DM Halo)", fontsize = 12)


#ax.plot(Mean_vec_T[mask3,0], Std_vec_T1[mask3,1], c='blue',linestyle='-',alpha=0.9, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T[mask3,0], Std_vec_T2[mask3,1], c='blue',linestyle='-',alpha=0.9, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T[mask3,0], Mean_vec_T[mask3,1], c='blue',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= 'App1')
#ax.fill_between(Mean_vec_T[mask3,0], Mean_vec_T[mask3,1], Std_vec_T2[mask3,1], facecolor='blue', interpolate=True, alpha=0.2)
#ax.fill_between(Mean_vec_T[mask3,0], Mean_vec_T[mask3,1], Std_vec_T1[mask3,1], facecolor='blue', interpolate=True, alpha=0.2)

#ax.plot(Mean_vec_T4[mask4,0], Std_vec_T14[mask4,1], c='darkgreen',linestyle='-',alpha=0.9, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T4[mask4,0], Std_vec_T24[mask4,1], c='darkgreen',linestyle='-',alpha=0.9, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T4[mask4,0], Mean_vec_T4[mask4,1], c='darkgreen',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= 'App2')
#ax.fill_between(Mean_vec_T4[mask4,0], Mean_vec_T4[mask4,1], Std_vec_T24[mask4,1], facecolor='green', interpolate=True, alpha=0.3)
#ax.fill_between(Mean_vec_T4[mask4,0], Mean_vec_T4[mask4,1], Std_vec_T14[mask4,1], facecolor='green', interpolate=True, alpha=0.3)

ysmoothed_min_mean = gaussian_filter1d(Mean_vec_T[:, 1], sigma=2)
ysmoothed_min_std1 = gaussian_filter1d(Std_vec_T1[:,1], sigma=2)
ysmoothed_min_std2 = gaussian_filter1d(Std_vec_T2[:,1], sigma=2)

ax.plot(Mean_vec_T[:,0], ysmoothed_min_std1, c='darkred',linestyle='-',alpha=0.45, markeredgewidth=5, linewidth=0, markersize=12)
ax.plot(Mean_vec_T[:,0], ysmoothed_min_std2, c='darkred',linestyle='-',alpha=0.45, markeredgewidth=5, linewidth=0, markersize=12)
ax.plot(Mean_vec_T[:,0], ysmoothed_min_mean, c='darkred',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'LSIM')
ax.fill_between(Mean_vec_T[:, 0], ysmoothed_min_mean, ysmoothed_min_std2, facecolor='darkred', interpolate=True, alpha=0.5)
ax.fill_between(Mean_vec_T[:, 0], ysmoothed_min_mean,ysmoothed_min_std1, facecolor='darkred', interpolate=True, alpha=0.5)
ax.set_aspect('auto')


ysmoothed2_min_mean = gaussian_filter1d(Mean_vec_T2[:, 1], sigma=2)
ysmoothed2_min_std1 = gaussian_filter1d(Std_vec_T12[:,1], sigma=2)
ysmoothed2_min_std2 = gaussian_filter1d(Std_vec_T22[:,1], sigma=2)


#ax.plot(Mean_vec_T2[:,0], ysmoothed2_min_std1, c='crimson',linestyle='-',alpha=0.6, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T2[:,0], ysmoothed2_min_std2, c='crimson',linestyle='-',alpha=0.6, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T2[:,0], ysmoothed2_min_mean, c='crimson',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'LSIM(Semi-Major)')
#ax.fill_between(Mean_vec_T2[:,0], ysmoothed2_min_mean, ysmoothed2_min_std2, facecolor='crimson', interpolate=True, alpha=0.4)
#ax.fill_between(Mean_vec_T2[:,0], ysmoothed2_min_mean, ysmoothed2_min_std1, facecolor='crimson', interpolate=True, alpha=0.4)
#forceAspect(ax,aspect=1)

#ysmoothed4_min_mean = gaussian_filter1d(Mean_vec_T4[:, 1], sigma=2)
#ysmoothed4_min_std1 = gaussian_filter1d(Std_vec_T14[:,1], sigma=2)
#ysmoothed4_min_std2 = gaussian_filter1d(Std_vec_T24[:,1], sigma=2)


#ax.plot(Mean_vec_T4[:,0], ysmoothed4_min_std1, c='orange',linestyle='-',alpha=0.8, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T4[:,0], ysmoothed4_min_std2, c='orange',linestyle='-',alpha=0.8, markeredgewidth=5, linewidth=0, markersize=12)
#ax.plot(Mean_vec_T4[:,0], ysmoothed4_min_mean, c='orange',linestyle='-',alpha=1.0, markeredgewidth=5, linewidth=2.1, markersize=12, label= r'EVIM')
#ax.fill_between(Mean_vec_T4[:,0], ysmoothed4_min_mean, ysmoothed4_min_std2, facecolor='orange', interpolate=True, alpha=0.4)
#ax.fill_between(Mean_vec_T4[:,0], ysmoothed4_min_mean, ysmoothed4_min_std1, facecolor='orange', interpolate=True, alpha=0.4)


#legend = plt.legend(loc=2, borderaxespad=1., ncol=1,numpoints=1,fontsize = 10,fancybox=True)

plt.tick_params(axis='both', which='major', labelsize=10)
axes = plt.gca()

plt.ylabel(r'$T = \frac{(1-(b/c)^2)}{(1-(a/c)^2)}$',fontsize=13)
plt.xlabel(r'$ r(kpc)$', fontsize=14)
axes.set_xlim([0.0, 225])
axes.set_ylim([0.1, 0.9])
#plt.xscale('log')


plt.savefig('/n/home08/twaters/shape_finder_outputs/mean_plots/Shape_sqT_Hot.pdf',dpi = 400, transparent = True,bbox_inches='tight')
