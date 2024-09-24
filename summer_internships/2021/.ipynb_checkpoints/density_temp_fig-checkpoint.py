import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import sys
import sphviewer
import pdb
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sphviewer.tools import QuickView




labb_tit = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)', '(10)','(11)', '(12)', '(13)', '(14)', '(15)', '(16)', '(17)', '(18)', '(19)', '(20)', '(21)', '(22)', '(23)', '(24)', '(25)']

gal_identifier = ['476266', '478216', '479938', '480802', '485056', '488530', '494709', '497557', '501208', '501725', '502995', '503437', '505586', '506720', '509091', '510585', '511303', '513845', '519311', '522983', '523889', '529365', '530330', '535410', '538905']


Real_DISK_Gal = np.array([ 2,  4,  5,  6,  7, 10, 15, 17, 20, 21, 23, 24, 27, 29, 31, 34, 35, 39, 48, 52, 54, 57, 58, 63, 65])

### Interesting toturial: http://alejandrobll.github.io/py-sphviewer/content/tutorial_projections.html

virial_mean_array = []

for uu in range(len(Real_DISK_Gal)):
    ii = Real_DISK_Gal[uu]
    print(uu)

    gas_location = (1.0/0.6774)*np.genfromtxt('/n/home08/twaters/gas_morphology/data/TNG50/LocationGas50/gas_coorRE_Gal_'+str(ii).zfill(3)+'.txt')
    Virial_Gas_Temp = np.genfromtxt('/n/home08/twaters/gas_morphology/data/TNG50/VirialTemperature50/Gas_Virial_Temp_Gal'+str(ii).zfill(3)+'.txt')
    Temp = np.genfromtxt('/n/home08/twaters/gas_morphology/data/TNG50/Temperature50/Gas_Temp_Gal'+str(ii).zfill(3)+'.txt')
    
    Virial_mean = np.mean(Virial_Gas_Temp)
    virial_mean_array.append(Virial_mean)
    
    Cold_Temp_Mask = Temp < 10**(4.2)
    Warm_Temp_Mask = np.logical_and(Temp > 10**(4.2), Temp < Virial_mean*3)
    Hot_Temp_Mask = Temp > Virial_mean*3

    cold_gas = Temp[Cold_Temp_Mask]
    warm_gas = Temp[Warm_Temp_Mask]
    hot_gas = Temp[Hot_Temp_Mask]
    
    
    ###
    ####
    ######
    gas_mask1 = Cold_Temp_Mask
    gas_mask2 = Warm_Temp_Mask
    gas_mask3 = Hot_Temp_Mask
    
    gas1 = cold_gas
    gas2 = warm_gas
    gas3 = hot_gas
    virial_radius = 400
    #v_min = 0
    #v_max = 1.5
    ######
    ####
    ###
    
    gas_Loc1 = gas_location[gas_mask1]
    gas_Loc2 = gas_location[gas_mask2]
    gas_Loc3 = gas_location[gas_mask3]


    gas_Length = np.sqrt(gas_Loc[:,0]*gas_Loc[:,0] + gas_Loc[:,1]*gas_Loc[:,1] + gas_Loc[:,2]*gas_Loc[:,2])
    NN = virial_radius
    
    mask1 = abs(gas_Loc[:,0]) <= NN
    mask2 = abs(gas_Loc[:,1]) <= NN
    mask3 = abs(gas_Loc[:,2]) <= NN

    mask_fin = np.logical_and(mask1, np.logical_and(mask2, mask3))
    pos = gas_Loc[mask_fin,:]

    Temp_masked_1 = gas[mask_fin]
    hh = (0.39/0.6774)*np.ones(len(pos))
    mm = np.ones(len(pos))

    Particles_density = sphviewer.Particles(pos, mm, hh)
    Particles_temp = sphviewer.Particles(pos, Temp_masked, hh)

    Scene = sphviewer.Scene(Particles)
    
    fig = plt.figure(1,figsize=(15,5))
    
    #set a figure title on top
    
    #
    ###
    ####
    ######
    #fig.suptitle("Warm Gas Density: " + str(gal_identifier[uu]), fontsize=17, x=0.5, y=1.5)
    ######
    ####
    ###
    #
    
    plt.subplots_adjust(top =1.8, bottom=0.2,hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    
    extendd = [-NN,NN,-NN,NN]
    Scene.update_camera(r='infinity', t=0, p = 0, roll = 0, x = 0, y = 0, z = 0, extent= extendd)

    #ax[0,0]
    
    Render = sphviewer.Render(Scene)
    Render.set_logscale()
    img1 = Render.get_image()
    extent1 = Render.get_extent()
    divider = make_axes_locatable(ax1)



    image1 = ax1.imshow(img1, extent=extent1, origin='lower', cmap=plt.cm.jet, vmin = v_min, vmax = v_max, rasterized=True)
    ax1.set_xlabel('X(kpc)', size=12)
    ax1.set_ylabel('Y(kpc)', size=12)

    cax = divider.new_vertical(size="7%", pad=0.7, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(image1, cax=cax, orientation="horizontal")
    #cb.set_label(label='$log_{10}(Density) \quad ((10^{10}M_{\odot}/h)/(ckpc/h)^2)$', size='large', weight='bold')
    cb.ax.tick_params(labelsize=12)


    #Scene.update_camera(r='infinity', t=-90, p = -90, roll = 0, x = 0, y = 0, z = 0,  extent= extendd)

    #ax[0,1]
    

    Render = sphviewer.Render(Scene)
    Render.set_logscale()
    img2 = Render.get_image()
    extent2 = Render.get_extent()
    divider = make_axes_locatable(ax2)
    
    
    image2 = ax2.imshow(img2, extent=extent2, origin='lower',cmap=plt.cm.jet, vmin = v_min, vmax = v_max, rasterized=True)
    ax2.set_xlabel('Y(kpc)', size=12)
    ax2.set_ylabel('Z(kpc)', size=12)
    
    cax = divider.new_vertical(size="7%", pad=0.7, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(image2, cax=cax, orientation="horizontal")
    cb.set_label(label='$log_{10}(Density) \quad ((10^{10}M_{\odot}/h)/(ckpc/h)^2)$', size='large', weight='bold')
    cb.ax.tick_params(labelsize=12)


    #Scene.update_camera(r='infinity', t=90, p = 0, roll = -90, x = 0, y = 0, z = 0,  extent= extendd)


    #ax[0,2]

    Render = sphviewer.Render(Scene)
    Render.set_logscale()
    img3 = Render.get_image()
    extent3 = Render.get_extent()
    divider = make_axes_locatable(ax3)

    image3 = ax3.imshow(img3, extent=extent2, origin='lower',cmap=plt.cm.jet, vmin = v_min, vmax = v_max, rasterized=True)
    ax3.set_xlabel('Z(kpc)', size=12)
    ax3.set_ylabel('X(kpc)', size=12)

    cax = divider.new_vertical(size="7%", pad=0.7, pack_start=True)
    fig.add_axes(cax)
    cb = fig.colorbar(image3, cax=cax, orientation="horizontal")
    #cb.set_label(label='$log_{10}(Density) \quad ((10^{10}M_{\odot}/h)/(ckpc/h)^2)$', size='large', weight='bold')
    cb.ax.tick_params(labelsize=12)


    #ax[1,0]

    #ax[1,1]

    #ax[1,2]









    #
    ###
    ####
    ######
    plt.savefig('/n/home08/twaters/py-sphviewer/density_temp_maps/gas_dens_temp_N_' + str(uu).zfill(3) +'.png', dpi = 400, transparent = True,bbox_inches='tight')
    ######
    ####
    ###
    #
    pdb.set_trace()


virial_mean_tot = np.mean(virial_mean_array)
print(f'mean virial temp of all galaxies: {}')
