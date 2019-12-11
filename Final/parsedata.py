##################################################################################
### This script takes in data from bedmachine to make input images for the CNN #### 
### Data is not included, but it could be modified for future use ################
##################################################################################

###########################################################
### Import libraries and define some cropping functions ###
###########################################################

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pyproj import Proj, transform
import glob
from osgeo import gdal
import pickle as pk
import scipy.interpolate as si

def clip_array(F, x, y, xmin, xmax, ymin, ymax):
    x0 = len(x[(x<xmin)])
    x1 = len(x[(x<xmax)])
    y0 = len(y[(y<ymin)])
    y1 = len(y[(y<ymax)])
    if y0 == 0:
        return F[-y1:,x0:x1]
    else:
        return F[-y1:-y0,x0:x1]

def clip_coordarr(Fx, Fy, x, y, xmin, xmax, ymin, ymax):
    x0 = len(x[(x<xmin)])
    x1 = len(x[(x<xmax)])
    y0 = len(y[(y<ymin)])
    y1 = len(y[(y<ymax)])
    if y0 == 0:
        return Fx[x0:x1], Fx[-y1:]
    else:
        return Fx[x0:x1], Fx[-y1:-y0]

#########################################################
########### Preprocessing ###############################
#########################################################

#### Import Bed Machine Data #####
fi = Dataset('../BedMachineGreenland-2017-09-20.nc', mode ='r')

p_bedmach = Proj(init='epsg:3413') #define projections
p_gap = Proj(init='epsg:4326')

### Block Site where fieldwork was conducted by QSSI, good reference for images 
### of this region of the ice sheet
blk_lat = 67.182013
blk_lon = -49.569493
blk_x, blk_y = transform(p_gap, p_bedmach, blk_lon, blk_lat) #Transform Coords


dx = 150 #grid spacing of data (m)
dy = 150

xdow = 500 #number of points west to grab around coordinate
xup = 1000 #points east
ydow = 750 #points north(ish)
yup = 750 #points south


lon = fi.variables['x'][:] #separate data for cropping
lat = fi.variables['y'][:]

### Crop thickness, bed elevation, surface elevation, x and y coordinates ###
thick = clip_array(fi.variables['thickness'], lon, lat, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*dy,blk_y+yup*dy)

bed = clip_array(fi.variables['bed'], lon, lat, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*dy,blk_y+yup*dy)

surface = clip_array(fi.variables['surface'], lon, lat, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*dy,blk_y+yup*dy)

#geoid = clip_array(fi.variables['geoid'], lon, lat, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*dy,blk_y+yup*dy) #Could add geoid data if interested

xmac = clip_array(np.meshgrid(lon, lat)[0], lon, lat, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*dy,blk_y+yup*dy)[0,:]

ymac = clip_array(np.meshgrid(lon, lat)[1], lon, lat, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*dy,blk_y+yup*dy)[:,0]

fi.close()


#### Import Insar Vel Data #####
### Same projection as bedmachine different spacing ###

direc = '../insarvels/'
veltif = glob.glob(direc+'*vv*.tif') ### impor velocity tifs

### GAP coordinate
p_gap = Proj(init='epsg:4326') ## define projections
p_bedmach = Proj(init='epsg:3413')
blk_lat = 67.182013
blk_lon = -49.569493
blk_x, blk_y = transform(p_gap, p_bedmach, blk_lon, blk_lat) ## transform to our coord system

def croptif(tifs, i): #just in case I want multiple tifs, note, there are two different spatial resolutions which will produce different tifs
    tif = gdal.Open(tifs) 
    
    # Coordinate System
    coord = tif.GetGeoTransform()
    ll_x = coord[0]
    ll_y = coord[3]
    dx = coord[1]
    dy = coord[-1]
    xsize = tif.RasterXSize
    ysize = tif.RasterYSize
    xarr = np.arange(ll_x, ll_x + xsize*dx, dx)
    yarr = np.arange(ll_y, ll_y + ysize*dy, dy)
    xdow = 100 ## all of this clips it to the right size
    xup = 250
    ydow = 200
    yup = 200
    
    if i == 1 or i==11 or i==12 or i==14:
        xdow = 2.5*xdow
        xup = 2.5*xup
        ydow = 2.5*ydow
        yup = 2.5*yup    

    # Raster
    raster_obj = tif.GetRasterBand(1)
    raster = raster_obj.ReadAsArray()
    raster = clip_array(raster, xarr, yarr, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*(-dy),blk_y+yup*(-dy))
    coordsx = clip_array(np.meshgrid(xarr, yarr)[0], xarr, yarr, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*(-dy),blk_y+yup*(-dy))[0,:]
    coordsy = clip_array(np.meshgrid(xarr, yarr)[1], xarr, yarr, blk_x-xdow*dx, blk_x+xup*dx,blk_y-ydow*(-dy),blk_y+yup*(-dy))[:,0]
    
    return raster, coordsx, coordsy

velmag, xsv, ysv = croptif(veltif[12], 12) #Velocity magnitude around the site 

##### Routing Data derived from bedmachine, essentially upstream pixels #######

routefile = '../Aidan_datasets/flow_rt.p'
fl = open(routefile, 'rb')
a = pk.load(fl)
fl.close()
flow = a['flow_acc']
xfl = a['x']
yfl = a['y']

##### Get the data into one set #####

xmaxes = [max(xmac), max(xsv), max(xfl)]  ### just making everything the same region
xmins = [min(xmac), min(xsv), min(xfl)]
ymaxes = [max(ymac), max(ysv), max(yfl)]
ymins = [min(ymac), min(ysv), min(yfl)]

thick = clip_array(thick, xmac, ymac, max(xmins), min(xmaxes), max(ymins), min(ymaxes))  #ice thickness
bed  = clip_array(bed, xmac, ymac, max(xmins), min(xmaxes), max(ymins), min(ymaxes)) #bed topography
surface = clip_array(surface, xmac, ymac, max(xmins), min(xmaxes), max(ymins), min(ymaxes)) #bed surface
velmag = clip_array(velmag, xsv, ysv, max(xmins), min(xmaxes), max(ymins), min(ymaxes)) #velocity magnitude
flow = clip_array(flow, xfl, yfl, max(xmins), min(xmaxes), max(ymins), min(ymaxes)) #routing data
xmac2 = clip_array(np.meshgrid(xmac,ymac)[0], xmac, ymac, max(xmins), min(xmaxes), max(ymins), min(ymaxes))[0,:]
ymac2 = clip_array(np.meshgrid(xmac,ymac)[1], xmac, ymac, max(xmins), min(xmaxes), max(ymins), min(ymaxes))[:,0]
                  
xsv = xsv[(xsv > max(xmins))]
xsv = xsv[(xsv < min(xmaxes))]
ysv = ysv[(ysv > max(ymins))]
ysv = ysv[(ysv < min(ymaxes))]

xmacg, ymacg = np.meshgrid(xmac2, ymac2) #meshgrid coordinates
xsvg, ysvg = np.meshgrid(xsv, ysv)

velmag = si.griddata((xsvg.flatten(), ysvg.flatten()), velmag.flatten(), (xmacg, ymacg), method = 'linear') #reproject velocity magnitude onto smaller grid
velmag[(thick == 0)] = 0. #set off ice sheet data to zero
velmag[(np.isnan(velmag))] = 0.

### Now everything should be sampled correctly ####
print(len(xmacg.flatten()),len(velmag.flatten()), len(thick.flatten()))

##### Plot the datasets just to make sure everything looks right ####

plt.figure()
plt.imshow(bed, interpolation = 'none', aspect = 'auto', extent = (0,.15*1085,0,1198*.15))
plt.colorbar(label = 'Bed Elevation (m)')
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.title('Bedrock Elevation Western Greenland')

plt.figure()
plt.imshow(surface, interpolation = 'none', aspect = 'auto', extent = (0,.15*1085,0,1198*.15))
plt.colorbar(label = 'Ice Elevation (m)')
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.title('Surface Elevation Western Greenland')

plt.figure()
plt.imshow(thick, interpolation = 'none', aspect = 'auto', extent = (0,.15*1085,0,1198*.15))
plt.colorbar(label = 'Ice Thickness (m)')
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.title('Ice Thickness Western Greenland')

plt.figure()
plt.imshow(velmag, interpolation = 'none',vmin = 0, vmax = 200, aspect = 'auto', extent = (0,.15*1085,0,1198*.15))
plt.colorbar(label = 'Velocity Magnitude (m/yr)')
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.title('Ice Velocity Magnitude Western Greenland')

plt.figure()
plt.imshow(np.log(flow), interpolation='none', aspect = 'auto', extent = (0,.15*1085,0,1198*.15))
plt.colorbar(label = 'log(upstream pixels)')
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.title('Water Routing Western Greenland')

print(flow.shape) #(1198, 1085) Im goint to make 20 separate images height 59 pixels

### Save the data ###
for i in range(20):
	np.savetxt('surfacepredictordata'+str(i+1)+'.txt', (velmag[i*59:(i*59+59)].flatten(), xmacg[i*59:(i*59+59)].flatten(), ymacg[i*59:(i*59+59)].flatten(), thick[i*59:(i*59+59)].flatten(), bed[i*59:(i*59+59)].flatten(), surface[i*59:(i*59+59)].flatten(), flow[i*59:(i*59+59)].flatten()), delimiter = ',', header = 'Velocity  Xcoord  Ycoord  Ice_Thickness  Bed  Surface  Routing')

### Save all data in one file ###


np.savetxt('surfacepredictordata_all.txt', (velmag.flatten(), xmacg.flatten(), ymacg.flatten(), thick.flatten(), bed.flatten(), surface.flatten(), flow.flatten()), delimiter = ',', header = 'Velocity  Xcoord  Ycoord  Ice_Thickness  Bed  Surface  Routing')

print(max(velmag.flatten()), min(velmag.flatten())) #need to know for later rescaling
plt.show()





