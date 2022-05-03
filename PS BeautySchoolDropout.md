---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

Conducts all necessary preprocessing for PlanetScope (PS) Dove imagery files for the Super Resolution neural network (pulls mask, mosaics images, preps orbit data, and conducts normalization and downsampling).  NOTE: This notebook MUST be run after S2 Pyrite, because that notebook provides the Sentinel-2 input used in this notebook (s2_ref variable) for image co-registration.

```python
import os
import json
import pandas as pd
import geopandas as gpd
import rasterio, rasterio.mask
from rasterio import Affine, merge
from rasterio.enums import Resampling
import numpy as np
import numpy.ma as ma
from arosics import COREG, DESHIFTER
```

```python
# Knocks out all imagery for a single day.  THIS IS THE ONLY BLOCK YOU'LL NEED TO EDIT

# Set date for unique file names
date = '8Jun19_'

# Location of raw Dove images
# If NOT clipped: Set to folder ending with 'PSScene4Band'.  
# If clipped, set to folder 'files'
main_dir = '.../files'

# Sentinel-2 image to use for co-registration (NOT normalized, ...10_stack.tif or 10m_mosaic.tif)
s2_ref = '.../T16TCR_20190608T164849_10_stack.tif'

# Output folder (location of resulting preprocessed imagery -- recommend same location as S2 Pyrite output)
out_folder = '.../'

# Filepath to the shapefile of AOI
geom_fp = '...MyAOI.shp'
#geom_fp = '/home/sarahwegmueller/Documents/Casden_dev/Lakewood_OW/Full_OW_test_right.shp'
```

```python
def file_setup(main_dir):
    '''Gets either the filenames, or folders for each image, depending whether clipping was used'''   
    if main_dir[-5:] == 'files':  #these are clipped images; file system differs
        os.chdir(main_dir)
        filenames = []

        files = os.listdir()
        
        # Get a list of image names:
        for name in files:
            if name.endswith("R_clip.tif"):
                file = name[:-23]
                filenames.append(file)
        
        return filenames
        
    else:
        os.chdir(main_dir)  #These are normally downloaded images

        # Get a list of image names:
        files = os.listdir()
        filenames = []
        for file in files:
            file = file + '_3B'
            filenames.append(file)
            
        # Get a list of image directories:
        new_wrk_dirs = []
        for file in files:
            new_dir = main_dir + '/' + file + '/analytic_sr_udm2'
            new_wrk_dirs.append(new_dir)
            
        return filenames, new_wrk_dirs

    
def create_ps_conf_mask(udm2_filename):
    '''Uses UDM2 file to write a binary mask: 1: clear with confidence > 75; 0: not clear)'''
    
    udm2_img = rasterio.open(udm2_filename)
    conf = udm2_img.read(7)
    mask = conf > 75
    mask_int = mask.astype('uint16')

    return mask_int

        
def rescale_image(scale, image, output_fp):
    '''Upsamples or downsamples, per scale input'''
        
    img = rasterio.open(image)
    profile = img.profile
    t = img.transform
    transform = Affine (t.a*scale, t.b, t.c, t.d, t.e*scale, t.f)
    height = int(img.height / scale)
    width = int(img.width / scale)
    profile.update(transform=transform, driver='GTiff', height=height, width=width)
    
    band_data = img.read([1, 2, 3, 4],
            out_shape=(4, height, width),
            resampling=Resampling.bilinear,
        )
    
    mask_data = img.read([5],
            out_shape=(1, height, width),
        )
    
    strip_data = img.read([6],
            out_shape=(1, height, width),
        )
    
    total_arr = np.concatenate((band_data, mask_data, strip_data), axis=0)
    
    with rasterio.open(output_fp, 'w', **profile) as dst:
        dst.write(total_arr)
    
def do_coreg(ref_img, tgt_img, reg_img_fp): #, mask_fp, reg_mask_fp):
    '''Conduct co-registration on PS clipped imagery using the NIR band -- assumed to be band 4 in images'''
    CR = COREG(ref_img, tgt_img, reg_img_fp,  #the string here is the output name -- can be adjusted
           fmt_out='GTIFF',
           r_b4match=4,  #using the NIR band because its the best one on the Doves
           s_b4match=4,
           ws=(200,200), #100,100
           nodata=(0,0),
          )

    CR.calculate_spatial_shifts()
    CR.correct_shifts()  # Corrects image and then write it to a file
    
    #DESHIFTER(mask_fp, CR.coreg_info, path_out=reg_mask_fp, fmt_out='GTIFF').correct_shifts()
    
def get_sat_gen(metadata_filepath):
    '''Get PS satellite generation from metadata''' 
    
    metadata = json.load(open(metadata_filepath))
    gen = metadata['properties']['instrument']
    sat_id = metadata['properties']['satellite_id']
    
    return gen, sat_id
    
def normalize_bands(image_fp, out_fp):
    '''Normalizes bands using the 1st and 99th percentile. 
    Assumes the mask is the last band unless mask_bool is given.'''
    
    norm_arrays = []
    img = rasterio.open(image_fp)

    # Get the full mask -- values to exclude when calc norm percentages
    mask = img.read(5)
    mask_bool = mask==0
    
    # Create a mask of the border area
    borders = img.read(1)
    borders_bool = borders == 0
        
    for num, band in enumerate(range(1, 5)):
        data = img.read(band)

        tmp = ma.masked_array(data, mask_bool)
        arr = np.zeros((np.shape(data)[0], np.shape(data)[1]), dtype=np.uint16)
        arr = ma.filled(tmp, 0)

        data_clean = arr[arr !=0]
        low = np.percentile(data_clean, 1)
        high = np.percentile(data_clean, 99)

        dif = high-low

        low_arr = np.full(arr.shape, low)
        dif_arr = np.full(arr.shape, dif)

        data_norm = (data - low_arr) / dif_arr

        data_norm_scaled = (data_norm+100) * 10000

        #new_bool = arr==0
        new_bool = data=0

        tmp2 = ma.masked_array(data_norm_scaled, new_bool)
        norm_arr = np.zeros((np.shape(data_norm)[0], np.shape(data_norm)[1]), dtype=np.uint16)
        norm_arr = ma.filled(tmp2, 0)
        
        tmp3 = ma.masked_array(norm_arr, borders_bool)
        norm_arr_final = np.zeros((np.shape(data_norm)[0], np.shape(data_norm)[1]), dtype=np.uint16)
        norm_arr_final = ma.filled(tmp3, 0)
        
        norm_arr_final.astype('uint16')
        
        if num == 0:
            with rasterio.open(out_fp, 'w', **img.profile) as dst:
                dst.write(norm_arr_final, 1)
        else:
            with rasterio.open(out_fp, 'r+', **img.profile) as dst:
                dst.write(norm_arr_final, num+1)
                
        norm_arr_final = None
                
    with rasterio.open(out_fp, 'r+', **img.profile) as dst:
                dst.write(mask, 5)


def append_strip_number(image_fp, num, out_fp):
    '''Appends a band made up of the num variable in place of valid pixels'''
    
    img = rasterio.open(image_fp)
    bands = img.read()
    ex_band = img.read(1)
    borders_bool = ex_band == 0
        
    strip_num = np.full((bands.shape[1], bands.shape[2]), num, dtype=np.uint16)
    
    tmp = ma.masked_array(strip_num, borders_bool)
    strip_final = np.zeros((ex_band.shape[0], ex_band.shape[1]), dtype=np.uint16)
    strip_final = ma.filled(tmp, 0)
    
    strip_final = np.expand_dims(strip_final, axis=0)

    full_array = np.concatenate((bands, strip_final), axis=0)
    
    profile = img.profile
    profile['count'] = 6
    
    with rasterio.open(out_fp, 'w',**profile) as dst:
        dst.write(full_array)
```

```python
# Create dataframe to collect satellite id (for strip number) and mask percentage (mosaic hierarchy)

if main_dir[-5:] == 'files':
    filenames = file_setup(main_dir)
else: 
    filenames, new_wrk_dirs = file_setup(main_dir)       

df = pd.DataFrame(index=range(0, len(filenames)), columns=['img', 'sat_id'])
```

```python
aoi = gpd.read_file(geom_fp)
geom = [aoi['geometry'].iloc[0]]

left = aoi.bounds.iloc[0][0]
bottom = aoi.bounds.iloc[0][1]
right = aoi.bounds.iloc[0][2]
top = aoi.bounds.iloc[0][3]


if main_dir[-5:] == 'files':  
    files_to_remove_after_mosaic = []   
    
    for num, filename in enumerate(filenames):
        print('This is file: ', filename)
        
        #Get sat_id
        meta = filename[:-3] + '_metadata.json'
        gen, sat_id = get_sat_gen(meta)

        # Create mask
        udm2_fp = filename + '_udm2_clip.tif'
        mask = create_ps_conf_mask(udm2_fp)
        mask = np.expand_dims(mask, axis=0)
        
        # Append the mask to the stack
        org_sr = rasterio.open(filename + '_AnalyticMS_SR_clip.tif')
        org_bands = org_sr.read()
        
        profile = org_sr.profile
        profile['count'] = 5
        
        band_mask_stack = np.concatenate((org_bands, mask))
        
        org_bands=None
        mask=None
                
        stack_fp = filename + 'withMask.tif'
        files_to_remove_after_mosaic.append(stack_fp)
        
        with rasterio.open(stack_fp, 'w', **profile) as dst:
            dst.write(band_mask_stack)
        
        band_mask_stack = None
            
        # Populate the dataframe
        df['img'].iloc[num] = stack_fp
        df['sat_id'].iloc[num] = sat_id

    # Mosaic images from the same satellite by order of least masking
    sats = df['sat_id'].unique()

    mosaic_order_df = pd.DataFrame(index=range(0, len(df['sat_id'].unique())), columns=['mosaic', 'masked_ratio']) 
    
    for num, sat in enumerate(sats):
        working_df = df[df['sat_id']==sat]
        
        if len(working_df) > 1: #more than one image     
            # Get the images from the same satellite 
            fps = list(working_df['img'].values)
            # Mosaic them, and crop to AOI
            mosaic_array, mosaic_transform = rasterio.merge.merge(fps, 
                                                                  bounds=(left, bottom, right, top),
                                                                  method='first')
            # Get the masked ratio
            mask = mosaic_array[4,:,:]
            ratio = (len(mask[mask==0])) / len(mask.flatten())
            mask = None
            
            # Use the satellite ID to create a unique filename
            sat_id = working_df['sat_id'].unique()[0]
            
            # Write the mosaic
            profile = rasterio.open(fps[0]).profile
            profile['transform'] = mosaic_transform
            profile['height'] = mosaic_array.shape[1]
            profile['width'] = mosaic_array.shape[2]

            dst_path = sat_id + '_mosaic.tif'

            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(mosaic_array)
                
            mosaic_array = None

            # Register the mosaic to the Sentinel-2 image
            reg_img_fp = out_folder + dst_path[:-4] + '_reg.tif'
            do_coreg(s2_ref, dst_path, reg_img_fp)
            
            os.remove(dst_path)
            
            # Normalize bands to 1st and 99th percentile         
            norm_fp = reg_img_fp[:-4] + 'norm.tif'
            normalize_bands(reg_img_fp, norm_fp)
            
            os.remove(reg_img_fp)
            
            # Record the masked percentage
            mosaic_order_df['mosaic'].iloc[num] = norm_fp
            mosaic_order_df['masked_ratio'].iloc[num] = ratio
            
        else:
            image_fp = working_df['img'].iloc[0]
            
            # Crop the image to the AOI
            img = rasterio.open(image_fp)
            cropped_arr, transform = rasterio.mask.mask(img, geom, crop=True)

            # Write to file
            cropped_image = sat_id + '_cropped_image.tif'
            
            # Update profile
            profile = img.profile
            profile['transform'] = transform
            profile['height'] = cropped_arr.shape[1]
            profile['width'] = cropped_arr.shape[2]

            # Write the image
            with rasterio.open(cropped_image, 'w', **profile) as dst:
                dst.write(cropped_arr)

            # Clear some memory
            cropped_arr = None
            
            # Register the image to the Sentinel-2 image
            reg_img_fp = out_folder + cropped_image[:-4] + '_reg.tif'
            do_coreg(s2_ref, cropped_image, reg_img_fp)
            
            # Normalize bands to 1st and 99th percentile         
            norm_fp = reg_img_fp[:-4] + 'norm.tif'
            normalize_bands(reg_img_fp, norm_fp)
            
            os.remove(cropped_image)
            os.remove(reg_img_fp)
            
            # Find percent masked
            mask_band = rasterio.open(norm_fp).read(5)
            ratio = (len(mask_band[mask_band==0])) / len(mask_band.flatten())
            
            mosaic_order_df['mosaic'].iloc[num] = norm_fp
            mosaic_order_df['masked_ratio'].iloc[num] = ratio           

    for f in files_to_remove_after_mosaic:
        os.remove(f)

    files_to_remove_after_mosaic = []  
    
else:
    files_to_remove_after_mosaic = []
    for num, folder in enumerate(new_wrk_dirs):
        os.chdir(folder)
        filename = filenames[num]
        print(filename)
        
         #Check metadata for satellite generation 
        os.chdir(os.path.abspath(os.path.join(folder, os.pardir)))
        meta = [f for f in os.listdir() if f.endswith('.json')]
        gen, sat_id = get_sat_gen(meta[0])
        
        os.chdir(folder)
        
        # Create mask
        udm2_fp = filename + '_udm2.tif'
        mask_out = filename + '_udm2_mask.tif'
        mask = create_ps_conf_mask(udm2_fp)
        mask = np.expand_dims(mask, axis=0)
        
        # Append the mask to the stack
        org_sr = rasterio.open(filename + '_AnalyticMS_SR.tif')
        org_bands = org_sr.read()
        
        profile = org_sr.profile
        profile['count'] = 5
        
        band_mask_stack = np.concatenate((org_bands, mask))
        
        org_bands=None
        mask=None
                
        stack_fp = filename + 'withMask.tif'
        files_to_remove_after_mosaic.append(folder + '/' + stack_fp)
        
        with rasterio.open(stack_fp, 'w', **profile) as dst:
            dst.write(band_mask_stack)
        
        band_mask_stack = None
               
        # Populate the dataframe
        df['img'].iloc[num] = folder + '/' + stack_fp
        df['sat_id'].iloc[num] = sat_id
        
    # Mosaic images from the same satellite by order of least masking
    sats = df['sat_id'].unique() 
        
    mosaic_order_df = pd.DataFrame(index=range(0, len(df['sat_id'].unique())), columns=['mosaic', 'masked_ratio']) 
    
    for num, sat in enumerate(sats):
        os.chdir(out_folder)
        working_df = df[df['sat_id']==sat]
        
        if len(working_df) > 1: #more than one image     
            # Get the images from the same satellite 
            fps = list(working_df['img'].values)
            # Mosaic them, and crop to AOI
            mosaic_array, mosaic_transform = rasterio.merge.merge(fps, 
                                                                  bounds=(left, bottom, right, top),
                                                                  method='first')            
            
            # Get the masked ratio
            mask = mosaic_array[4,:,:]
            ratio = (len(mask[mask==0])) / len(mask.flatten())
            mask = None
            
            # Use the satellite ID to create a unique filename
            sat_id = working_df['sat_id'].unique()[0]
            
            # Write the mosaic
            profile = rasterio.open(fps[0]).profile
            profile['transform'] = mosaic_transform
            profile['height'] = mosaic_array.shape[1]
            profile['width'] = mosaic_array.shape[2]

            dst_path = sat_id + '_mosaic.tif'

            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(mosaic_array)
                
            mosaic_array = None
                
            # Register the mosaic to the Sentinel-2 image
            reg_img_fp = out_folder + dst_path[:-4] + '_reg.tif'
            do_coreg(s2_ref, dst_path, reg_img_fp)
            
            os.remove(dst_path)
            
            # Normalize bands to 1st and 99th percentile         
            norm_fp = reg_img_fp[:-4] + 'norm.tif'
            normalize_bands(reg_img_fp, norm_fp)
            
            os.remove(reg_img_fp)
            
            # Record the masked percentage
            mosaic_order_df['mosaic'].iloc[num] = norm_fp
            mosaic_order_df['masked_ratio'].iloc[num] = ratio
            
        else:
            image_fp = working_df['img'].iloc[0]
            
            # Crop the image to the AOI
            img = rasterio.open(image_fp)
            cropped_arr, transform = rasterio.mask.mask(img, geom, crop=True)

            # Write to file
            cropped_image = image_fp[:-4]+ sat_id + '_cropped_image.tif'
            
            # Update profile
            profile = img.profile
            profile['transform'] = transform
            profile['height'] = cropped_arr.shape[1]
            profile['width'] = cropped_arr.shape[2]

            # Write the image
            with rasterio.open(cropped_image, 'w', **profile) as dst:
                dst.write(cropped_arr)

            # Clear some memory
            cropped_arr = None
            
            # Register the image to the Sentinel-2 image
            reg_img_fp = cropped_image[:-4] + '_reg.tif'
            do_coreg(s2_ref, cropped_image, reg_img_fp)
            
            # Normalize bands to 1st and 99th percentile         
            norm_fp = reg_img_fp[:-4] + 'norm.tif'
            normalize_bands(reg_img_fp, norm_fp)
            
            os.remove(reg_img_fp)
            
            # Find percent masked
            mask_band = rasterio.open(norm_fp).read(5)
            ratio = (len(mask_band[mask_band==0])) / len(mask_band.flatten())
            
            mosaic_order_df['mosaic'].iloc[num] = norm_fp
            mosaic_order_df['masked_ratio'].iloc[num] = ratio           

    for f in files_to_remove_after_mosaic:
        os.remove(f)

    files_to_remove_after_mosaic = []  

    
# Get mosaic order   
df2 = mosaic_order_df.sort_values(by='masked_ratio').copy()

# Create list of images from df; order to be mosaicked (images with least masking are considered best)
img_order = list(df2['mosaic'].values)

# Get strip numbers; Append a 6th band with strip number
df2['strip_num'] = range(1, len(df2)+1) # set strip numbers
for row in range(len(df2)):  # Add the strip number band
    image_fp = df2['mosaic'].iloc[row]
    num = df2['strip_num'].iloc[row]
    out_fp = image_fp[:-4] + '_strip.tif'
    append_strip_number(image_fp, num, out_fp)
    files_to_remove_after_mosaic.append(out_fp)

for num, img in enumerate(img_order): # Adjust names in the image order
    img_order[num] = img[:-4] + '_strip.tif'

# Create full mosaic ... FINALLY
datasets = []

for img in img_order:
    image = rasterio.open(img)
    datasets.append(image)

dst_path = out_folder + date + 'PS_mosaic.tif'
mosaic_array, mosaic_transform = rasterio.merge.merge(datasets, method='first')

profile = datasets[1].profile
profile['transform'] = mosaic_transform
profile['height'] = mosaic_array.shape[1]
profile['width'] = mosaic_array.shape[2]

with rasterio.open(dst_path, 'w', **profile) as dst:
    dst.write(mosaic_array)
    
# Clear some memory
mosaic_array = None
datasets = None
# Clear working files:
for file in files_to_remove_after_mosaic:
    os.remove(file)
for file in df2['mosaic'].values:
    os.remove(file)

# Resample mosaic to 2.5m resolution
scale = 2.5/3 
img_resamp = dst_path[:-4] + '_resamp.tif'
rescale_image(scale, dst_path, img_resamp)

os.remove(dst_path)
```

```python
#### Downsample images for NN training #####
```

```python
def rescale_dove_image(scale, image, output_fp):
    '''Upsamples or downsamples, per scale input'''
        
    img = rasterio.open(image)
    profile = img.profile
    
    t = img.transform
    transform = Affine (t.a*scale, t.b, t.c, t.d, t.e*scale, t.f)
    height = int(img.height / scale)
    width = int(img.width / scale)
    profile.update(transform=transform, driver='GTiff', height=height, width=width)
    
    vis_bands = img.read([1, 2, 3, 4], out_shape=(4, height, width), resampling=Resampling.cubic)
    aux_bands = img.read([5, 6], out_shape=(2, height, width), resampling=Resampling.nearest)
    
    new_bands = np.concatenate((vis_bands, aux_bands), axis=0)
    vis_bands = None
    aux_bands = None
    
    with rasterio.open(output_fp, 'w', **profile) as dst:
        dst.write(new_bands)
```

```python
# PS --> 10m
ps10m = out_folder + 'dove_10m.tif'
rescale_dove_image(4, img_resamp, ps10m)

# PS --> 20m
ps20m = out_folder + 'dove_20m.tif'
rescale_dove_image(8, img_resamp, ps20m)
```

```python
# Pull the orbits (strips) out of the mosaic

images_to_pull_orbits = [img_resamp, ps10m, ps20m]

corresponding_out_fps = [date+'Dove_mosaic.tif', date+'Dove_10m_mosaic.tif', date+'Dove_20m_mosaic.tif']
orbit_fps = [date+'Dove_Orbits.tif', date+'Dove_Orbits_10m.tif', date+'Dove_Orbits_20m.tif']

for num, mos in enumerate(images_to_pull_orbits):
    
    img = rasterio.open(mos)
    bands = img.read([1, 2, 3, 4, 5])
    bands.shape

    profile = img.profile
    profile['count'] = 5

    new_fp = out_folder + corresponding_out_fps[num]

    # Rewrite image without orbits
    with rasterio.open(new_fp, 'w', **profile) as dst:
        dst.write(bands)

    # Read in orbits and write to separate image (prep for one-hot encoding)
    orbits = img.read(6)

    profile = img.profile
    profile['count'] = 1

    orbit_fp = out_folder + orbit_fps[num]

    with rasterio.open(orbit_fp, 'w', **profile) as dst:
        dst.write(orbits, 1)
```

```python
# Remove working files
for file in images_to_pull_orbits:
    os.remove(file)
```

```python

```
