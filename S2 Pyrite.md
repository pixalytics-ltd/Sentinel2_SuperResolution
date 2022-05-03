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
    display_name: Python (arosics)
    language: python
    name: arosics
---

This code is used to prep the Sentinel-2 imagery for the Super Resolution neural network (stacks bands, pulls mask, crops imagery to desired AOI, and conducts normalization and downsampling). Output is the stacked 10m and 20m bands, respectively, with 10m and 20m masks. The downsampled resolution files are also created. Run this before BeautySchoolDropout because it provides the reference image used by AROSICS to register the PS images to.

NOTE: If it takes more than one Sentinel-2 tile to cover the AOI, the code will mosaic them together BEFORE normalization (doing the normalization before mosaicking them causes spectral differences within the mosaic).  This is all automated.  Just provide the filepaths where directed in block 2 of this notebook.

```python
import os
import numpy as np
import numpy.ma as ma
import rasterio, rasterio.mask
from rasterio import Affine, merge
from rasterio.enums import Resampling
import geopandas as gpd
```

SCL details (used for masking):

Scene Classification (those with * are used for masking below):
- 0 = No Data (missing data)
- 1* = Saturated or defective pixel
- 2 = Dark features/Shadows
- 3* = Cloud shadows
- 4 = Vegetation
- 5 = Not-vegetated
- 6 = Water
- 7 = Unclassified
- 8* = Cloud medium probability
- 9* = Cloud high probability
- 10* = Thin cirrus
- 11* = Snow or ice

```python
### This is the only block you will need to edit ###

# Paths to Sentinel-2 images -- 'IMG_DATA'           
s2_imgs = ['.../IMG_DATA',
          ]

# Path to shapefile to crop image to
geom_fp = '.../MyAOI.shp'

# Path to folder for output:
out_folder = '.../'
```

```python
def get_image_name(image_directory):
    file = image_directory + '/R10m'        
    names = os.listdir(file)[0]
    img_name = names[:22]
    return img_name

def untile_image(img_fp):
    '''
    Removed tiling from image to allow for resamling to match imagery
    '''
    img = rasterio.open(img_fp)
    img_array = img.read(1)    
    untiled_fp = '%s_untiled.tif' %img_fp[:-4]

    with rasterio.Env():
        profile = img.profile
        profile['tiled']=False
        profile['driver']='GTiff'
        with rasterio.open(untiled_fp, 'w', **profile) as dst:
            dst.write(img_array, 1)

    return untiled_fp

def rescale_mask (scale, image, output_fp): 

    img = rasterio.open(image)
    profile = img.profile
    
    t = img.transform
    transform = Affine (t.a*scale, t.b, t.c, t.d, t.e*scale, t.f)
    height = int(img.height / scale)
    width = int(img.width / scale)
    profile.update(transform=transform, driver='GTiff', height=height, width=width)
    
    mask = img.read(out_shape=(1, height, width), resampling=Resampling.nearest)
        
    with rasterio.open(output_fp, 'w', **profile) as dst:
        dst.write(mask)
                

def normalize_bands(image_fp):
    '''Normalizes bands using the 1st and 99th percentile. 
    Assumes the mask is the last band.'''
    
    norm_arrays = []
    img = rasterio.open(image_fp)

    mask = img.read(img.count)
    mask_bool = mask==0
        
    for band in range(1, img.count):
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
        
        norm_arrays.append(norm_arr)
    
    norm_arrays.append(mask)
    
    normed_array = np.array(norm_arrays, dtype=np.uint16)
        
    return normed_array

def rescale_S2_10m_image(scale, image, output_fp):
    '''Upsamples or downsamples, per scale input'''
        
    img = rasterio.open(image)
    profile = img.profile
    
    t = img.transform
    transform = Affine (t.a*scale, t.b, t.c, t.d, t.e*scale, t.f)
    height = int(img.height / scale)
    width = int(img.width / scale)
    profile.update(transform=transform, driver='GTiff', height=height, width=width)
    
    vis_bands = img.read([1, 2, 3, 4], out_shape=(4, height, width), resampling=Resampling.cubic)
    mask = img.read([5], out_shape=(1, height, width), resampling=Resampling.nearest)
    
    new_bands = np.concatenate((vis_bands, mask), axis=0)
    vis_bands = None
    mask = None
    
    with rasterio.open(output_fp, 'w', **profile) as dst:
        dst.write(new_bands)
        
def rescale_S2_20m_image(scale, image, output_fp):
    '''Upsamples or downsamples, per scale input'''
        
    img = rasterio.open(image)
    profile = img.profile
    
    t = img.transform
    transform = Affine (t.a*scale, t.b, t.c, t.d, t.e*scale, t.f)
    height = int(img.height / scale)
    width = int(img.width / scale)
    profile.update(transform=transform, driver='GTiff', height=height, width=width)
    
    vis_bands = img.read([1, 2, 3, 4, 5, 6], out_shape=(6, height, width), resampling=Resampling.cubic)
    mask = img.read([7], out_shape=(1, height, width), resampling=Resampling.nearest)
    
    new_bands = np.concatenate((vis_bands, mask), axis=0)
    vis_bands = None
    mask = None
        
    with rasterio.open(output_fp, 'w', **profile) as dst:
        dst.write(new_bands)      
```

```python
all_dates = []
mosaic_sets = []

for fp in s2_imgs:
    date = fp[-24:-16]
    all_dates.append(date)

dates = dict.fromkeys(set(all_dates))
dates
```

```python
for date in dates:
    print(date)
    fp_list = []
    for fp in s2_imgs:
        if date in fp:
            fp_list.append(fp)
        else:
            continue
    dates[date]=fp_list
```

```python
aoi = gpd.read_file(geom_fp)
geom = [aoi['geometry'].iloc[0]]

for date in dates:
    print("Date ", date)
    if len(dates[date])>1:
        print('These will be mosaicked: ', dates[date])
        fps = dates[date]
        fps_for_10m_mosaic = []  
        fps_for_20m_mosaic = []  
        
        for fp in fps:
            
            os.chdir(fp)

            img_identifier = get_image_name(fp)
            print('Image: ', img_identifier)

            # Masking band and reference for rescaling
            qi = 'R20m/%s_SCL_20m.jp2'%img_identifier 
            ref_img = 'R10m/%s_B02_10m.jp2' %img_identifier 

            # Create masks at 10m and 20m
            qi = rasterio.open(qi)

            # Crop to AOI
            mask_array, mask_transform = rasterio.mask.mask(qi, geom, crop=True)

            #Remove first dimension introduced in the last line
            mask_array = np.squeeze(mask_array, axis=0)

            # Mask clouds, shadows, and bad pixels
            qa_mask = (mask_array==1)|(mask_array==3)|(mask_array==8)|(mask_array==9)|(mask_array==10)|(mask_array==11)

            # Invert such that 0 = masked areas
            qa_mask = np.invert(qa_mask)

            # Convert to integar array
            twenty_mask_int = qa_mask.astype('uint16')

            # Write mask to file
            kwds = qi.profile
            kwds['transform'] = mask_transform
            kwds['height'] = mask_array.shape[0]
            kwds['width'] = mask_array.shape[1]
            kwds['driver'] = 'GTiff'
            kwds['dtype'] = 'uint16'

            with rasterio.open('Mask_20m.tif', 'w', **kwds) as dst:
                 dst.write(twenty_mask_int, 1)

            # Rescale the mask to match 10m, then write to file
            mask_10m_fp = 'Resampled_Mask_10m.tif'
            mask_10m = rescale_mask(.5, 'Mask_20m.tif', mask_10m_fp) 
            
            os.remove('Mask_20m.tif')           
            
            # Crop to geom to ensure the same array shape
            mask_to_crop = rasterio.open(mask_10m_fp)
            cropped_10m_mask, cropped_10m_transform = rasterio.mask.mask(mask_to_crop, geom, crop=True)
            
            os.remove(mask_10m_fp)

            # Create 20m and 10m stacks; mask bands using above masks   
            twenty_stack_list = ['R20m/%s_B05_20m.jp2' %img_identifier,
                                 'R20m/%s_B06_20m.jp2' %img_identifier,
                                 'R20m/%s_B07_20m.jp2' %img_identifier,
                                 'R20m/%s_B8A_20m.jp2' %img_identifier,
                                 'R20m/%s_B11_20m.jp2' %img_identifier, 
                                 'R20m/%s_B12_20m.jp2' %img_identifier]

            ten_stack_list = ['R10m/%s_B02_10m.jp2' %img_identifier,
                              'R10m/%s_B03_10m.jp2' %img_identifier,
                              'R10m/%s_B04_10m.jp2' %img_identifier,
                              'R10m/%s_B08_10m.jp2' %img_identifier]

            # Write 20m stack    
            untiled_20m_images = []

            for image in twenty_stack_list:
                img = rasterio.open(image)
                if img.profile['tiled'] == True:
                    untiled_img_fp = untile_image(image)
                    untiled_20m_images.append(untiled_img_fp)

            if len(untiled_20m_images) != 0:
                twenty_stack_list = untiled_20m_images

            with rasterio.open(twenty_stack_list[0]) as src20:
                profile_20 = src20.profile

            # Update meta to reflect the number of layers
            profile_20['count'] = len(twenty_stack_list)+1
            profile_20['tiled'] = False
            profile_20['nodata'] = 0

            twenty_stack = out_folder + '%s_20_stack.tif' %img_identifier
            
            all_20_arrs = []

            for num, image in enumerate(twenty_stack_list):
                print('Cropping and Writing 20m band ', num+1)
                org = rasterio.open(image)
                org_band = org.read(1)

                # Crop to AOI
                arr, transform = rasterio.mask.mask(org, geom, crop=True)

                all_20_arrs.append(arr)

            twenty_mask_int = np.expand_dims(twenty_mask_int, axis=0)
            all_20_arrs.append(twenty_mask_int)
            twenty_arrs = np.array(all_20_arrs)
            twenty_arrs = np.squeeze(twenty_arrs, axis=1)
            
            # Update profile
            profile_20['transform'] = transform
            profile_20['height'] = arr.shape[1]
            profile_20['width'] = arr.shape[2]
            
            # Write
            with rasterio.open(twenty_stack, 'w', **profile_20) as dst:
                dst.write(twenty_arrs)
            
            fps_for_20m_mosaic.append(twenty_stack)
            
            all_20_arrs = None
            twenty_arrs = None

            # Write 10m stack
            untiled_10m_images = []

            for image in ten_stack_list:
                img = rasterio.open(image)
                if img.profile['tiled'] == True:
                    untiled_img_fp = untile_image(image)
                    untiled_10m_images.append(untiled_img_fp)

            if len(untiled_10m_images) != 0:
                ten_stack_list = untiled_10m_images

            with rasterio.open(ten_stack_list[0]) as src10:
                profile_10 = src10.profile
                profile_10['tiled'] = False
                profile_10['nodata'] = 0

            # Update meta to reflect the number of layers
            profile_10['count'] = len(ten_stack_list)+1

            ten_stack = out_folder + '%s_10_stack.tif' %img_identifier
            
            all_10_arrs = []

            for num, image in enumerate(ten_stack_list):
                print('Cropping and writing 10m band ', num+1)
                org = rasterio.open(image)
                org_band = org.read(1)

                # Crop to AOI
                arr, transform = rasterio.mask.mask(org, geom, crop=True)

                all_10_arrs.append(arr)

            all_10_arrs.append(cropped_10m_mask)
            ten_arrs = np.array(all_10_arrs)
            ten_arrs = np.squeeze(ten_arrs, axis=1)            
            
            # Update the profile
            profile_10['transform'] = transform
            profile_10['height'] = arr.shape[1]
            profile_10['width'] = arr.shape[2]

            # Write               
            with rasterio.open(ten_stack, 'w', **profile_10) as dst:
                dst.write(ten_arrs)

            fps_for_10m_mosaic.append(ten_stack)
                
            all_20_arrs = None
            ten_arrs = None

        # Mosaic all files, then normalize
        # Write the 10m mosaic
        mosaic_10m, mosaic_10m_transform = rasterio.merge.merge(fps_for_10m_mosaic)

        profile_10m_mosaic = rasterio.open(fps_for_10m_mosaic[0]).profile
        profile_10m_mosaic['transform'] = mosaic_10m_transform
        profile_10m_mosaic['height'] = mosaic_10m.shape[1]
        profile_10m_mosaic['width'] = mosaic_10m.shape[2]

        ten_m_mosaic = out_folder + date + '_10m_mosaic.tif'

        with rasterio.open(ten_m_mosaic, 'w', **profile_10m_mosaic) as dst:
            dst.write(mosaic_10m)

        mosaic_10m = None
        
        for fp in fps_for_10m_mosaic:
            os.remove(fp)
        
        # Write the 20m mosaic
        mosaic_20m, mosaic_20m_transform = rasterio.merge.merge(fps_for_20m_mosaic)

        profile_20m_mosaic = rasterio.open(fps_for_20m_mosaic[0]).profile
        profile_20m_mosaic['transform'] = mosaic_20m_transform
        profile_20m_mosaic['height'] = mosaic_20m.shape[1]
        profile_20m_mosaic['width'] = mosaic_20m.shape[2]

        twenty_m_mosaic = out_folder + date + '_20m_mosaic.tif'

        with rasterio.open(twenty_m_mosaic, 'w', **profile_20m_mosaic) as dst:
            dst.write(mosaic_20m)

        mosaic_20m = None
        
        for fp in fps_for_20m_mosaic:
            os.remove(fp)
            
        # Normalize the bands, stack with mask, save concatenated image to file
        norm_10m = out_folder + '%s_10_stack_norm.tif' %date
        norm_10_array = normalize_bands(ten_m_mosaic)

        with rasterio.open(norm_10m, 'w', **profile_10m_mosaic) as dst:
            dst.write(norm_10_array)

        norm_20m = out_folder + '%s_20_stack_norm.tif' %date
        norm_20_array = normalize_bands(twenty_m_mosaic)

        with rasterio.open(norm_20m, 'w', **profile_20m_mosaic) as dst:
            dst.write(norm_20_array)

        # Remove created files
        if len(untiled_10m_images) != 0:
            for file in untiled_10m_images:
                os.remove(file)

        if len(untiled_20m_images) != 0:
            for file in untiled_20m_images:
                os.remove(file)

        ### Downsample the images ###
        # S2 10m --> 40, 80
        s2_10m = out_folder + '%s_10_stack_norm.tif' %date
        s2_40m = out_folder + '%s_10to40_stack_norm.tif' %date
        s2_80m = out_folder + '%s_10to80_stack_norm.tif' %date

        rescale_S2_10m_image(4, s2_10m, s2_40m)
        rescale_S2_10m_image(8, s2_10m, s2_80m)

        # S2 20m --> 80, 160
        s2_20m = out_folder + '%s_20_stack_norm.tif' %date
        s2_80m = out_folder + '%s_20to80_stack_norm.tif' %date
        s2_160m = out_folder + '%s_20to160_stack_norm.tif' %date

        rescale_S2_20m_image(4, s2_20m, s2_80m)
        rescale_S2_20m_image(8, s2_20m, s2_160m)

    
    else:
        print(date)
        fp = dates[date][0]
        os.chdir(fp)

        img_identifier = get_image_name(fp)
        print('Image: ', img_identifier)

        # Masking band and reference for rescaling
        qi = 'R20m/%s_SCL_20m.jp2'%img_identifier 
        ref_img = 'R10m/%s_B02_10m.jp2' %img_identifier 

        # Create masks at 10m and 20m
        qi = rasterio.open(qi)

        # Crop to AOI
        mask_array, mask_transform = rasterio.mask.mask(qi, geom, crop=True)

        #Remove first dimension introduced in the last line
        mask_array = np.squeeze(mask_array, axis=0)

        # Mask clouds, shadows, and bad pixels
        qa_mask = (mask_array==1)|(mask_array==3)|(mask_array==8)|(mask_array==9)|(mask_array==10)|(mask_array==11)

        # Invert such that 0 = masked areas
        qa_mask = np.invert(qa_mask)

        # Convert to integar array
        twenty_mask_int = qa_mask.astype('uint16')

        # Write mask to file
        kwds = qi.profile
        kwds['transform'] = mask_transform
        kwds['height'] = mask_array.shape[0]
        kwds['width'] = mask_array.shape[1]
        kwds['driver'] = 'GTiff'
        kwds['dtype'] = 'uint16'

        with rasterio.open('Mask_20m.tif', 'w', **kwds) as dst:
             dst.write(twenty_mask_int, 1)

        # Rescale the mask to match 10m, then write to file
        mask_10m_fp = 'Resampled_Mask_10m.tif'
        mask_10m = rescale_mask(.5, 'Mask_20m.tif', mask_10m_fp) 

        os.remove('Mask_20m.tif')           

        # Crop to geom to ensure the same array shape
        mask_to_crop = rasterio.open(mask_10m_fp)
        cropped_10m_mask, cropped_10m_transform = rasterio.mask.mask(mask_to_crop, geom, crop=True)

        os.remove(mask_10m_fp)

        # Create 20m and 10m stacks; mask bands using above masks   
        twenty_stack_list = ['R20m/%s_B05_20m.jp2' %img_identifier,
                             'R20m/%s_B06_20m.jp2' %img_identifier,
                             'R20m/%s_B07_20m.jp2' %img_identifier,
                             'R20m/%s_B8A_20m.jp2' %img_identifier,
                             'R20m/%s_B11_20m.jp2' %img_identifier, 
                             'R20m/%s_B12_20m.jp2' %img_identifier]

        ten_stack_list = ['R10m/%s_B02_10m.jp2' %img_identifier,
                          'R10m/%s_B03_10m.jp2' %img_identifier,
                          'R10m/%s_B04_10m.jp2' %img_identifier,
                          'R10m/%s_B08_10m.jp2' %img_identifier]

        # Write 20m stack    
        untiled_20m_images = []

        for image in twenty_stack_list:
            img = rasterio.open(image)
            if img.profile['tiled'] == True:
                untiled_img_fp = untile_image(image)
                untiled_20m_images.append(untiled_img_fp)

        if len(untiled_20m_images) != 0:
            twenty_stack_list = untiled_20m_images

        with rasterio.open(twenty_stack_list[0]) as src20:
            profile_20 = src20.profile

        # Update meta to reflect the number of layers
        profile_20['count'] = len(twenty_stack_list)+1
        profile_20['tiled'] = False
        profile_20['nodata'] = 0

        #twenty_stack = 'R20m/%s_20_stack_with_mask.tif' %img_identifier
        twenty_stack = out_folder + '%s_20_stack_with_mask.tif' %img_identifier

        for num, image in enumerate(twenty_stack_list):
            print('Cropping and Writing 20m band ', num+1)
            org = rasterio.open(image)
            org_band = org.read(1)

            # Crop to AOI
            arr, transform = rasterio.mask.mask(org, geom, crop=True)

            #Remove first dimension introduced in the last line
            arr = np.squeeze(arr, axis=0)

            # Update profile
            profile_20['transform'] = transform
            profile_20['height'] = arr.shape[0]
            profile_20['width'] = arr.shape[1]

            if num == 0:
                with rasterio.open(twenty_stack, 'w', **profile_20) as dst:
                    dst.write(arr, num+1)
            else:
                with rasterio.open(twenty_stack, 'r+', **profile_20) as dst:
                    dst.write(arr, num+1)

        # Write the mask as the last band
        with rasterio.open(twenty_stack, 'r+', **profile_20) as dst:
            dst.write(twenty_mask_int, profile_20['count'])

        # Write 10m stack
        untiled_10m_images = []

        for image in ten_stack_list:
            img = rasterio.open(image)
            if img.profile['tiled'] == True:
                untiled_img_fp = untile_image(image)
                untiled_10m_images.append(untiled_img_fp)

        if len(untiled_10m_images) != 0:
            ten_stack_list = untiled_10m_images

        with rasterio.open(ten_stack_list[0]) as src10:
            profile_10 = src10.profile
            profile_10['tiled'] = False
            profile_10['nodata'] = 0

        # Update meta to reflect the number of layers
        profile_10['count'] = len(ten_stack_list)+1

        #ten_stack = 'R10m/%s_10_stack_with_mask.tif' %img_identifier
        ten_stack = out_folder + '%s_10_stack_with_mask.tif' %img_identifier

        for num, image in enumerate(ten_stack_list):
            print('Cropping and writing 10m band ', num+1)
            org = rasterio.open(image)
            org_band = org.read(1)

            # Crop to AOI
            arr, transform = rasterio.mask.mask(org, geom, crop=True)

            #Remove first dimension introduced in the last line
            arr = np.squeeze(arr, axis=0)

            # Update the profile
            profile_10['transform'] = transform
            profile_10['height'] = arr.shape[0]
            profile_10['width'] = arr.shape[1]

            # Write               
            if num == 0:
                with rasterio.open(ten_stack, 'w', **profile_10) as dst:
                    dst.write(arr, num+1)
            else:
                with rasterio.open(ten_stack, 'r+', **profile_10) as dst:
                    dst.write(arr, num+1)  

        # Write the mask on its own using transform from stack
        mask_10_profile = profile_10.copy()
        mask_10_profile['count'] = 1

        with rasterio.open(mask_10m_fp, 'w', **mask_10_profile) as dst:
            dst.write(cropped_10m_mask)

        mask_10m_2d = rasterio.open(mask_10m_fp).read(1)  #this is necessary becuase numpy shapes

        with rasterio.open(ten_stack, 'r+', **profile_10) as dst:
            dst.write(mask_10m_2d, profile_10['count'])

        # Normalize the bands, stack with mask, save concatenated image to file
        norm_10m = out_folder + '%s_10_stack_norm.tif' %img_identifier
        norm_10_array = normalize_bands(ten_stack)

        with rasterio.open(norm_10m, 'w', **profile_10) as dst:
            dst.write(norm_10_array)

        norm_20m = out_folder + '%s_20_stack_norm.tif' %img_identifier
        norm_20_array = normalize_bands(twenty_stack)

        with rasterio.open(norm_20m, 'w', **profile_20) as dst:
            dst.write(norm_20_array)

        # Remove the mask from the orginal stacks for use with BeautySchoolDropout
        org_stack = rasterio.open(ten_stack)
        bands = org_stack.read()
        vis_bands = bands[:-1, :, :]
        profile = org_stack.profile
        profile['count'] = org_stack.count-1

        stack_10m_sansMask = out_folder + '%s_10_stack.tif' %img_identifier

        with rasterio.open(stack_10m_sansMask, 'w', **profile) as dst:
            dst.write(vis_bands)

        org_stack = rasterio.open(twenty_stack)
        bands = org_stack.read()
        vis_bands = bands[:-1, :, :]
        profile = org_stack.profile
        profile['count'] = org_stack.count-1

        stack_20m_sansMask = out_folder + '%s_20_stack.tif' %img_identifier

        with rasterio.open(stack_20m_sansMask, 'w', **profile) as dst:
            dst.write(vis_bands)

        # Remove created files
        if len(untiled_10m_images) != 0:
            for file in untiled_10m_images:
                os.remove(file)

        if len(untiled_20m_images) != 0:
            for file in untiled_20m_images:
                os.remove(file)

        os.remove(ten_stack)
        os.remove(twenty_stack)

        ### Downsample the images ###
        # S2 10m --> 40, 80
        s2_10m = out_folder + '%s_10_stack_norm.tif' %img_identifier
        s2_40m = out_folder + '%s_10to40_stack_norm.tif' %img_identifier
        s2_80m = out_folder + '%s_10to80_stack_norm.tif' %img_identifier

        rescale_S2_10m_image(4, s2_10m, s2_40m)
        rescale_S2_10m_image(8, s2_10m, s2_80m)

        # S2 20m --> 80, 160
        s2_20m = out_folder + '%s_20_stack_norm.tif' %img_identifier
        s2_80m = out_folder + '%s_20to80_stack_norm.tif' %img_identifier
        s2_160m = out_folder + '%s_20to160_stack_norm.tif' %img_identifier

        rescale_S2_20m_image(4, s2_20m, s2_80m)
        rescale_S2_20m_image(8, s2_20m, s2_160m)
 
```

```python

```
