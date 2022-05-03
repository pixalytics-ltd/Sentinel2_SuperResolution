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

This notebook is built to use the trained RCCN models to predict images at 2.5m resolution. You will need to run this notebook twice -- once to predict the 10m bands, and once to predict the 20m bands (ensure you ONLY run the block you want to predict below; block 4 or 5).

Only blocks 2 and 3 will need to be edited by the user to provide the necessary inputs (and possibly block 8).

I had to restart the kernel after each run. This is why the prediction of both wasn't built into a single notebook run. I've tried to be as efficent with the memory as possible, but depending on your RAM, you may still have problem (dev'd with an NVIDIA Corporation TU104BM [GeForce RTX 2080 Mobile], Intel® Core™ i7-9700K CPU @ 3.60GHz × 8; 32GB RAM)

Near the bottome, there is an option to look at the resulting image before writing it to a file. Its commented out right now, but the first time you use this, I recommend using this to check your image.

```python
import os
import numpy as np
import rasterio
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,MaxPool2D ,Convolution2D , Add, Dense , AveragePooling2D , UpSampling2D , Reshape , Flatten , Subtract , Concatenate, Cropping2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as k
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import tensorflow.keras.utils as ku

from sklearn.model_selection import train_test_split

%load_ext tensorboard

import shutil
import matplotlib.pyplot as plt
```

```python
# Change the directory to the location of the outputs by Pyrite (S2 imagery) and BeautySchoolDropout (PS imagery)
os.chdir('...')
```

```python
# This date (or file identifier) should match what was used in BeautySchoolDropout
date = '17Jun21_'

# These two files correspond to the Pyrite output.  Choose the files with the EXACT SAME ratio (10 and 20)
s210m_10_fn = "T16TCR_20210617T164839_10_stack_norm.tif"  # bands
s220m_20_fn = "T16TCR_20210617T164839_20_stack_norm.tif" # bands

# If you gave the same identifier above as used in BeautySchoolDropout, this should run without further change
dove10_fn = "%sDove_mosaic.tif" %date # 4 bands + mask
dove10orb_fn = "%sDove_Orbits.tif" %date  # orbit(strip number)

s210m_10_org = rasterio.open(s210m_10_fn).read().T
s220m_20_org = rasterio.open(s220m_20_fn).read().T
dove_img = rasterio.open(dove10_fn) # This will be used to build the profile for the resulting image
dove10_org = dove_img.read().T
dove10_orbits_org = rasterio.open(dove10orb_fn).read().T
```

```python
##### RUN TO PREDICT 10m BANDS ######
trained_model = 'finals/%s_trained_10m_model' %date # Filepath of trained model
image_result_out_fp = '%sDove_10mBands_predict.tif' %date # File name for resulting predicted image
```

```python
##### RUN TO PREDICT 20m BANDS ######
trained_model = 'finals/%s_trained_20m_model' %date# Folder with  trained model
image_result_out_fp = '%sDove_20mBands_predict.tif' %date # File name for resulting predicted image
```

```python
# Pull the profile for use later
profile = dove_img.profile
profile
```

```python
encoded = ku.to_categorical(dove10_orbits_org, dtype='uint16')
print(encoded.shape)

dove10_with_orb_org = np.concatenate((dove10_org, encoded),axis=-1)
print(dove10_with_orb_org.shape)

encoded=None
dove10_org = None
dove_orbits_org = None
dove_img = None
```

```python
# Check for shape symetry and pad with zeros as needed
model_images = [s210m_10_org, s220m_20_org, dove10_with_orb_org]

for num, img in enumerate(model_images):   
    print('Checking image ', num+1)    
    if img.shape[0] != img.shape[1]:
        if img.shape[0] - img.shape[1] <= 3 and img.shape[0] - img.shape[1] > 0:
            factor = img.shape[0] - img.shape[1]
            print('%s more pixels in height' %factor)
            model_images[num] = np.pad(img, ((0, 0), (0, factor), (0, 0)))
            print('Padded with zeros. New shape: ', model_images[num].shape)
        elif img.shape[1] - img.shape[0] <= 3 and img.shape[1] - img.shape[0] > 0:
            factor = img.shape[1] - img.shape[0]        
            print('%s more pixels in width' %factor)
            model_images[num] = np.pad(img, ((0, factor), (0, 0), (0, 0)))
            print('Padded with zeros. New shape: ', model_images[num].shape)
    else:
        print('Shape is good: ', img.shape)
```

```python
# Check for compatibility among images
mod_res_compare_size = model_images[0].shape[0]*4

# Sometimes the above restriction causes problems.  The below line can be used to manual set the value.
#mod_res_compare_size = 16000

if model_images[2].shape[0] < mod_res_compare_size:
    print('Fixing size compatibility. ')
    diff = mod_res_compare_size - model_images[2].shape[0]
    if diff < 5:
        model_images[2] = np.pad(model_images[2], ((0, 0), (0, diff), (0, 0)))
        model_images[2] = np.pad(model_images[2], ((0, diff), (0, 0), (0, 0)))
        print('Padded with zeros. New shape: ', model_images[2].shape)
    else:
        print('Image size compatibility error.')
```

```python
# Reassign variables
s210m_40_org = model_images[0]
s220m_80_org = model_images[1]
dove10_with_orb_org = model_images[2]

model_images = None
```

```python
def shLoss(y_true, y_pred, delta=2.0):
    diff = y_true-y_pred
    dsq = tf.keras.backend.square(delta)
    return tf.keras.backend.mean( dsq * (tf.sqrt(1+ tf.square(diff)/dsq)-1), axis=-1)


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


def PSNRLoss(y_true, y_pred):
        return 10* k.log(255**2 /(k.mean(k.square(y_pred - y_true))))


class SRResnet:
    def L1_loss(self , y_true , y_pred):
        return k.mean(k.abs(y_true - y_pred))
    
    def RDBlocks(self, x, name, count = 6, filter_count=32, RDB_feat=64):
        ## 6 layers of RDB block
        ## this thing need to be in a damn loop for more customisability
        li = [x]
        pas = Convolution2D(filters=filter_count, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)

        for i in range(2 , count+1):
            li.append(pas)
            out =  Concatenate(axis = -1)(li) # conctenated output self.channel_axis
            pas = Convolution2D(filters=filter_count, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)

        li.append(pas)
        out1 = Concatenate(axis = -1)(li) #self.channel_axis
        feat = Convolution2D(filters=RDB_feat, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out1)

        feat = Add()([feat , x])
        print("RDBlocks",feat)
        return feat
        
    def visualize(self):
        plot_model(self.model, to_file='model.png' , show_shapes = True)
    
    def get_model(self):
        return self.model
    
    def Block_Of_RDBBlocks(self, inp, RDB_count=20, count=6, filter_count=32, RDB_feat=64, end_feat=64):
        
        pass1 = Convolution2D(filters=RDB_feat, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(inp)

        pass2 = Convolution2D(filters=RDB_feat, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu')(pass1)


        RDB = self.RDBlocks(pass2 , 'RDB1', count=count, filter_count=filter_count, RDB_feat=RDB_feat)
        RDBlocks_list = [RDB,]
        for i in range(2,RDB_count+1):
            RDB = self.RDBlocks(RDB ,'RDB'+str(i), count=count, filter_count=filter_count, RDB_feat=RDB_feat)
            RDBlocks_list.append(RDB)
        out = Concatenate(axis = -1)(RDBlocks_list) #self.channel_axis
        out = Convolution2D(filters=RDB_feat, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(out)
        output = Add()([out, pass1])
        output = Convolution2D(filters=end_feat, kernel_size=(3,3), strides=(1,1), padding='same', name="rdb_out")(output)
        
        return output


    def __init__(self, s10img, s20img, psimg, lr=0.00005, patch_size=32, RDB_count=4, count=2, filter_count=64, RDB_feat=128, end_feat=128, chk = -1, scale = 4):
        self.channel_axis = 3
        inp10 = Input(shape = (s10img.shape[1], s10img.shape[2], s10img.shape[3]))   # (24,24,4)
        inp20 = Input(shape = (s20img.shape[1], s20img.shape[2], s20img.shape[3]))   # (12,12,6)
        inpPS = Input(shape = (psimg.shape[1], psimg.shape[2], psimg.shape[3]))   # (96,96,9)
        print(psimg.shape)
        print(s10img.shape)
        print(s20img.shape)
#         print(psorb.shape)
#         ps = tf.keras.layers.Concatenate(axis=2)([psimg, psorb])

        Subpixel_scale8 = Lambda(lambda x:tf.nn.depth_to_space(x,8))
        Subpixel_scale4 = Lambda(lambda x:tf.nn.depth_to_space(x,4))
        
        s220c = Convolution2D(filters = (s20img.shape[3])*8*8*mlt, kernel_size=1, strides=1, padding='valid')(inp20)
        print("S220c shape is", s220c.shape)
        s220s = Subpixel_scale8(inputs=s220c)
        print("S220s shape is", s220s.shape)
        m20 = Model(inputs=inp20, outputs=s220s)
        s210c = Convolution2D(filters = (s10img.shape[3])*4*4*mlt, kernel_size=1, strides=1, padding='valid')(inp10)
        print("S210c shape is", s210c.shape)
        s210s = Subpixel_scale4(inputs=s210c)
        print("S210s shape is", s210s.shape)
        m10 = Model(inputs=inp10, outputs=s210s)
        
        all_inp = Concatenate(axis=-1)([m10.output, m20.output, inpPS])
        print("all_inp shape is", all_inp.shape)
        allb = self.Block_Of_RDBBlocks(all_inp, RDB_count, count, filter_count, RDB_feat, end_feat)
        print("BofRDBlocks",allb)
        allm = Cropping2D(cropping=16)(allb)
        print("Cropping",allm)
        allrc = Convolution2D(filters = 6, kernel_size = 1, strides = 1, padding = "valid", activation = None)(allm)
        
        print("Output", allrc)
        model = Model(inputs=[m10.input, m20.input, inpPS], outputs = allrc)
#         print([n.input_tensors.name for n in model.get_layer('A_3').inbound_nodes])
        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, decay=0, amsgrad=False)

        model.compile(loss=shLoss, optimizer='adam' , metrics=['mae'])

        if chk >=0 :
            print("loading existing weights !!!")
            model.load_weights('final.h5')
        self.model = model

            
    def fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_data, steps_per_epoch):   
        hist = self.model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_data=validation_data, steps_per_epoch=steps_per_epoch)
        return hist.history

```

```python
def get_test_patches_orig(dset_10m, dset_20m, ps, patchSize=24, border=4, ps_patch=96, ps_border=16):

    PATCH_SIZE_HR = (patchSize, patchSize)
    PATCH_SIZE_LR = [p//2 for p in PATCH_SIZE_HR]
    BORDER_HR = border
    BORDER_LR = BORDER_HR//2
    PATCH_SIZE_PS = (ps_patch, ps_patch)
    

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(dset_10m, ((BORDER_HR, BORDER_HR), (BORDER_HR, BORDER_HR), (0, 0)))
    dset_20 = np.pad(dset_20m, ((BORDER_LR, BORDER_LR), (BORDER_LR, BORDER_LR), (0, 0)))
    dset_ps = np.pad(ps, ((ps_border, ps_border), (ps_border, ps_border), (0, 0))) 

    BANDS10 = dset_10.shape[2]
    BANDS20 = dset_20.shape[2]
    BANDSps = dset_ps.shape[2]
    patchesAlongi = (dset_20.shape[0] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    patchesAlongj = (dset_20.shape[1] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    #nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)
    nr_patches = patchesAlongi*patchesAlongj

    image_20 = np.zeros((nr_patches, BANDS20) + tuple(PATCH_SIZE_LR)).astype(np.float32)
    image_10 = np.zeros((nr_patches, BANDS10) + PATCH_SIZE_HR).astype(np.float32)
    image_ps = np.zeros((nr_patches, BANDSps) + PATCH_SIZE_PS).astype(np.float32)

    range_i = np.arange(0, (dset_20.shape[0] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)) * (
        PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    range_j = np.arange(0, (dset_20.shape[1] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)) * (
        PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    if not (np.mod(dset_20.shape[0] - 2 * BORDER_LR, PATCH_SIZE_LR[0] - 2 * BORDER_LR) == 0):
        range_i = np.append(range_i, (dset_20.shape[0] - PATCH_SIZE_LR[0]))
    if not (np.mod(dset_20.shape[1] - 2 * BORDER_LR, PATCH_SIZE_LR[1] - 2 * BORDER_LR) == 0):
        range_j = np.append(range_j, (dset_20.shape[1] - PATCH_SIZE_LR[1]))

    pCount = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            crop_point_lr = [upper_left_i,
                             upper_left_j,
                             upper_left_i + PATCH_SIZE_LR[0],
                             upper_left_j + PATCH_SIZE_LR[1]]
            crop_point_hr = [p*2 for p in crop_point_lr]
            crop_point_ps = [p*4 for p in crop_point_hr]
            
            image_20[pCount] = np.rollaxis(dset_20[crop_point_lr[0]:crop_point_lr[2],
                             crop_point_lr[1]:crop_point_lr[3]], 2)
            
            image_10[pCount] = np.rollaxis(dset_10[crop_point_hr[0]:crop_point_hr[2],
                             crop_point_hr[1]:crop_point_hr[3]], 2)
            
            image_ps[pCount] = np.rollaxis(dset_ps[crop_point_ps[0]:crop_point_ps[2],
                             crop_point_ps[1]:crop_point_ps[3]], 2)
            pCount += 1

    return image_10, image_20, image_ps
```

```python
# Break up the image into bite-sized chucks for memory
print('All calculations for Dove imagery.')
print(dove10_with_orb_org.shape)

div_factor = 50

# Width
w_int_64 = int(dove10_with_orb_org.shape[1] / 64)
print('Number of pixel blocks for this image (as integer): %s pixels (width) / 64 pixels =' %dove10_with_orb_org.shape[0], w_int_64)

piece_size_ps = div_factor*64  # Size of ps chunk, multiple of 64
print('Height and width of one square pixel block: %s*64 = ' %div_factor + str(piece_size_ps) + ' pixels')

num_of_pieces = int(w_int_64 / div_factor)
print('Number of pieces per row (one piece = 10 pixel blocks): pixel blocks / %s =' %div_factor, num_of_pieces)

print('Resulting number of pixels per row to be predicted (64 * %s * number of pieces): ' %div_factor, num_of_pieces*64*div_factor)

# Height = Number of rows to process
num_of_rows = int(dove10_with_orb_org.shape[0] / piece_size_ps)
print('Number of rows (rounded integer): Image Height (%s) / Piece height (%s) = ' %(dove10_with_orb_org.shape[0],piece_size_ps), num_of_rows)
```

```python
# Correlate the S2 imagery
piece_size_10m = int(piece_size_ps/4) # Size of S2 10m chunk  
piece_size_20m = int(piece_size_10m/2) # Size of S2 20m chunk  
print('Dove block size:  ', piece_size_ps)
print('S2 10m block size: ', piece_size_10m)
print('S2 20m block size: ', piece_size_20m)
```

```python
# Load model
model_final20 = tf.keras.models.load_model(trained_model, custom_objects={'SRResnet': SRResnet, 'shLoss': shLoss})
```

```python
# Create overlap to mitigate edge effects
for row in range(num_of_rows):
    upper10m = piece_size_10m * row
    lower10m = piece_size_10m * (row+1)
    
    upper20m = piece_size_20m * row
    lower20m = piece_size_20m * (row+1)
    
    upperps = piece_size_ps * row
    lowerps = piece_size_ps * (row+1)
    print('Row segment %s: ' %row, upperps, lowerps)

    final_image_pieces = []
    
    for piece in range(0, num_of_pieces):
        print('Piece: ', piece)
        left10m = piece_size_10m * piece
        right10m = piece_size_10m * (piece+1)
        #print(left10m, right10m)

        left20m = piece_size_20m * piece
        right20m = piece_size_20m * (piece+1)
        #print(left20m, right20m)

        leftps = piece_size_ps * piece
        rightps = piece_size_ps * (piece+1)
        #print(f'{leftps =}, {rightps =}')
            
        # Top left corner            
        if row == 0 and piece == 0:
            location = 'Top left corner'
            print(location)
            s210m_40 =           s210m_40_org[upper10m:lower10m+16, left10m:right10m+16, :]
            s220m_80 =           s220m_80_org[upper20m:lower20m+8, left20m:right20m+8, :]
            dove10_with_orb = dove10_with_orb_org[upperps:lowerps+64, leftps:rightps+64, :]

        # Top right corner            
        elif row == 0 and piece == num_of_pieces-1:
            location = 'Top right corner'
            print(location)
            s210m_40 =           s210m_40_org[upper10m:lower10m+16, left10m-16:right10m, :]
            s220m_80 =           s220m_80_org[upper20m:lower20m+8, left20m-8:right20m, :]
            dove10_with_orb = dove10_with_orb_org[upperps:lowerps+64, leftps-64:rightps, :]
        
        # Top row
        elif row == 0 and piece != 0 and piece != num_of_pieces-1:
            location = 'Top row'
            print(location)
            s210m_40 =           s210m_40_org[upper10m:lower10m+32, left10m-16:right10m+16, :]
            s220m_80 =           s220m_80_org[upper20m:lower20m+16, left20m-8:right20m+8, :]
            dove10_with_orb = dove10_with_orb_org[upperps:lowerps+128, leftps-64:rightps+64, :]
    
       # Bottom left corner   
        elif row == num_of_rows-1 and piece == 0:  
            location = 'Bottom left corner'
            print(location)
            s210m_40 =           s210m_40_org[upper10m-16:lower10m, left10m:right10m+16, :]
            s220m_80 =           s220m_80_org[upper20m-8:lower20m, left20m:right20m+8, :]
            dove10_with_orb = dove10_with_orb_org[upperps-64:lowerps, leftps:rightps+64, :]
            
        # Bottom right corner            
        elif row == num_of_rows-1 and piece == num_of_pieces-1:
            location = 'Bottom right corner'
            print(location)
            s210m_40 =           s210m_40_org[upper10m-16:lower10m, left10m-16:right10m, :]
            s220m_80 =           s220m_80_org[upper20m-8:lower20m, left20m-8:right20m, :]
            dove10_with_orb = dove10_with_orb_org[upperps-64:lowerps, leftps-64:rightps, :]
            
        # Bottom row
        elif row == num_of_rows-1 and piece != num_of_pieces-1:
            location = 'Bottom row'
            print(location)
            s210m_40 =           s210m_40_org[upper10m-32:lower10m, left10m-16:right10m+16, :]
            s220m_80 =           s220m_80_org[upper20m-16:lower20m, left20m-8:right20m+8, :]
            dove10_with_orb = dove10_with_orb_org[upperps-128:lowerps, leftps-64:rightps+64, :]
            
        # Left edge
        elif row != 0 and row != num_of_rows-1 and piece == 0:
            location = 'Left edge'
            print(location)
            s210m_40 =           s210m_40_org[upper10m-16:lower10m+16, left10m:right10m+32, :]
            s220m_80 =           s220m_80_org[upper20m-8:lower20m+8, left20m:right20m+16, :]
            dove10_with_orb = dove10_with_orb_org[upperps-64:lowerps+64, leftps:rightps+128, :]
        
        # Right edge
        elif row != 0 and row != num_of_rows-1 and piece == num_of_pieces-1:
            location = 'Right edge'
            print(location)
            s210m_40 =           s210m_40_org[upper10m-16:lower10m+16, left10m-32:right10m, :]
            s220m_80 =           s220m_80_org[upper20m-8:lower20m+8, left20m-16:right20m, :]
            dove10_with_orb = dove10_with_orb_org[upperps-64:lowerps+64, leftps-128:rightps, :]
            
        # Middle
        else:
            location = 'Middle'
            print(location)
            s210m_40 =           s210m_40_org[upper10m-16:lower10m+16, left10m-16:right10m+16, :]
            s220m_80 =           s220m_80_org[upper20m-8:lower20m+8, left20m-8:right20m+8, :]
            dove10_with_orb = dove10_with_orb_org[upperps-64:lowerps+64, leftps-64:rightps+64, :]

        images_210, images_220, images_ps = get_test_patches_orig(s210m_40, s220m_80, dove10_with_orb)

        images_210 = np.moveaxis(images_210, 1, 3)
        images_220 = np.moveaxis(images_220, 1, 3)
        images_ps = np.moveaxis(images_ps, 1, 3)

        g = model_final20.predict([images_210, images_220, images_ps])

        # Reduce size by converting back to uint16
        g = g.astype('uint16')
        
        # Set up iterator to recreate image
        patches_per_row = np.sqrt(g.shape[0]).astype('int')
        row_length = range(0, g.shape[0]+1, patches_per_row)
        
        # Recreate image
        for num, val in enumerate(row_length[:-1]):
            a=val
            b=row_length[num+1]
            full_block = np.hstack(g[a:b, :, :])
            if num==0:
                img_block = full_block.copy()
            else:
                img_block = np.vstack((img_block,full_block))

        if location == 'Top left corner':
            img_block = img_block[:-64, :-64, :]
            
        elif location == 'Top right corner':
            img_block = img_block[:-64, 64:, :]
            
        elif location == 'Top row':
            img_block = img_block[:-128, 64:-64, :]
            
        elif location == 'Bottom left corner':
            img_block = img_block[64:, :-64, :]
            
        elif location == 'Bottom right corner':
            img_block = img_block[64:, 64:, :]
            
        elif location == 'Bottom row':
            img_block = img_block[128:, 64:-64, :]   
            
        elif location == 'Left edge':
            img_block = img_block[64:-64, :-128, :]             
                    
        elif location == 'Right edge':
            img_block = img_block[64:-64, 128:, :]           
        
        else:
            img_block = img_block[64:-64, 64:-64, :]
        
        final_image_pieces.append(img_block)

        img_block = None
        g = None
        images_210 = None
        images_220 = None
        images_ps = None

    img_row = np.array(final_image_pieces)
    # Puts together the pieces in the row (portion of full image)
    img_row = np.hstack(img_row)

    if row==0:
        full_image = img_row.copy()

    elif row==num_of_rows-1:
        # Clear memory 
        s210m_40_org = None
        s220m_80_org = None
        dove10_org = None
        dove10_orbits_org = None
        dove10_with_orb_org = None
        tf.keras.backend.clear_session()
        model_final20 = None

        full_image = np.vstack((full_image,img_row))

    else:
        full_image = np.vstack((full_image,img_row))

    img_row = None
    final_image_pieces = None
    s210m_40 = None
    s220m_80 = None
    dove_with_orb = None


```

```python
#plt.imshow(full_image[:,:,0])
```

```python
profile['height']=full_image.shape[0]
profile['width']=full_image.shape[1]
profile['count']=full_image.shape[2]
profile['tiled']=True
profile
```

```python
with rasterio.open(image_result_out_fp, 'w', **profile) as dst:
    dst.write(full_image.T)

full_image = None
```

```python

```
