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

This notebook is built to develop and train the RCCN model for the 10 m bands (RGB and NIR).  Only blocks 2 and 3 will need to be edited by the user to provide the necessary inputs.

```python executionInfo={"elapsed": 3819, "status": "ok", "timestamp": 1635428061968, "user": {"displayName": "Amruta Vidwans", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "11094605598230226609"}, "user_tz": 300} id="p7kNDltfZnld"
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
```

```python
# Change the directory to the location of the outputs by Pyrite (S2 imagery) and BeautySchoolDropout (PS imagery)
os.chdir('...')

# Provide a folder with which to place the final model. 
# IMPORTANT: DO NOT put the same folder as the imagery.Temporary models will be stored there,
# but then deleted. If you do not give a separate location for the complete models, they will also be deleted.
# Ensure there is a '/' at the end.
output_folder = ".../"
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11593, "status": "ok", "timestamp": 1635428106011, "user": {"displayName": "Amruta Vidwans", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "11094605598230226609"}, "user_tz": 300} id="44EfYevlcbCM" outputId="8712900f-2745-40a6-e984-f31b3e3dca95"
# This date (or file identifier) should match what was used in BeautySchoolDropout
date = '17Jun21_'

# These two files correspond to the Pyrite output.  Choose the files with the EXACT SAME ratio (10to40, 20to80)
s210m_40_fn = "T16TCR_20210617T164839_10to40_stack_norm.tif" 
s220m_80_fn = "T16TCR_20210617T164839_20to80_stack_norm.tif" 

# Ground truth. Choose the file that ends with "10_stack_norm.tif"
s220m_20_fn = "T16TCR_20210617T164839_10_stack_norm.tif" 

# If you gave the same identifier above as used in BeautySchoolDropout, this should run without further change
dove10_fn = "%sDove_10m_mosaic.tif" %date # 4 bands + mask
dove10orb_fn = "%sDove_Orbits_10m.tif" %date # orbit(strip number)

s210m_40 = rasterio.open(s210m_40_fn).read().T
s220m_80 = rasterio.open(s220m_80_fn).read().T
dove10 = rasterio.open(dove10_fn).read().T
dove10_orbits = rasterio.open(dove10orb_fn).read().T

print('Shape of downsampled Sentinel-2 10m array: ', s210m_40.shape)
print('Shape of downsampled Sentinel-2 20m array: ', s220m_80.shape)
print('Shape of downsampled Dove array: ', dove10.shape)
print('Shape of downsampled Dove orbit array: ', dove10_orbits.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4180, "status": "ok", "timestamp": 1635428110187, "user": {"displayName": "Amruta Vidwans", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "11094605598230226609"}, "user_tz": 300} id="-Po_bHlNdmNJ" outputId="b5781ad2-847a-4b67-a3c7-21defca63268"
s220m_20 = rasterio.open(s220m_20_fn).read().T
s220m_20 = s220m_20[:, :, :-1] # Remove mask band for validation

s220m_20.shape
```

```python
encoded = ku.to_categorical(dove10_orbits, dtype='uint16')
print(encoded.shape)

dove10_with_orb = np.concatenate((dove10, encoded),axis=-1)
print(dove10_with_orb.shape)
```

```python
# Check for shape symetry and pad with zeros as needed
model_images = [s210m_40, s220m_80, dove10_with_orb, s220m_20]

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

```python executionInfo={"elapsed": 364, "status": "ok", "timestamp": 1635428111214, "user": {"displayName": "Amruta Vidwans", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "11094605598230226609"}, "user_tz": 300} id="SiuRB6K6eI5R"
def get_test_patches(dset_10m, dset_20m, ps, dset_20gt, patchSize=24, border=4, ps_patch=96, ps_border=16):

    PATCH_SIZE_HR = (patchSize, patchSize)
    PATCH_SIZE_LR = [p//2 for p in PATCH_SIZE_HR]
    BORDER_HR = border
    BORDER_LR = BORDER_HR//2
    PATCH_SIZE_PS = (ps_patch, ps_patch)
    PATCH_SIZE_GT = (ps_patch-2*ps_border, ps_patch-2*ps_border)
    
    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(dset_10m, ((BORDER_HR, BORDER_HR), (BORDER_HR, BORDER_HR), (0, 0)))
    dset_20 = np.pad(dset_20m, ((BORDER_LR, BORDER_LR), (BORDER_LR, BORDER_LR), (0, 0)))
    dset_ps = np.pad(ps, ((ps_border, ps_border), (ps_border, ps_border), (0, 0)))
    dset_gt = np.pad(dset_20gt, ((ps_border, ps_border), (ps_border, ps_border), (0, 0)))
    
    BANDS10 = dset_10.shape[2]
    BANDS20 = dset_20.shape[2]
    BANDSps = dset_ps.shape[2]
    BANDSgt = dset_gt.shape[2]
    patchesAlongi = (dset_20.shape[0] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    patchesAlongj = (dset_20.shape[1] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)

    label_20 = np.zeros((nr_patches, BANDSgt) + PATCH_SIZE_GT).astype(np.float32)   #initialize with PATCH_SIZE_PS but crop to PATCH_SIZE_GT
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
            crop_point_gt = [p*4 for p in crop_point_hr]

            
            label_20[pCount] = np.rollaxis(dset_gt[crop_point_gt[0]+ps_border:crop_point_gt[2]-ps_border, 
                                                crop_point_gt[1]+ps_border:crop_point_gt[3]-ps_border], 2)
            
            image_20[pCount] = np.rollaxis(dset_20[crop_point_lr[0]:crop_point_lr[2],
                             crop_point_lr[1]:crop_point_lr[3]], 2)
            
            image_10[pCount] = np.rollaxis(dset_10[crop_point_hr[0]:crop_point_hr[2],
                             crop_point_hr[1]:crop_point_hr[3]], 2)
            
            image_ps[pCount] = np.rollaxis(dset_ps[crop_point_ps[0]:crop_point_ps[2],
                             crop_point_ps[1]:crop_point_ps[3]], 2)
            pCount += 1

    return image_10, image_20, image_ps, label_20
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4205, "status": "ok", "timestamp": 1635428115415, "user": {"displayName": "Amruta Vidwans", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "11094605598230226609"}, "user_tz": 300} id="2DclF1KceMrS" outputId="265613c7-b5d8-4a1a-b1b7-5a71673e5751"
images_210, images_220, images_ps, label_20 = get_test_patches(model_images[0], model_images[1], model_images[2], model_images[3])
#images_210, images_220, images_ps, label_20 = get_test_patches(s210m_40, s220m_80, dove10_with_orb, s220m_20)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1635428116386, "user": {"displayName": "Amruta Vidwans", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "11094605598230226609"}, "user_tz": 300} id="s94kbHRwePXH" outputId="ce1b7b5e-ec98-48a9-8d2d-f36b76df9a6b"
images_210 = np.moveaxis(images_210, 1, 3)
images_220 = np.moveaxis(images_220, 1, 3)
images_ps = np.moveaxis(images_ps, 1, 3)
label_20 = np.moveaxis(label_20, 1, 3)

print(images_210.shape)
print(images_220.shape)
print(images_ps.shape)
print(label_20.shape)
```

```python executionInfo={"elapsed": 2097, "status": "ok", "timestamp": 1635428119401, "user": {"displayName": "Amruta Vidwans", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "11094605598230226609"}, "user_tz": 300} id="eokpDUcOeaQ6"
images_210_tr1, images_210_test, images_220_tr1, images_220_test, images_ps_tr1, images_ps_test, label_20_tr1, label_20_test = train_test_split(images_210, images_220, images_ps, label_20, test_size=0.1, random_state=1)
images_210_train, images_210_val, images_220_train, images_220_val, images_ps_train, images_ps_val, label_20_train, label_20_val = train_test_split(images_210_tr1, images_220_tr1, images_ps_tr1, label_20_tr1, test_size=0.1, random_state=1)
```

```python id="fzts5DOledmK"
lr = 0.0001
batch_size1 = 32
epochs1 = 100
steps_per_epoch = 300
tryout = 200
gpu = 16 #check
sample = 32
mlt = 5
scale = 4
# patch_size = int(scale * 1024)
patch_size = 96

test_only = False

chk = 1
CHANNEL = 3

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
        allrc = Convolution2D(filters = 4, kernel_size = 1, strides = 1, padding = "valid", activation = None)(allm)
        
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


chk = -1
net = SRResnet(images_210, images_220, images_ps, lr = lr ,scale = scale , chk = chk)
net.visualize()
# net.get_model().summary()

my_callbacks =[
            tf.keras.callbacks.ModelCheckpoint(filepath='model4x.{epoch:02d}-{val_loss:.2f}',
              monitor = "loss",
              verbose = 1,
              save_best_only = True,
              save_freq = "epoch"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
              monitor = "loss",
              factor = 0.9,
              patience = 20,
              verbose = 1,
              min_lr = 0.0001 / 10
            ),
            tf.keras.callbacks.EarlyStopping(
              monitor = "loss",
              min_delta = 2,
              patience = 200,
              verbose = 1,
              baseline = None,
              restore_best_weights = False
            ),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
```

```python
model10 = net.get_model()
model10.summary()
%tensorboard --logdir logs/
model10.fit(x=[images_210_train, images_220_train, images_ps_train], y=label_20_train, batch_size=batch_size1, epochs=epochs1, verbose=1, callbacks=my_callbacks, validation_data=([images_210_val, images_220_val, images_ps_val], label_20_val))#, validation_steps = 3) #steps_per_epoch=steps_per_epoch

model10.save(output_folder + date + 'trained_10m_model')
```

```python id="gSJOnNzQm55S"
# Remove old models and log files
files_list = os.listdir()

for file in files_list:
    if 'model' in file:
        shutil.rmtree(file)
        
for file in files_list:
    if 'logs' in file:
        shutil.rmtree(file)
```

```python

```
