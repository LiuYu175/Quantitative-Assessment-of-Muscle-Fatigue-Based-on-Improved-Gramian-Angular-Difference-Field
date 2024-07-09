import time
import numpy as np
import os
import shutil
import random
from numpy import max, min, average, std, sum, sqrt, where, argmax, absolute, array, random, zeros
from pyts.image import GramianAngularField,MarkovTransitionField,RecurrencePlot
from PIL import Image, ImageOps
from numba import njit, prange

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass

@njit()
def _gasf(X_cos, X_sin, n_samples, image_size):
    X_gasf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        X_gasf[i] = np.outer(X_cos[i], X_cos[i]) - np.outer(X_sin[i], X_sin[i])
    return X_gasf

@njit()
def _gadf(X_cos, X_sin, n_samples, image_size):
    X_gadf = np.empty((n_samples, image_size, image_size))
    for i in prange(n_samples):
        X_gadf[i] = np.outer(X_sin[i], X_cos[i]) - np.outer(X_cos[i], X_sin[i])
    return X_gadf

def GAF_0_1(X, method_, mean, std):
    n_samples, n_timestamps = X.shape
    image_size = n_timestamps

    x_min = np.min(X)
    x_max = np.max(X)

    # 0-1 normalization
    X = (X - x_min) / (x_max - x_min)

    # Normalized to [-1, 1].
    X = 2 * X - 1

    X_cos = np.arccos(X)

    X_sin = np.arcsin(X)

    if method_ in ['s', 'summation']:
        X_new = _gasf(X_cos, X_sin, n_samples, image_size)
    else:
        X_new = _gadf(X_cos, X_sin, n_samples, image_size)

    return X_new

def GAF_Gass(X, method_, mean, std):
    
    n_samples, n_timestamps = X.shape
    image_size = n_timestamps

    #Gaussian normalization
    X = (X - mean) / std

    # Triple standard deviation truncation
    truncated_data = np.clip(X, -3, 3)

    # Normalized to [-1, 1].
    min_val = np.min(truncated_data)
    max_val = np.max(truncated_data)
    X = 2 * (truncated_data - min_val) / (max_val - min_val) - 1

    X_cos = np.arccos(X)
        
    X_sin = np.arcsin(X)
    
    if method_ in ['s', 'summation']:
        X_new = _gasf(X_cos, X_sin, n_samples, image_size)
    else:
        X_new = _gadf(X_cos, X_sin, n_samples, image_size)
    
    return X_new

def crop(im):
    X_Crop=[]
        
    width, height = im.size

    crop_length, stride = 224, 224
    # Determine the start position of the move crop
    x_start = 0
    y_start = 0
    
    # Crop the image and save the cropped image, keeping only half of the cropped image
    k=1
    while y_start + crop_length <= height:
        while x_start + crop_length <= width:
            # Cropping images
            box = (x_start, y_start, x_start + crop_length, y_start + crop_length)
            cropped_image = im.crop(box)
            
            X_Crop.append(cropped_image)
            
            # Moving the crop window
            x_start += stride
        y_start += stride
        x_start = 0             # Retain all
        k+=1

    return X_Crop

def to_2D(path, A1_combine, Borg, feature, mean, std, scale):
    if feature == 'MTF':
        method = MarkovTransitionField(n_bins=8)
    elif feature == 'RP':
        method = RecurrencePlot(threshold='point',percentage=50)
        
    path=path+'/ALL/'+Borg+'/'
    
    for i in range(len(A1_combine)):
        X=np.array([A1_combine[i]])
    
        if feature == 'MTF' or feature =='RP':
            im=method.fit_transform(X)
            im=(im*255).astype(np.uint8)
            im=Image.fromarray(im[0])
        elif feature == 'GASF':
            im=GAF_0_1(X, 's', mean, std)
            im=((im+1)/2*255).astype(np.uint8)
            im=Image.fromarray(im[0])
        elif feature == 'GADF':
            im=GAF_0_1(X, 'd', mean, std)
            im=((im+1)/2*255).astype(np.uint8)
            im=Image.fromarray(im[0])
        elif feature == 'G-GASF' :
            im=GAF_Gass(X, 's', mean, std)
            im=((im+1)/2*255).astype(np.uint8)
            im=Image.fromarray(im[0])
        elif feature == 'G-GADF':
            im=GAF_Gass(X, 'd', mean, std)
            im=((im+1)/2*255).astype(np.uint8)
            im=Image.fromarray(im[0])

        im=im.resize((scale,scale))
        X_Crop=crop(im)
        
        #Save all the splits and divide them later.
        for l in range(len(X_Crop)):
            X_Crop[l].save(path+str(len(os.listdir(path)))+'.png')

def shuffle_and_divide_files(path,feature,i_name,train,val,test):
    files = os.listdir(f'{path}DATA_{feature}/ALL/{i_name}')
    '''Disrupt local file order'''
    random.shuffle(files)
    L1 = int(len(files)*0.1*train)
    L2 = int(len(files)*0.1*(train+val))
    for l in files[:L1]:
        shutil.copyfile(f'{path}DATA_{feature}/ALL/{i_name}/{l}', f'{path}DATA_{feature}/train/{i_name}/{l}')
    for l in files[L1:L2]:
        shutil.copyfile(f'{path}DATA_{feature}/ALL/{i_name}/{l}', f'{path}DATA_{feature}/validation/{i_name}/{l}')
    for l in files[L2:]:
        shutil.copyfile(f'{path}DATA_{feature}/ALL/{i_name}/{l}', f'{path}DATA_{feature}/test/{i_name}/{l}')
    
    print(f'{path}DATA_{feature}/{i_name} is Done')

def shuffle(folder_path):
    '''Globally randomly disrupt files in a folder'''

    file_list = os.listdir(folder_path)

    random.shuffle(file_list)

    for i, filename in enumerate(file_list):
        _, extension = os.path.splitext(filename)
        new_filename = f"_{i + 1}{extension}"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        shutil.move(old_filepath, new_filepath) 

def get_mean_std(path):
    '''Calculate mean and variance'''
    files = os.listdir(path)
    sig_data=[]
    data=[]
    for file in files:
        d=np.load(path+'/'+file)
        data=np.hstack((data, d))
        sig_data.append(d)
        
    mean_ = np.mean(data)
    std_ = np.std(data)
    return mean_, std_, sig_data

def image_encoding():
    #This program generates images
    #The adjustable parameters are as follows
    set_names = [
        'A1',
        'A2',
        'A3',
        'Max'
    ]
    feature=[
        'G-GASF',
        'G-GADF',
        'GASF',
        'GADF',
        'MTF',
        'RP',
    ]
    
    train, val, test = 7, 2, 1
    scale = 224 * 4
    create_folder_if_not_exists('ALL')
    for set_name in set_names:
        try:
            shutil.rmtree('ALL/'+set_name)
            pass
        except:
            pass
        create_folder_if_not_exists('ALL/'+set_name)
        for f in feature:
            create_folder_if_not_exists('ALL/'+set_name+'/DATA_'+f+'/')
            for k in ['ALL','train','validation','test']:
                create_folder_if_not_exists('ALL/'+set_name+'/DATA_'+f+'/'+k)
                for i in ['0','1','2','3','4','5','6','7','8','9']:
                    create_folder_if_not_exists('ALL/'+set_name+'/DATA_'+f+'/'+k+'/'+i)

        #Read the data
        for k in ['0','1','2','3','4','5','6','7','8','9']:
            data_path='Preprocessing/'+set_name+'/'+k
            #Calculate mean and variance
            mean_, std_, sig_data = get_mean_std(data_path)
            # print(mean_, std_, len(sig_data))
            for f in feature:
                to_2D('ALL/' + set_name + '/DATA_' + f, sig_data, k, f, mean_, std_, scale)
                print(set_name,f,k)

        '''Disrupting and dividing the dataset'''
        start_time = time.time()
        for f in feature:
            for i in ['0','1','2','3','4','5','6','7','8','9']:
                shuffle_and_divide_files('ALL/'+set_name+'/',f,i,train,val,test)

        for f in feature:
            for k in ['train','validation','test']:
                for i in ['0','1','2','3','4','5','6','7','8','9']:
                    shuffle('ALL/'+set_name+'/DATA_'+f+'/'+k+'/'+i)
                    print('ALL/'+set_name+'/DATA_'+f+'/'+k+'/'+i+' is shuffled')
    