import pandas as pd
import numpy as np
import os
import biosignalsnotebooks as bsnb
import shutil
from numpy import max
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

def create_folder_if_not_exists(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
        os.makedirs(folder_path)
    else:
        pass

def change_unit(signal):
    '''conversion unit'''
    device='bitalino_rev'
    sr=1000
    resolution=10
    signal_mv = bsnb.raw_to_phy("EMG", device, signal, resolution, option="mV") # Conversion to mV
    return signal_mv

def Filter(signal):
    '''
    denoising function
    '''
    fs = 1000  # sampling rate

    # Design of fourth-order zero-phase lag bandpass Butterworth filters
    lowcut = 5  # Low Frequency Cutoff Frequency
    highcut = 495  # High Frequency Cutoff Frequency
    notch_freq = 50  # Power Line Frequency
    
    # Normalized frequency range
    nyq = 0.5 * fs  # Nyquist frequency
    
    low = lowcut / nyq
    high = highcut / nyq
    notch = notch_freq / nyq
    
    # Design of fourth-order zero-phase lag bandpass Butterworth filters
    b_band, a_band = butter(4, [low, high], btype='band')

    # Design of second-order bandstop filters
    b_notch, a_notch = butter(2, [notch - 0.02, notch + 0.02], btype='bandstop')
    
    # Applied bandpass filters
    filtered_semg_bandpass = filtfilt(b_band, a_band, signal)
    
    # Applying bandstop filters
    filtered_semg_notch = filtfilt(b_notch, a_notch, filtered_semg_bandpass)
    
    return filtered_semg_notch

def slide_cut(Time,Time_click,signal,L,R):
    '''Slicing the EMG signal each time, the signal before the first and after the last don't'''
    result1=[]
    index=[0]
    for i in range(len(Time_click)):
        for j in range(index[-1],len(Time)):
            if Time[j]>Time_click[i]:
                index.append(j)
                break

    for i in range(1,len(index)-1):
        result1.append(signal[index[i]:index[i+1]])
    
    result2=[]
    for i in range(len(result1)):
        result2.append(result1[i][int(L[i]):int(R[i])])

    return result2

def rescale_time_series(time_series, target_length):
    # Generate an index corresponding to the length of the time series
    original_length = len(time_series)
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)

    # Interpolation using linear interpolation
    interpolator = interp1d(original_indices, time_series, kind='linear', fill_value='extrapolate')
    rescaled_time_series = interpolator(target_indices)

    return rescaled_time_series

def main(ID):
    raw_data_path='raw_2.5/'+ID+'.csv'
    raw_data_click_path='raw_2.5/'+ID+'_click.csv'

    raw_data=pd.read_csv(raw_data_path)
    raw_data_click=pd.read_csv(raw_data_click_path)

    Time=raw_data['Time'].to_numpy()
    A1=raw_data['A1'].to_numpy()
    A2=raw_data['A2'].to_numpy()
    A3=raw_data['A3'].to_numpy()

    Time_click=raw_data_click['Time'].to_numpy()
    Borg=raw_data_click['Borg'].to_numpy()#Subjective levels of fatigue were recorded
    position_L=raw_data_click['L'].to_numpy()#Records the location of the click
    position_R=raw_data_click['R'].to_numpy()#Records the location of the click

    A1_cut=slide_cut(Time,Time_click,Filter(change_unit(A1)),position_L,position_R)
    A2_cut=slide_cut(Time,Time_click,Filter(change_unit(A2)),position_L,position_R)
    A3_cut=slide_cut(Time,Time_click,Filter(change_unit(A3)),position_L,position_R)
    
    '''Combining data from three channels'''
    A_combine_1=[]

    A1,A2,A3=[],[],[]
    for i in range(len(A1_cut)):
        S=[]
        for j in range(len(A1_cut[i])):
            unit=[int(A1_cut[i][j]/abs(A1_cut[i][j])),int(A2_cut[i][j]/abs(A2_cut[i][j])),int(A3_cut[i][j]/abs(A3_cut[i][j]))]
            B = [abs(A1_cut[i][j]),abs(A2_cut[i][j]),abs(A3_cut[i][j])]
            
            max_index = B.index(max(B))
            max_value = B[max_index]*unit[max_index]
            S.append(max_value)
                
        #Maximum
        A_combine_1.append(S)
        
        a1=A1_cut[i]
        a2=A2_cut[i]
        a3=A3_cut[i]
        
        A1.append(a1)
        A2.append(a2)
        A3.append(a3)        

    
    '''Save data, save according to RMF's level'''
    for i in range(len(A1)):
        np.save('Preprocessing/A1/'+str(int(Borg[i]))+'/'+str(len(os.listdir('Preprocessing/A1/'+str(int(Borg[i])))))+'.npy',A1[i])
        np.save('Preprocessing/A2/'+str(int(Borg[i]))+'/'+str(len(os.listdir('Preprocessing/A2/'+str(int(Borg[i])))))+'.npy',A2[i])
        np.save('Preprocessing/A3/'+str(int(Borg[i]))+'/'+str(len(os.listdir('Preprocessing/A3/'+str(int(Borg[i])))))+'.npy',A3[i])
        np.save('Preprocessing/Max/'+str(int(Borg[i]))+'/'+str(len(os.listdir('Preprocessing/Max/'+str(int(Borg[i])))))+'.npy',A_combine_1[i])

def preprocessing():
    #This program pre-processes the raw data and saves it in the form of numpy
    filename = os.listdir('raw_2.5')
    
    ID=set()
    for i in range(len(filename)):
        ID.add(filename[i][:2])
    ID=list(ID)
    ID.sort()

    try:
        shutil.rmtree('Preprocessing')
        pass
    except:
        pass
    
    create_folder_if_not_exists('Preprocessing/')
    for r in ['A1','A2','A3','Max','Sta']:
        create_folder_if_not_exists('Preprocessing/'+r)
        for k in ['0','1','2','3','4','5','6','7','8','9']:
            create_folder_if_not_exists('Preprocessing/'+r+'/'+k)
    
    for i in ID:
        print('The currently processed ID is: ',i)
        main(i)
