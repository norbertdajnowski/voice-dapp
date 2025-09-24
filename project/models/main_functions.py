import pyaudio
from IPython.display import Audio, display, clear_output
import wave
from scipy.io.wavfile import read
import sys
from sklearn import mixture
from sklearn import preprocessing
import python_speech_features as mfcc
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


#Calculate and returns the delta of given feature vector matrix
def calculate_delta(array):
    rows, cols = array.shape
    # Use the actual number of columns instead of hardcoded 20
    deltas = np.zeros((rows, cols))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1        
            else:                       
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

#convert audio to mfcc features
def extract_features(audio,rate):    
    # Enhanced MFCC parameters for better voice recognition
    # Increased number of MFCC coefficients from 20 to 26 for better spectral representation
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 26, appendEnergy=True, nfft=2048)
    
    # Apply robust scaling instead of standard scaling for better outlier handling
    mfcc_feat = preprocessing.robust_scale(mfcc_feat)
    
    # Calculate delta features (first derivatives)
    delta = calculate_delta(mfcc_feat)
    
    # Calculate delta-delta features (second derivatives) for better temporal modeling
    delta_delta = calculate_delta(delta)
    
    # Combine MFCC, delta, and delta-delta features
    combined = np.hstack((mfcc_feat, delta, delta_delta)) 
    
    # Apply additional smoothing to reduce noise
    combined = preprocessing.scale(combined)
    
    return combined
