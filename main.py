
import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment
import features_extraction as fextr
import preprocessing as prep
import classification as classf
import pydub
from pylab import *

import scipy.io.wavfile as wavfile
AudioSegment.converter = "C:\\Program Files\\ffmpeg-20180331-be502ec-win64-static\\bin\\ffmpeg.exe"
import sys

#substraction no sound signal?

# arCoeff(): Autorregresion coefficients with Burg order equal to 4
# correlation(): correlation coefficient between two signals
# maxInds(): index of the frequency component with largest magnitude
# meanFreq(): Weighted average of the frequency components to obtain a mean frequency
# skewness(): skewness of the frequency domain signal
# kurtosis(): kurtosis of the frequency domain signal
# bandsEnergy(): Energy of a frequency interval within the 64 bins of the FFT of each window.
# angle(): Angle between to vectors.


data_files=['266766_23_K_21_1.wav', '266766_23_K_22_2.wav', '266766_23_K_9_3.wav', '266766_23_K_11_4.wav']
info_files=['266766_23_K_21_1wav.txt', '266766_23_K_22_2wav.txt', '266766_23_K_9_3wav.txt', '266766_23_K_11_4wav.txt']
output_path="segmented_commands\\"
input_path="recordings\\"
analysis_path="analysis_coeffs\\"


prep.wav_files_segmentation(data_files, info_files, input_path, output_path)
#PSD=prepr.commands_PSD(input_path, analysis_path, True)

print('what')

# low_freq=30
# high_freq=3200
# data_filtr = prepr.data_filtering(data, fs, low_freq,high_freq, 2, True)
x=np.loadtxt('analysis_coeffs\\mean_recordings_266766_23_K_9_3.wav.txt', dtype=float)
print('d[a')


coeffs = fextr.compute_coefficients(input_path)
print(coeffs)