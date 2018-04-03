
import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment
import segmentation as prepr
import pydub
from pylab import *
import pywt
import scipy.io.wavfile as wavfile
AudioSegment.converter = "C:\\Program Files\\ffmpeg-20180331-be502ec-win64-static\\bin\\ffmpeg.exe"
import sys

data_files=['266766_23_K_21_1.wav', '266766_23_K_22_2.wav', '266766_23_K_9_3.wav', '266766_23_K_11_4.wav']
info_files=['266766_23_K_21_1wav.txt', '266766_23_K_22_2wav.txt', '266766_23_K_9_3wav.txt', '266766_23_K_11_4wav.txt']
output_path="segmented_commands\\"
input_path="recordings\\"
analysis_path="analysis_coeffs\\"


#prepr.wav_files_segmentation(data_files, info_files, input_path, output_path)
#prepr.commands_SPD(output_path)
#prepr.commands_CWT(input_path,analysis_path, False)
data, fs =prepr.read_wav_file(data_files[1], input_path, True)
low_freq=30
high_freq=3200
data_filtr = prepr.data_filtering(data, fs, low_freq,high_freq, 2, True)
