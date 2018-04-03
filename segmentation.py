import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment
import segmentation as seg
import pydub
from pathlib import Path
import pylab
import pywt
import scipy.io.wavfile as wavfile
import matplotlib.cm as cm
import scipy as scipy
from scipy.signal import butter, lfilter
def read_wav_file(data_file, input_path, plotting = True):
    data_wav = wave.open(input_path + data_file, 'r')

    # Extract Raw Audio from Wav File
    raw_signal = data_wav.readframes(-1)
    fs=data_wav.getframerate()

    raw_signal = np.fromstring(raw_signal, 'Int16')
    timeline = np.arange(0, 1 / fs * len(raw_signal), 1 / fs)
    if plotting == True:
        plt.figure
        plt.title('Signal of spoken commands from wav file')
        plt.plot(timeline, raw_signal)
        plt.grid(True)
        plt.show()

    return raw_signal, fs

def wav_files_segmentation(data_files,info_files,input_path, output_path):

    for file_idx in range(len(data_files)):

        commands_file = open(input_path+info_files[file_idx],"r")
        commands = commands_file.readlines()
        commands_file.close()

        nb_command = 0
        for line in commands:
            command = commands[nb_command].split()
            start_ime=float(command[0])*1000
            end_time=float(command[1])*1000
            nb_command += 1

            command_audio = AudioSegment.from_wav(input_path+data_files[file_idx])
            command_audio = command_audio[start_ime:end_time]

            command_audio.export(output_path+command[2]+str(file_idx)+'.wav', format="wav")


def commands_PSD(input_path, analysis_path, plotting=True):

    data_files=[f for f in Path(input_path).glob('**/*.wav') if f.is_file()]

    for file_idx in range(len(data_files)):
        data_wav = wave.open(str(data_files[file_idx]),'r')
        fs = data_wav.getframerate()

        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        power_spectrum = np.abs(np.fft.fft(raw_signal)) ** 2
        if plotting == True:
            time_step = 1 / fs
            freqs = np.fft.fftfreq(raw_signal.size, time_step)
            idx = np.argsort(freqs)

            plt.plot(freqs[idx], power_spectrum[idx])
            plt.show()
            file_name=str(data_files[file_idx])
            file_name=file_name.replace("\\","_")
        np.savetxt(str(analysis_path)+'PSD_'+str(file_name)+'.txt', power_spectrum, fmt='%d')
        #np.loadtxt('test1.txt', dtype=int)

def commands_CWT(input_path, analysis_path, plotting=True):

    data_files=[f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    for file_idx in range(len(data_files)):
        data_wav = wave.open(str(data_files[file_idx]),'r')

        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        CWT_coeffs = pywt.dwt(raw_signal, 'db5')

        if plotting == True:
            pylab.gray()
            scalogram(raw_signal)
            pylab.show()
        file_name = str(data_files[file_idx])
        file_name = file_name.replace("\\", "_")
        np.savetxt(str(analysis_path) + 'CWT_' + str(file_name) + '.txt', CWT_coeffs, fmt='%d')
        return CWT_coeffs


# Make a scalogram given an MRA tree.
def scalogram(data):
    x = pylab.arange(0, 1, 1. /len(data))
    data = np.asarray(data)
    wavelet = 'db5'
    level = 1
    order = "freq"
    interpolation = 'nearest'
    cmap = cm.cool #settings colormap?

    wp = pywt.WaveletPacket(data, wavelet, 'sym', maxlevel=level)
    nodes = wp.get_level(level, order=order)
    labels = [n.path for n in nodes]
    values = pylab.array([n.data for n in nodes], 'd')
    values = abs(values)

    cwt_figure = pylab.figure()

    cwt_figure.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
    pylab.subplot(2, 1, 1)
    pylab.title("raw signal")
    pylab.plot(x, data, 'b')
    pylab.xlim(0, x[-1])

    pylab.subplot(2, 1, 2)
    pylab.title("Wavelet packet coefficients at level %d" % level)
    pylab.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto",
                 origin="lower", extent=[0, 1, 0, len(values)])
    pylab.yticks(pylab.arange(0.5, len(labels) + 0.5), labels)

    pylab.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def data_filtering(data, fs, low_freq,high_freq, order, plotting=True):

    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    filtered_data = lfilter(b, a, data)

    timeline = np.arange(0, 1 / fs * len(filtered_data), 1 / fs)
    if plotting == True:
        plt.figure
        plt.title('Signal of spoken commands from wav file')
        plt.plot(timeline, filtered_data)
        plt.grid(True)
        plt.show()

    return filtered_data
