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
import pandas
#from spectrum import *
from scipy.signal import butter, lfilter

def read_wav_file(data_file, input_path, plotting = True):

    data_wav = wave.open(input_path + data_file, 'r')

    raw_signal = data_wav.readframes(-1)
    fs=data_wav.getframerate()

    raw_signal = np.fromstring(raw_signal, 'Int16')
    timeline = np.arange(0, 1 / fs * len(raw_signal), 1 / fs)

    if plotting:
        plt.figure
        plt.title('Signal of spoken commands from wav file')
        plt.plot(timeline, raw_signal)
        plt.grid(True)
        plt.show()

    return raw_signal, fs

def butter_bandpass(low_freq, high_freq, fs, order = 5):

    freq_nyq = 0.5 * fs
    low_cut = low_freq / freq_nyq
    high_cut = high_freq / freq_nyq
    b_coeff, a_coeff = butter(order, [low_cut, high_cut], btype='band')

    return b_coeff, a_coeff

def data_filtering(data, fs, low_freq,high_freq, order, plotting = True):

    b_coeff, a_coeff = butter_bandpass(low_freq, high_freq, fs, order=order)
    filtered_data = lfilter(b_coeff, a_coeff, data)

    timeline = np.arange(0, 1 / fs * len(filtered_data), 1 / fs)

    if plotting:
        plt.figure
        plt.title('Signal of spoken commands from wav file')
        plt.plot(timeline, filtered_data)
        plt.grid(True)
        plt.show()

    return filtered_data

def normalization(data):

    offset = np.mean(data)
    data = data - offset
    norm_data = (data - min(data)) / (max(data) - min(data))

    return norm_data

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


def commands_PSD(input_path, analysis_path = False, saving = False, plotting = False):

    data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]

    power_spectrum_all = []


    for file_idx in range(len(data_files)):
        data_wav = wave.open(str(data_files[file_idx]),'r')
        fs = data_wav.getframerate()

        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        power_spectrum = np.abs(np.fft.fft(raw_signal)) ** 2

        power_spectrum_all.append(power_spectrum)

        time_step = 1 / fs
        freqs = np.fft.fftfreq(raw_signal.size, time_step)
        idx = np.argsort(freqs)
        print(type(freqs))
        print(type(power_spectrum))

        if plotting:

            plt.plot(freqs[idx], power_spectrum[idx])
            plt.show()

        if saving and analysis_path:
            data = np.vstack((freqs[idx],power_spectrum[idx]))
            file_name=str(data_files[file_idx])
            file_name=file_name.replace("\\","_")
            file_name = file_name.replace(".", "")
            file_path = open(str(analysis_path) + 'PSD_'+str(file_name)+'.txt''.txt', 'w')
            print('DATAAA')
            print(data)
            for line in data:
                for item in line:
                    file_path.write("%f " % item)
                file_path.write("\n")
            #np.savetxt(str(analysis_path)+'PSD_'+str(file_name)+'.txt', data, fmt='%f')
    if saving:
        input_path = str(input_path)
        input_path = input_path.replace("\\", "")
        input_path = input_path.replace(".", "")
        file_path = open(str(analysis_path) + 'PSD_'+ input_path + '.txt', 'w')

        for line in power_spectrum_all:
            for item in line:
                file_path.write("%f " % item)
            file_path.write("\n")
        #np.savetxt(str(analysis_path) + 'PSD_' + str(input_path) + '.txt', power_spectrum_all)


    return power_spectrum_all

def commands_DWT(input_path, analysis_path,  saving = False, plotting = False):
    #Discrete Wavelet Transform

    data_files=[f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    DWT_coeffs_all = []

    for file_idx in range(len(data_files)):
        data_wav = wave.open(str(data_files[file_idx]),'r')

        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        DWT_coeffs = pywt.dwt(raw_signal, 'db5')
        DWT_coeffs_all.append(DWT_coeffs)

        if plotting:
            pylab.gray()
            scalogram(raw_signal)
            pylab.show()

        if saving:
            file_name = str(data_files[file_idx])
            file_name = file_name.replace("\\", "_")
            file_name = file_name.replace(".", "")
            np.savetxt(str(analysis_path) + 'CWT_' + str(file_name) + '.txt', DWT_coeffs, fmt='%d')
    if saving:
        input_path = str(input_path)
        input_path = input_path.replace("\\", "")
        input_path = input_path.replace(".", "")
        file_path = open(str(analysis_path) + 'PSD_' + input_path +'.txt', 'w')

        for line in DWT_coeffs_all:
            for item in line:
                file_path.write("%f " % item)
            file_path.write("\n")

    return DWT_coeffs


# Make a scalogram given an MRA tree.
def scalogram(data):
    xrange = pylab.arange(0, 1, 1. /len(data))
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
    pylab.plot(xrange, data, 'b')
    pylab.xlim(0, xrange[-1])

    pylab.subplot(2, 1, 2)
    pylab.title("Wavelet packet coefficients at level %d" % level)
    pylab.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto",
                 origin="lower", extent=[0, 1, 0, len(values)])
    pylab.yticks(pylab.arange(0.5, len(labels) + 0.5), labels)

    pylab.show()


def commands_mean_std_min_max(input_path, analysis_path = False):
    #mean(): Mean value
    # std(): Standard deviation
    # mad(): Median absolute deviation
    # max(): Largest value in array
    # min(): Smallest value in array
    print('FOILEE')
    print(input_path)
    if str(input_path)[-4:] == '.wav':
        data_files=[input_path]
    else:
        data_files=[f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    print(data_files)
    commands_stats_all = np.zeros(shape = (4, len(data_files)))

    for file_idx in range(len(data_files)):

        data_wav = wave.open(str(data_files[file_idx]),'r')

        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        command_mean = np.array(np.mean(raw_signal))
        command_std = np.std(raw_signal)
        command_min = np.min(raw_signal)
        command_max = np.max(raw_signal)
        command_stats = [command_mean, command_std , command_min, command_max]

        commands_stats_all[:][file_idx] = command_stats

        if analysis_path:
            file_name = str(data_files[file_idx])
            file_name = file_name.replace("\\", "_")
            file_name = file_name.replace(".", "")
            np.savetxt(str(analysis_path) + 'mean_' + str(file_name) + '.txt', command_stats, fmt='%f')
    if analysis_path:
        np.savetxt(str(analysis_path) + 'PSD_' + str(input_path) + '.txt', commands_stats_all)

    return commands_stats_all

def commands_SMA(input_path, analysis_path, saving = False):
    #signal magnitiude area

    data_files=[f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    SMA_all = np.zeros(len(data_files))

    for file_idx in range(len(data_files)):

        data_wav = wave.open(str(data_files[file_idx]),'r')
        fs = data_wav.getframerate()
        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')
        sum = 0
        timeline = np.arange(0, 1 / fs * len(raw_signal), 1 / fs)

        for sample_idx in range(len(raw_signal)):
            sum += (abs(timeline[sample_idx]) + abs(raw_signal[sample_idx]))

        command_SMA = sum / len(raw_signal)
        SMA_all[file_idx] = command_SMA

        command_SMA = [command_SMA]


        if saving:
            file_name = str(data_files[file_idx])
            file_name = file_name.replace("\\", "_")
            file_name = file_name.replace(".", "")
            np.savetxt(str(analysis_path) + 'SMA_' + str(file_name) + '.txt', command_SMA, fmt='%f')

    if saving:
        np.savetxt(str(analysis_path) + 'PSD_' + str(input_path) + '.txt', SMA_all)


    return SMA_all

#energy(): Energy measure. Sum of the squares divided by the number of values.

def commands_energy(input_path, analysis_path, saving = True):

    data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    energy_all = np.zeros(len(data_files))

    for file_idx in range(len(data_files)):

        data_wav = wave.open(str(data_files[file_idx]), 'r')
        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')
        sum = 0

        for sample_idx in range(len(raw_signal)):
            sum += (raw_signal[sample_idx]) ** 2

        command_energy = sum / len(raw_signal)
        energy_all[file_idx] = command_energy

        command_energy = [command_energy]

        if saving:
            file_name = str(data_files[file_idx])
            file_name = file_name.replace("\\", "_")
            np.savetxt(str(analysis_path) + 'Energy_' + str(file_name) + '.txt', command_energy, fmt='%f')
    if saving:
        np.savetxt(str(analysis_path) + 'Energy_' + str(input_path) + '.txt', energy_all)

    return energy_all


#Interquartile range
def commands_IQR(input_path, analysis_path, saving = True):

    data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    IQR_all = np.zeros(len(data_files))

    for file_idx in range(len(data_files)):

        data_wav = wave.open(str(data_files[file_idx]), 'r')
        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        q75, q25 = np.percentile(raw_signal, [75, 25])
        command_IQR = float(q75 - q25)
        IQR_all[file_idx] = command_IQR
        command_IQR = [command_IQR]

        if saving:
            file_name = str(data_files[file_idx])
            file_name = file_name.replace("\\", "_")
            np.savetxt(str(analysis_path) + 'Energy_' + str(file_name) + '.txt', command_IQR, fmt='%f')
    if saving:
        input_path = input_path.replace("\\", "")
        np.savetxt(str(analysis_path) + 'IQR_' + str(input_path) + '.txt', IQR_all)

    return IQR_all


#signal entropy

def commands_entropy(input_path, analysis_path, saving = True):

    data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    entropy_all = np.zeros(len(data_files))

    for file_idx in range(len(data_files)):

        data_wav = wave.open(str(data_files[file_idx]), 'r')
        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')


        #command_entropy = get_ApEn(raw_signal)
        p_data = pandas.value_counts(raw_signal) / len(raw_signal)  # calculates the probabilities
        command_entropy = scipy.stats.entropy(p_data)
        print(command_entropy)

        entropy_all[file_idx] = command_entropy
        command_entropy = [command_entropy]

        if saving:
            file_name = str(data_files[file_idx])
            file_name = file_name.replace("\\", "_")
            np.savetxt(str(analysis_path) + 'Entropy_' + str(file_name) + '.txt', command_entropy, fmt='%f')
    if saving:
        input_path = input_path.replace("\\", "")
        np.savetxt(str(analysis_path) + 'Entropy_' + str(input_path) + '.txt', entropy_all)

    return entropy_all




def get_ApEn(RR_intervals, m=2, r_mlp=0.2):
    """
    Estimate Approximate Entropy of signal
    ----------
    Parameters:
    ----------
    fHR: array
        signal.
    t_fHR: array
        Timestamps for fHR calculated with ZuzaDSP, fs=0.4 Hz
    m: int
        Lenght of compared RR intervals sequences.
    r_mlp: float
        Multiple standard deviation. Tolerance for accepting matches is standard deviation multiplied by r_mpl.

    Outputs:
    -------
    ApEn: float
        Approximate Entropy for fHR record.
    """
    # parameters

    r = r_mlp * np.nanstd(RR_intervals)
    N = len(RR_intervals)
    Phi_m_r = np.zeros(2)

    for n in range(2):
        m = m + n
        Pm = np.zeros((N - m + 1, m))
        # Pm vectors
        for j in range(N - m + 1):
            for i in range(m):
                Pm[j, i] = RR_intervals[j + i]
        # calculate distances vector
        pm_distances = np.zeros((N - m + 1, N - m + 1))
        for i in range(N - m + 1):
            for j in range(N - m + 1):
                dist = np.zeros(m)
                for k in range(m):
                    dist[k] = np.abs(Pm[j, k] - Pm[i, k])
                    pm_distances[i, j] = np.nanmax(dist)
                    pm_distances[j, i] = np.nanmax(dist)
                    # comparision with tolerance
        pm_similarity = pm_distances > r
        # function Cmr
        C_m_r = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            n_i = np.nansum(pm_similarity[i])
            C_m_r[i] = float(n_i) / float(N)
        # phi parameter- Cmr logarithms mean
        Phi_m_r[n] = np.nanmean(np.log(C_m_r))
    ApEn = np.abs(Phi_m_r[0] - Phi_m_r[1])


    return ApEn

def commands_autoreggression_coeffs(input_path, analysis_path, saving = True):
    # Levinson-Durbin algorithm for solving the Hermitian Toeplitz system of Yule-Walker equations in the AR estimation problem
#     #from spectrum import *
# from pylab import *
# a,b, rho = arma_estimate(marple_data, 15, 15, 30)
# psd = arma2psd(A=a, B=b, rho=rho, sides='centerdc', norm=True)
# plot(10 * log10(psd))
# ylim([-50,0])


    data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    entropy_all = np.zeros(len(data_files))

    for file_idx in range(len(data_files)):

        data_wav = wave.open(str(data_files[file_idx]), 'r')
        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')


        #command_entropy = get_ApEn(raw_signal)
        p_data = pandas.value_counts(raw_signal) / len(raw_signal)  # calculates the probabilities
        command_entropy = scipy.stats.entropy(p_data)
        print(command_entropy)

        entropy_all[file_idx] = command_entropy
        command_entropy = [command_entropy]

        if saving:
            file_name = str(data_files[file_idx])
            file_name = file_name.replace("\\", "_")
            np.savetxt(str(analysis_path) + 'Entropy_' + str(file_name) + '.txt', command_entropy, fmt='%f')
    if saving:
        input_path = input_path.replace("\\", "")
        np.savetxt(str(analysis_path) + 'Entropy_' + str(input_path) + '.txt', entropy_all)
        return 0


def commands_max_freq(input_path, analysis_path, saving = False, plotting = False):

    data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]

    max_idx_all = []


    for file_idx in range(len(data_files)):
        data_wav = wave.open(str(data_files[file_idx]),'r')
        fs = data_wav.getframerate()

        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        power_spectrum = np.abs(np.fft.fft(raw_signal)) ** 2
        print(power_spectrum)

        time_step = 1 / fs
        freqs = np.fft.fftfreq(raw_signal.size, time_step)
        #idx = np.argsort(freqs)
        #freqs = freqs[idx]
        max_idx=np.argmax(power_spectrum)

        max_freq = freqs[max_idx]

        max_idx_all.append(max_freq)

        max_idx = [max_idx]

        if saving:
            file_name=str(data_files[file_idx])
            file_name=file_name.replace("\\","_")
            file_name = file_name.replace(".", "")
            np.savetxt(str(analysis_path)+'Max_freq_'+str(file_name)+'.txt', max_idx, fmt='%f')
    if saving:
        input_path = str(input_path)
        input_path = input_path.replace("\\", "")
        input_path = input_path.replace(".", "")
        file_path = open(str(analysis_path) + 'Max_freq_'+ input_path + '.txt', 'w')

        for line in max_idx_all:
                file_path.write("%f " % line)
                file_path.write("\n")
        #np.savetxt(str(analysis_path) + 'PSD_' + str(input_path) + '.txt', power_spectrum_all)


    return max_idx_all




