import matplotlib.pyplot as plt
import numpy as np
import wave
import preprocessing as prep
from pathlib import Path
import pylab
import matplotlib.cm as cm
import scipy as scipy
import pandas
import pywt
import scipy.stats as stats


def compute_coefficients(command_signal, fs):

    command_coeffs = []
    command_coeffs.extend(commands_mean_std_min_max(command_signal))
    command_coeffs.append(float(len(command_signal)))

    # part of coefficients is calculated after normalization
    prep.normalization(command_signal)

    command_coeffs.append(commands_max_freq(command_signal, fs))
    command_coeffs.append(commands_mean_freq(command_signal, fs))
    command_coeffs.append(commands_energy(command_signal))
    command_coeffs.append(commands_iqr(command_signal))
    command_coeffs.append(commands_sma(command_signal, fs))
    command_coeffs.append(commands_skewness_freq(command_signal, fs))
    command_coeffs.append(commands_kurtosis_freq(command_signal, fs))

    return command_coeffs

def compute_training_coefficients(input_path, analysis_path=0, saving=False):

    if input_path[-4:] == '.wav':
        data_files = [input_path] #.split('\\')[-1]]
    else:
        data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]

    # matrix_coeffs = np.matrix([])
    matrix_coeffs = np.zeros((len(data_files), 12))
    commands_groups = list()

    for file_idx in range(len(data_files)):
        data_wav = wave.open(str(data_files[file_idx]), 'r')
        fs = data_wav.getframerate()

        raw_signal = data_wav.readframes(-1)
        command_signal = np.fromstring(raw_signal, 'Int16')
        command_signal = prep.data_filtering(command_signal, fs, 30, 3200, 2, False)

        command_coeffs = compute_coefficients(command_signal, fs)

        matrix_coeffs[file_idx, :] = command_coeffs
        command_group = str(data_files[file_idx]).split('_')[-2]
        commands_groups.append(command_group)
    # np.append(matrix_coeffs,[command_coeffs], axis=1)

    # np.savetxt(str(analysis_path)+'PSD_'+str(file_name)+'.txt', data, fmt='%f')
    if saving:
        input_path = str(input_path)
        input_path = input_path.replace("\\", "")
        input_path = input_path.replace(".", "")
        file_path = open(str(analysis_path) + 'Coeffs_' + input_path + '.txt', 'w')

        for line in matrix_coeffs:
            for item in line:
                file_path.write("%f " % item)
            file_path.write("\n")
            # np.savetxt(str(analysis_path) + 'PSD_' + str(input_path) + '.txt', power_spectrum_all)

    return matrix_coeffs, commands_groups


def compute_sample_coefficients(input_path, analysis_path=0, saving=False):

    if input_path[-4:] == '.wav':
        data_files = [input_path] #.split('\\')[-1]]
    else:
        data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]

    # matrix_coeffs = np.matrix([])
    matrix_coeffs = np.zeros((len(data_files), 12))

    for file_idx in range(len(data_files)):
        data_wav = wave.open(str(data_files[file_idx]), 'r')
        fs = data_wav.getframerate()
        command_coeffs = []
        raw_signal = data_wav.readframes(-1)
        command_signal = np.fromstring(raw_signal, 'Int16')
        low_freq=30
        high_freq=3200
        order = 2

        command_signal = prep.data_filtering(command_signal, fs, low_freq, high_freq, order, False)

        command_coeffs = compute_coefficients(command_signal, fs)


    # np.append(matrix_coeffs,[command_coeffs], axis=1)

    # np.savetxt(str(analysis_path)+'PSD_'+str(file_name)+'.txt', data, fmt='%f')
    if saving:
        input_path = str(input_path)
        input_path = input_path.replace("\\", "")
        input_path = input_path.replace(".", "")
        file_path = open(str(analysis_path) + 'Coeffs_' + input_path + '.txt', 'w')

        for line in matrix_coeffs:
            for item in line:
                file_path.write("%f " % item)
            file_path.write("\n")
            # np.savetxt(str(analysis_path) + 'PSD_' + str(input_path) + '.txt', power_spectrum_all)

    return command_coeffs

def commands_psd(command_signal, fs, plotting = False):

    power_spectrum = np.abs(np.fft.fft(command_signal)) ** 2

    time_step = 1 / fs
    freqs = np.fft.fftfreq(command_signal.size, time_step)
    idx = np.argsort(freqs)

    if plotting:
        plt.plot(freqs[idx], power_spectrum[idx])
        plt.show()

    return power_spectrum, freqs


def commands_dwt(command_signal, plotting = False):
    # Discrete Wavelet Transform

    dwt_coeffs = pywt.dwt(command_signal, 'db5')

    if plotting:
        pylab.gray()
        scalogram(command_signal)
        pylab.show()

    return dwt_coeffs



def scalogram(data):
    # Make a scalogram given an MRA tree.

    xrange = pylab.arange(0, 1, 1. /len(data))
    data = np.asarray(data)
    wavelet = 'db5'
    level = 1
    order = "freq"
    interpolation = 'nearest'
    cmap = cm.cool # settings colormap?

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


def commands_mean_std_min_max(command_signal):

    # mean(): Mean value
    # std(): Standard deviation
    # mad(): Median absolute deviation
    # max(): Largest value in array
    # min(): Smallest value in array

    command_mean = np.array(np.mean(command_signal))
    command_std = np.std(command_signal)
    command_min = np.min(command_signal)
    command_max = np.max(command_signal)
    command_stats = np.array([command_mean, command_std , command_min, command_max])

    return command_stats


def commands_sma(command_signal, fs):

    # signal magnitiude area

    sum = 0
    timeline = np.arange(0, 1 / fs * len(command_signal), 1 / fs)

    for sample_idx in range(len(command_signal)):
        sum += (abs(timeline[sample_idx]) + abs(command_signal[sample_idx]))

    command_sma = sum / len(command_signal)

    return command_sma


def commands_energy(command_signal):
    # energy(): Energy measure. Sum of the squares divided by the number of values.

    sum = 0

    for sample_idx in range(len(command_signal)):
        sum += (command_signal[sample_idx]) ** 2

    command_energy = sum / len(command_signal)

    return command_energy


def commands_iqr(command_signal):

    # Interquartile range
    q75, q25 = np.percentile(command_signal, [75, 25])
    command_iqr = float(q75 - q25)

    return command_iqr


# signal entropy

def commands_entropy(command_signal):

    # command_entropy = get_ApEn(raw_signal)
    p_data = pandas.value_counts(command_signal) / len(command_signal)  # calculates the probabilities
    command_entropy = scipy.stats.entropy(p_data)

    return command_entropy


def command_entropy(command_signal, m=2, r_mlp=0.2):

    r = r_mlp * np.nanstd(command_signal)
    N = len(command_signal)
    phi_m_r = np.zeros(2)

    for n in range(2):
        m = m + n
        pm = np.zeros((N - m + 1, m))
        # Pm vectors
        for j in range(N - m + 1):
            for i in range(m):
                pm[j, i] = command_signal[j + i]
        # calculate distances vector
        pm_distances = np.zeros((N - m + 1, N - m + 1))
        for i in range(N - m + 1):
            for j in range(N - m + 1):
                dist = np.zeros(m)
                for k in range(m):
                    dist[k] = np.abs(pm[j, k] - pm[i, k])
                    pm_distances[i, j] = np.nanmax(dist)
                    pm_distances[j, i] = np.nanmax(dist)
                    # comparision with tolerance
        pm_similarity = pm_distances > r
        # function Cmr
        c_m_r = np.zeros(N - m + 1)
        for i in range(N - m + 1):
            n_i = np.nansum(pm_similarity[i])
            c_m_r[i] = float(n_i) / float(N)
        # phi parameter- Cmr logarithms mean
        phi_m_r[n] = np.nanmean(np.log(c_m_r))
    command_entropy = np.abs(phi_m_r[0] - phi_m_r[1])

    return command_entropy

def commands_autoreggression_coeffs(command_signal):

    # Levinson-Durbin algorithm for solving the Hermitian Toeplitz system of Yule-Walker equations in the AR estimation
    # problem

    p_data = pandas.value_counts(command_signal) / len(command_signal)  # calculates the probabilities
    command_entropy = scipy.stats.entropy(p_data)
    return 0


def commands_max_freq(command_signal, fs):

    power_spectrum = np.abs(np.fft.fft(command_signal)) ** 2

    time_step = 1 / fs
    freqs = np.fft.fftfreq(command_signal.size, time_step)
    # idx = np.argsort(freqs)
    # freqs = freqs[idx]
    max_idx=np.argmax(power_spectrum)

    command_max_freq = freqs[max_idx]

    return command_max_freq

def commands_mean_freq(command_signal, fs):

    power_spectrum = np.abs(np.fft.fft(command_signal)) ** 2

    time_step = 1 / fs
    freqs = np.fft.fftfreq(command_signal.size, time_step)
    # idx = np.argsort(freqs)
    # freqs = freqs[idx]


    command_mean_freq = sum(freqs*power_spectrum)/sum(power_spectrum)

    return command_mean_freq

def commands_mean_freq(command_signal, fs):

    power_spectrum = np.abs(np.fft.fft(command_signal)) ** 2

    time_step = 1 / fs
    freqs = np.fft.fftfreq(command_signal.size, time_step)
    # idx = np.argsort(freqs)
    # freqs = freqs[idx]


    command_mean_freq = sum(freqs*power_spectrum)/sum(power_spectrum)

    return command_mean_freq

def commands_kurtosis_freq(command_signal, fs):

    power_spectrum = np.abs(np.fft.fft(command_signal)) ** 2

    time_step = 1 / fs
    freqs = np.fft.fftfreq(command_signal.size, time_step)
    # idx = np.argsort(freqs)
    # freqs = freqs[idx]


    command_kurtosis_freq = stats.kurtosis(power_spectrum)

    return command_kurtosis_freq

def commands_skewness_freq(command_signal, fs):

    power_spectrum = np.abs(np.fft.fft(command_signal)) ** 2

    time_step = 1 / fs
    freqs = np.fft.fftfreq(command_signal.size, time_step)
    # idx = np.argsort(freqs)
    # freqs = freqs[idx]


    command_skewness_freq = stats.skew(power_spectrum)

    return command_skewness_freq

