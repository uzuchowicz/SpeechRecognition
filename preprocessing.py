import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment
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


def data_filtering(data, fs, low_freq,high_freq, order, plotting = False):

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


def match_target_amplitude(sound, target_dBFS):

    change_in_dBFS = target_dBFS - sound.dBFS

    return sound.apply_gain(change_in_dBFS)


def wav_files_segmentation(data_files, info_files, input_path, output_path):
    try:
        for file_idx in range(len(data_files)):

            commands_file = open(input_path+info_files[file_idx],"r")
            commands = commands_file.readlines()

            commands_file.close()

            nb_command = 0
            for line in commands:
                data_wav = wave.open(input_path + data_files[file_idx], 'r')

                command = commands[nb_command].split()
                start_time = float(command[0])*1000
                end_time = float(command[1])*1000
                nb_command += 1

                command_audio = AudioSegment.from_wav(input_path+data_files[file_idx])

                command_audio = command_audio[start_time:end_time]

                command_audio.export(output_path + '_' + command[2]+ '_' + str(file_idx)+'.wav', format="wav")
        result = 'Command segmentation - success!'
    except:
        result = 'Error occurs'

    return result
