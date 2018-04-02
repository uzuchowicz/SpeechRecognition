import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment
import segmentation as seg
import pydub
from pathlib import Path

def read_wav_file(data_file, input_path):
    data_wav = wave.open(input_path + data_file, 'r')

    # Extract Raw Audio from Wav File
    raw_signal = data_wav.readframes(-1)
    Fs=data_wav.getframerate()

    raw_signal = np.fromstring(raw_signal, 'Int16')
    timeline = np.arange(0, 1 / Fs * len(raw_signal), 1 / Fs)
    plt.figure
    plt.title('Signal of spoken commands from wav file')
    plt.plot(timeline, raw_signal)
    plt.grid(True)
    plt.show()

    return raw_signal

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


def commands_SPD(input_path):

    data_files=[f for f in Path(input_path).glob('**/*.wav') if f.is_file()]
    for file_idx in range(len(data_files)):
        print(data_files[file_idx])
        data_wav = wave.open(str(data_files[file_idx]),'r')
        Fs = data_wav.getframerate()

        raw_signal = data_wav.readframes(-1)
        raw_signal = np.fromstring(raw_signal, 'Int16')

        power_spectrum = np.abs(np.fft.fft(raw_signal)) ** 2
        print(power_spectrum)
        print(type(power_spectrum))
        print(np.size(power_spectrum))

        time_step = 1 / Fs
        freqs = np.fft.fftfreq(raw_signal.size, time_step)
        idx = np.argsort(freqs)

        plt.plot(freqs[idx], power_spectrum[idx])
        plt.show()