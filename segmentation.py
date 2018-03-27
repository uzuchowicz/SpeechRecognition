
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment
import sys

data_files=['266766_23_K_21_1.wav', '266766_23_K_22_2.wav', '266766_23_K_9_3.wav', '266766_23_K_11_4.wav']
info_files=['266766_23_K_21_1wav.txt', '266766_23_K_22_2wav.txt', '266766_23_K_9_3wav.txt', '266766_23_K_11_4wav.txt']


for file_idx in range(len(data_files)):
    data_wav = wave.open(data_files[file_idx],'r')

    #Extract Raw Audio from Wav File
    raw_signal = data_wav.readframes(-1)
    raw_signal = np.fromstring(raw_signal, 'Int16')

    #If Stereo
    #if data_wav.getnchannels() == 2:
    #    print ('Just mono files')
     #   sys.exit(0)

    plt.figure(file_idx)
    plt.title('Signal of spoken commands from wav file')
    plt.plot(raw_signal)
    plt.show()



    a = open(info_files[file_idx],"r")
    b = a.readlines()
    a.close()
    path_word=['zapal.wav', 'swiatlo.wav', 'w.wav', 'kuchni.wav', 'otworz.wav', 'drzwi.wav', 'do.wav', 'garazu.wav', 'wlacz.wav', 'zmywarke.wav', 'wylacz.wav', 'telewizor.wav', 'podnies.wav', 'rolety.wav', 'sypialni.wav', 'zamknij.wav', 'brame.wav', 'zwieksz.wav', 'o.wav', 'jeden.wav',
               'stopien.wav', 'zarkec.wav', 'wode.wav', 'lazience.wav', 'ustaw.wav', 'alarm.wav', 'przycisz.wav', 'radio.wav', 'zmien.wav', 'kanal.wav', 'podlej.wav', 'kwiatki.wav', 'zaparz.wav', 'kawe.wav', 'alarm.wav', 'zagotuj.wav']

    count = -1
    for line in b:
        count += 1
        if count >= 0:
            d = b[count].split()
            print (d)
            t1=d[0]
            t2=d[1]
            t1=float(t1)
            t2=float(t2)
            t1=t1*1000 # works in miliseconds
            t2=t2*1000
            print (t1)
            print (t2)


        newAudio = AudioSegment.from_wav(data_files[file_idx])
        newAudio = newAudio[t1:t2]

        newAudio.export(d[2], format="wav")

