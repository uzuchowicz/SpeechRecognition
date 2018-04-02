
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from pydub import AudioSegment


data_files=['266766_23_K_21_1.wav', '266766_23_K_22_2.wav', '266766_23_K_9_3.wav', '266766_23_K_11_4.wav']
info_files=['266766_23_K_21_1wav.txt', '266766_23_K_22_2wav.txt', '266766_23_K_9_3wav.txt', '266766_23_K_11_4wav.txt']
output_path="segmented_commands\\"

spf = wave.open('266766_23_K_21_1.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')


#If Stereo
if spf.getnchannels() == 2:
    print ('Just mono files')
    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()


path='266766_23_K_21_1wav.txt'
a = open(path,"r")
b = a.readlines()
a.close()
path_word=['otworz.wav', 'zamknij.wav', 'zrob.wav', 'nastroj.wav', 'wlacz.wav', 'wylacz.wav', 'muzyke.wav', 'swiatlo.wav', 'zapal.wav', 'podnies.wav', 'rolety.wav', 'telewizor.wav']
output_path="segmented_commands\\"

count = -1
for line in b:
    count += 1
    if count >= 0:      
        d = b[count].split()
        print(d)
        t1=d[0]
        t2=d[1]
        t1=float(t1)
        t2=float(t2)
        t1=t1*1000 # works in miliseconds
        t2=t2*1000
        print(t1)
        print(t2)
 
   
    newAudio = AudioSegment.from_wav("266766_23_K_21_1.wav")
    newAudio = newAudio[t1:t2]
    
    newAudio.export(d[2], format="wav")

