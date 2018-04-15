
from pydub import AudioSegment
import features_extraction as fextr
import preprocessing as prep
import classification as classf
from pylab import *
from pathlib import Path
import seaborn as sns
import data_analysis as da

AudioSegment.converter = "C:\\Program Files\\ffmpeg-20180331-be502ec-win64-static\\bin\\ffmpeg.exe"


#substraction no sound signal?

# arCoeff(): Autorregresion coefficients with Burg order equal to 4
# correlation(): correlation coefficient between two signals
# meanFreq(): Weighted average of the frequency components to obtain a mean frequency
# skewness(): skewness of the frequency domain signal
# kurtosis(): kurtosis of the frequency domain signal
# bandsEnergy(): Energy of a frequency interval within the 64 bins of the FFT of each window.
# angle(): Angle between to vectors.


data_files = ['266766_23_K_21_1.wav', '266766_23_K_22_2.wav', '266766_23_K_9_3.wav', '266766_23_K_11_4.wav']
info_files = ['266766_23_K_21_1wav.txt', '266766_23_K_22_2wav.txt', '266766_23_K_9_3wav.txt', '266766_23_K_11_4wav.txt']
output_path = "segmented_commands\\"
input_path = "recordings\\"
analysis_path = "analysis_coeffs\\"


# prep.wav_files_segmentation(data_files, info_files, input_path, output_path)
# #PSD=prepr.commands_PSD(input_path, analysis_path, True)
#
#
# # data_filtr = prepr.data_filtering(data, fs, low_freq,high_freq, 2, True)
x = np.loadtxt('analysis_coeffs\\mean_recordings_266766_23_K_9_3.wav.txt', dtype=float)
#
coeffs, commands_groups = fextr.compute_training_coefficients(output_path)
sample = fextr.compute_sample_coefficients(output_path + '_DO_1.wav')
# #print(sample)
# training_data_groups = 2
results = classf.knn_classification(sample, coeffs, commands_groups)
print('results')
print(results)
# results = classf.neural_network_classification(sample, coeffs, commands_groups)
# print('results')
# print(results)

data_files = [f for f in Path(output_path).glob('**/*.wav') if f.is_file()]
result_matrix = np.empty(shape=[0,len(data_files)])
coeffs, commands_groups = fextr.compute_training_coefficients(output_path)
marks = np.full([40, 40], ' ')

# for file_idx in range(len(data_files)):
#     sample = data_files[file_idx]
#     sample_features = fextr.compute_sample_coefficients(str(sample))
#     classified_group, dist, ind = classf.minimum_features_distance_classification(sample_features, coeffs, commands_groups)
#     min_idx = da.second_min(results)
#     print(results)
#     marks[file_idx, min_idx] = 'X'
#
#     result_matrix = np.concatenate((result_matrix, [results]), axis=0)
# #
for file_idx in range(len(data_files)):
    sample = data_files[file_idx]
    sample_features = fextr.compute_sample_coefficients(str(sample))
    classified_group, dist, ind = classf.knn_classification(sample_features, coeffs, commands_groups)
    print('DIST')
    print(dist)
    marks[file_idx, ind] = 'X'

    result_matrix = np.concatenate((result_matrix, dist), axis=0)


cmap = plt.get_cmap('inferno', 30)
cmap.set_under('white')  # Colour values less than vmin in white
cmap.set_over('yellow')  # colour valued larger than vmax in red
print(type(marks))
print(marks)
#min_indexes = np.unravel_index(result_matrix.argmin(), result_matrix.shape)


#marks[min_indexes[0],min_indexes[1]] = 'X'


print(type(marks[1,2]))

ax = sns.heatmap(result_matrix, cmap=cmap, annot=marks, fmt = '', annot_kws={"size": 2})
plt.show()




analysis_path = open('Result_matrix.txt','w')
for line in result_matrix:
    for item in line:
        analysis_path.write("%f " % item)
    analysis_path.write("\n")