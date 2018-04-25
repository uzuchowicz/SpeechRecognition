from pydub import AudioSegment
import features_extraction as fextr
import classification as classf
from pylab import *
from pathlib import Path
import seaborn as sns
import preprocessing as prep

################################INFORMATION ABOUT DATA#######################################################
AudioSegment.converter = "C:\\Program Files\\ffmpeg-20180331-be502ec-win64-static\\bin\\ffmpeg.exe"

data_files = ['266766_23_K_21_1.wav', '266766_23_K_22_2.wav', '266766_23_K_9_3.wav', '266766_23_K_11_4.wav']
info_files = ['266766_23_K_21_1wav.txt', '266766_23_K_22_2wav.txt', '266766_23_K_9_3wav.txt', '266766_23_K_11_4wav.txt']
commands_path = "segmented_commands\\"
input_path = "recordings\\"
analysis_path = "analysis_coeffs\\"



###########################COMMANDS SEGMENTATION############################################################
result = prep.wav_files_segmentation(data_files, info_files, input_path, commands_path)
print(result)



################NEURAL NETWORK CLASSIFICATION ###############################################################

# coeffs, commands_groups = fextr.compute_training_coefficients(commands_path)
# sample = fextr.compute_sample_coefficients(commands_path + '_DO_1.wav')
# results = classf.neural_network_classification(sample, coeffs, commands_groups)
# print('results')
# print(results)



#################################KNN CLASIFICATION AND DISPLAYING RESULTS####################################
data_files = [f for f in Path(commands_path).glob('**/*.wav') if f.is_file()]
result_matrix = np.empty(shape=[0,len(data_files)])
coeffs, commands_groups = fextr.compute_training_coefficients(commands_path)
marks = np.full([len(data_files), len(data_files)], ' ')


for file_idx in range(len(data_files)):
    sample = data_files[file_idx]
    sample_features = fextr.compute_sample_coefficients(str(sample))
    classified_group, dist, ind = classf.knn_classification(sample_features, coeffs,
                                                                    commands_groups)
    marks[file_idx, ind[0]] = 'o'
    marks[file_idx, ind[1]] = 'o'

    result_matrix = np.concatenate((result_matrix, dist), axis=0)


cmap = plt.get_cmap('inferno', 30)
cmap.set_under('white')
cmap.set_over('yellow')

ax = sns.heatmap(result_matrix, cmap=cmap, annot=marks, fmt = '', annot_kws={"size": 6, "color":'g'})
plt.show()

#################################SIMPLE CLASIFICATION AND DISPLAYING RESULTS####################################
data_files = [f for f in Path(commands_path).glob('**/*.wav') if f.is_file()]
result_matrix = np.empty(shape=[0,len(data_files)])
coeffs, commands_groups = fextr.compute_training_coefficients(commands_path)
marks = np.full([len(data_files), len(data_files)], ' ')


for file_idx in range(len(data_files)):
    sample = data_files[file_idx]
    sample_features = fextr.compute_sample_coefficients(str(sample))
    classified_group, dist, ind = classf.minimum_features_distance_classification(sample_features, coeffs, commands_groups)

    marks[file_idx, ind[0]] = 'o'
    marks[file_idx, ind[1]] = 'o'

    result_matrix = np.concatenate((result_matrix, dist), axis=0)


cmap = plt.get_cmap('inferno', 30)
cmap.set_under('white')
cmap.set_over('yellow')


ax = sns.heatmap(result_matrix, cmap=cmap, annot=marks, fmt = '', annot_kws={"size": 6, "color":'g'})
plt.show()



#######################SAVING RESULTS ########################################################################
analysis_file = open('analysis_results\\Result_matrix.txt','w')
try:
    for line in result_matrix:
        for item in line:
            analysis_file.write("%f " % item)
        analysis_file.write("\n")
    print('Saving results - success!')
except:
    print('Error saving results!')