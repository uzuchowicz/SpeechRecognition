from sklearn.neural_network import MLPClassifier
from pathlib import Path
import segmentation as prepr
import numpy as np



def compute_coefficients(input_path, analysis_path = 0, saving = True):
    data_files = [f for f in Path(input_path).glob('**/*.wav') if f.is_file()]

    coeffs_all = np.zeros(shape = (4, len(data_files)))
    #print(len(data_files))
    #print(np.shape(coeffs_all))

    command_PSD = prepr.commands_mean_std_min_max(input_path)
    print(np.shape(command_PSD))
    print(command_PSD)
    print(type(command_PSD))
    coeffs_all[:] = command_PSD

                # np.savetxt(str(analysis_path)+'PSD_'+str(file_name)+'.txt', data, fmt='%f')
    if saving:
        input_path = str(input_path)
        input_path = input_path.replace("\\", "")
        input_path = input_path.replace(".", "")
        file_path = open(str(analysis_path) + 'PSD_' + input_path + '.txt', 'w')

        for line in coeffs_all:
            for item in line:
                file_path.write("%f " % item)
            file_path.write("\n")
            # np.savetxt(str(analysis_path) + 'PSD_' + str(input_path) + '.txt', power_spectrum_all)

    return coeffs_all
def neural_network_training(data_coeffs, data_group):
       nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

       nn_classifier.fit(data_coeffs, data_group)
       # MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       # beta_1=0.9, beta_2=0.999, early_stopping=False,
       # epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       # learning_rate_init=0.001, max_iter=200, momentum=0.9,
       # nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       # solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       # warm_start=False)
       return nn_classifier


def neural_network_prediction(clf,data):
       labels = clf.predict([[2., 2.], [-1., -2.]])

       # MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       # beta_1=0.9, beta_2=0.999, early_stopping=False,
       # epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       # learning_rate_init=0.001, max_iter=200, momentum=0.9,
       # nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       # solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       # warm_start=False)
       return labels
