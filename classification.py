from sklearn.neural_network import MLPClassifier
from pathlib import Path
import features_extraction as fextr
import numpy as np



def neural_network_training(training_data_coeffs, training_data_groups):
       nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

       nn_classifier.fit(training_data_coeffs, training_data_groups)
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

def minimum_features_distance_classification(sample_features, training_data_coeffs, training_data_groups):

    distances = np.zeros(shape=np.shape(training_data_coeffs))
    for feature_idx in range(len(sample_features)):

        for data_idx in range(len(training_data_coeffs)):
            distances[data_idx,feature_idx] = (sample_features[feature_idx] - training_data_coeffs[data_idx,feature_idx]) ** 2

    results = np.sum(distances, axis=1)
    group_idx = np.argmin(results)
    classified_group = training_data_groups[group_idx]

    return classified_group, results


