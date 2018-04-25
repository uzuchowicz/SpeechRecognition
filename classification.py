from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import copy


def neural_network_classification(sample_features, training_data_coeffs, training_data_groups):

    clf = neural_network_training(training_data_coeffs, training_data_groups)
    result = neural_network_prediction(clf, [sample_features])
    print(result)
    print('************** Confusion matrix **************')
    print(confusion_matrix([sample_features], result))
    print('************** Clasification report **************')
    print(classification_report([sample_features], result))

    return result


def neural_network_training(training_data_coeffs, training_data_groups):

    nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    nn_classifier.fit(training_data_coeffs, training_data_groups)

    return nn_classifier


def neural_network_prediction(clf, sample_data):
    label = clf.predict(sample_data)
    return label


def minimum_features_distance_classification(sample_features, training_data_coeffs, training_data_groups):

    distances = np.zeros(shape=np.shape(training_data_coeffs))

    # features normalization
    feature_max = np.amax(np.append(training_data_coeffs, sample_features), 0)
    feature_min = np.amin(np.append(training_data_coeffs, sample_features), 0)
    sample_features = (sample_features - feature_min)/(feature_max - feature_min)
    training_data_coeffs = (training_data_coeffs - feature_min)/(feature_max - feature_min)

    for feature_idx in range(len(sample_features)):

        for data_idx in range(len(training_data_coeffs)):
            distances[data_idx,feature_idx] = np.sqrt((sample_features[feature_idx] - training_data_coeffs[data_idx, feature_idx]) ** 2)

    results = np.sum(distances, axis=1)
    results_data = copy.deepcopy(results)
    group_idx = np.argpartition(results_data, 2)
    group_idx = group_idx[0:3]
    classified_group = training_data_groups[group_idx[1]]

    return classified_group, [results], group_idx


def knn_classification(sample_features, training_data_coeffs, training_data_groups):

    feature_max = np.amax(np.append(training_data_coeffs, sample_features), 0)
    feature_min = np.amin(np.append(training_data_coeffs, sample_features), 0)
    sample_features = (sample_features - feature_min) / (feature_max - feature_min)
    training_data_coeffs = (training_data_coeffs - feature_min) / (feature_max - feature_min)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(training_data_coeffs, training_data_groups)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(training_data_coeffs, training_data_groups)))

    result = knn.predict([sample_features])

    dist_2, ind_2 =knn.kneighbors([sample_features], n_neighbors=2, return_distance=True)
    ind_2=ind_2[0]
    dist_all, ind_all = knn.kneighbors([sample_features], n_neighbors=len(training_data_groups), return_distance=True)
    dist_all=dist_all[0]
    ind_all = ind_all[0]
    distances=np.zeros(shape=np.shape(dist_all))

    for i in range(len(training_data_groups)):
        distances[ind_all[i]]=dist_all[i]
    return result, [distances], ind_2







