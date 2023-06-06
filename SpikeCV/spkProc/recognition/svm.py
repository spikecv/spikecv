from sklearn.svm import SVC, LinearSVC
import numpy as np
import pickle

class TemporalFilteringSVM:
    '''
    Temporal filtering support vector machine

    It consists of two parts: time domain filter and support vector machine. Recognition of time-domain filtering features of spikes by SVM

    '''
    def __init__(self, filter, **svm_kwargs):
        '''

        :param filter: temporal filter
        :param penalty: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param loss: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param dual: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param tol: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param C: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param multi_class: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param fit_intercept: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param intercept_scaling: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param class_weight: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param verbose: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param random_state: SVM parameters. For details, see sklearn.svm.LinearSVC
        :param max_iter: SVM parameters. For details, see sklearn.svm.LinearSVC
        '''
        self.filter = filter
        self.svm = LinearSVC(**svm_kwargs)

    def extract_feature(self, data):
        '''
        Extract filter features

        :param data: spike streams
        :return: Filtered features
        '''
        assert(isinstance(data, np.ndarray))
        n_samples = data.shape[0]
        features = self.filter(data)
        return features.reshape(n_samples, -1)

    def fit(self, train_data, train_label):
        '''
        Support Vector Machine Fitting

        :param train_data: training dataset
        :param train_label: labels
        :return: SVM
        '''
        train_feature = self.extract_feature(train_data)
        self.svm.fit(train_feature, train_label)
        return self.svm

    def predict(self, test_data):

        test_feature = self.extract_feature(test_data)
        pred = self.svm.predict(test_feature)
        return pred
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            res = pickle.dump(self.svm, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.svm = pickle.load(f)