import pandas as pd
# from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt

# import train and test data
# removing redundant columns
train = pd.read_csv(
    "./Data/Train.csv").drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
pb_test = pd.read_csv(
    "./Data/PB_Test.csv").drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
yt_test = pd.read_csv(
    "./Data/YT_Test.csv").drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
test = [pb_test, yt_test]

# column 40 has the 12 classes labeled as 0-11 in alphabetical order of langauges
test_label = [pb_test['40'], yt_test['40']]
covariance_type = ['full', 'diag']  # storing covariance types

# GMM_Model
for cov_type in covariance_type:
    for t in range(len(test)):
        gmm_accuracies = []
        for q in range(1, 7):  # q= number of components
            gmm = []
            likelihood = []
            prior = []
            for i in range(0, 12):
                class_train = train[train['40'] == i].drop(['40'], axis=1)
                prior.append(len(class_train)/len(train))  # prior probability
                gmm_class = GaussianMixture(n_components=q, covariance_type=cov_type).fit(
                    class_train)  # building gmm class wise
                gmm.append(gmm_class)  # storing all the class wise models
                # .score_samples gives the likelihood of test samples
                likelihood.append(
                    np.exp(gmm_class.score_samples(test[t].drop(['40'], axis=1))))

            pred = []
            for i in range(len(test[t].drop(['40'], axis=1))):
                prob = []
                total_prob = 0
                for j in range(12):
                    total_prob = total_prob + \
                        ((likelihood[j][i] * prior[j])/((likelihood[j]
                         [i]*prior[j]) + (likelihood[j][i]*prior[j])))
                for j in range(12):
                    prob.append(likelihood[j][i] * prior[j])
                # assigning that class which has maximum posterior probability
                pred.append(np.argmax(prob))

            conf_matrix = confusion_matrix(test_label[t], pred)
            accuracy = np.round(accuracy_score(test_label[t], pred), 3)
            gmm_accuracies.append(accuracy)  # storing accuracies for graph
            if (t == 0):
                print(
                    f"The accuracy for pb_test q = {q} is {accuracy} in case of {cov_type} covariance matrix")
            else:
                print(
                    f"The accuracy for yt_test q = {q} is {accuracy} in case of {cov_type} covariance matrix")
            print("Confusion Matrix:")
            print(conf_matrix)
        components = [i for i in range(1, 7)]
        plt.plot(components, gmm_accuracies)
        plt.title(
            f"GMM Accuracies vs Number of Components ({cov_type} covariance matrix)")
        plt.show()

# UBM-Model
for cov_type in covariance_type:
    for t in range(len(test)):
        ubm_accuracies = []
        r = 0.7  # relevance factor
        for q in range(1, 4):  # q= number of components
            ubm = []
            likelihood = []
            prior = []
            for i in range(0, 12):
                UBM = GaussianMixture(n_components=q, covariance_type=cov_type)
                UBM.fit(train.drop(['40'], axis=1))  # common ubm
                uk = UBM.means_[0]  # mean from the UBM
                class_train = train[train['40'] == i].drop(['40'], axis=1)
                prior.append(len(class_train)/len(train))  # prior probability
                # ak=nk/(r+nk), nk= the effective number of examples from the ith component
                ak = len(class_train)/(len(class_train) + r)
                xki = class_train.mean().values  # partial estimate of the mean vector
                # adapting ubm to get MAP estimates of the mean vectors
                uki = ak * xki + (1 - ak) * uk
                UBM.means_ = np.array([uki])  # adapting mean
                ubm.append(UBM)
                likelihood.append(
                    np.exp(UBM.score_samples(test[t].drop(['40'], axis=1))))

            pred = []
            for i in range(len(test[t].drop(['40'], axis=1))):
                prob = []
                # total_prob = 0
                # for j in range(12):
                #     total_prob = total_prob + ( (likelihood[j][i] * prior[j])/((likelihood[j][i]*prior[j]) + (likelihood[j][i]*prior[j])) )
                for j in range(12):
                    prob.append(likelihood[j][i] * prior[j])
                # assigning that class which has maximum posterior probability
                pred.append(np.argmax(prob))

            conf_matrix = confusion_matrix(test_label[t], pred)
            accuracy = np.round(accuracy_score(test_label[t], pred), 3)
            ubm_accuracies.append(accuracy)
            if (t == 0):
                print(
                    f"The accuracy for pb_test q = {q} is {accuracy} in case of {cov_type} covariance matrix")

            else:
                print(
                    f"The accuracy for yt_test q = {q} is {accuracy} in case of {cov_type} covariance matrix")
            print("Confusion Matrix:")
            print(conf_matrix)
        components = [i for i in range(1, 4)]
        plt.plot(components, ubm_accuracies)
        plt.title(
            f"UBM Accuracies vs Number of Components ({cov_type} covariance matrix)")
        plt.show()
