# thank you Shiv @https://github.com/shivmaheshwari/HideAndSeek

import csv # for reading the csv file
import sys # for max integer num in forward stepwise feature selection
import pandas as pd # uses dataframes from pandas
import numpy as np # used for arrays
from sklearn.ensemble import RandomForestClassifier #  for random forests
from sklearn import svm # used for creating mortality models --> Support Vector Machine algorithm
from sklearn.neighbors import KNeighborsClassifier # for KNN algorithm
from sklearn.linear_model import LogisticRegression # for logistic regression algorithm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics # for accuracy calculation
from sklearn.ensemble import RandomForestClassifier # used for creating FSS score predictors --> Random Forest algorithm
from sklearn.preprocessing import LabelEncoder # for converting the strs to ints
import statistics # used for calculating standard deviation
from sklearn.metrics import hinge_loss # used to calculating the hinge loss
from sklearn.model_selection import KFold # for implementing k-folds
import matplotlib.pyplot as plt # used for plotting the graphs
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler # used to allow SVM to run faster


# FUNCTIONS:

# creates the confusion matrix array by calculating the true and false positives and negatives
def create_conf_matrix(y_preds, y_reals):
	TP = 0
	FN = 0
	FP = 0
	TN = 0
	falsePos = []
	falseNeg = []

	for i in range(len(y_preds)):
		pred_val = y_preds[i]
		test_val = y_reals[i]

		if test_val == 1 and pred_val == 1:
			TP += 1
		if test_val == 1 and pred_val == 0:
			FN += 1
		if test_val == 0 and pred_val == 1:
			FP += 1
		if test_val == 0 and pred_val == 0:
			TN += 1

	return [TP, FN, FP, TN]

# calculates the accuracy of mortality model results
def accuracy(matrix):
	TP = matrix[0]
	TN = matrix[3]
	acc = (TP + TN) / (matrix[0] + matrix[1] + matrix[2] + matrix[3])
	return acc

# calculates the sensitivity of mortality model results
def sensitivity(matrix):
	TP = matrix[0]
	FN = matrix[1]
	sens = TP / (TP + FN)
	return sens

# calculates the specificity of mortality model results
def specificity(matrix):
	TN = matrix[3]
	FP = matrix[2]
	spec = TN / (TN + FP)
	return spec


# MAIN:

dataset = pd.read_csv('ModelData.csv') # dataset of confidence levels
dataset = dataset.replace(np.nan, 0)
dataset = np.array(dataset)
target_data = dataset[:, 0] # UPDATE!
dataset = dataset[:, [1, 2, 3]]
num_entries = len(dataset)

# definitions...
labels = ["ID", "Thermal", "OpenFace", "Kinect"]
classes = ["0", "1"]


kfold = KFold(n_splits = 10, shuffle = True, random_state = 100) # creates 10-fold splits

final_truths = []
final_preds1 = []
final_preds2 = []
final_preds3 = []
final_preds4 = []
# final_preds5 = []

proba_preds1 = []
proba_preds2 = []
proba_preds3 = []
proba_preds4 = []
# proba_preds5 = []

for train_index, test_index in kfold.split(dataset):
	x_train, x_test = dataset[train_index], dataset[test_index]
	y_train, y_test = target_data[train_index], target_data[test_index]


	classifier1 = RandomForestClassifier(max_depth = 6, min_samples_leaf = 3, min_samples_split = 10, class_weight = 'balanced', bootstrap = False, max_features = "auto", random_state = 100)
	classifier2 = svm.SVC(kernel = 'linear', random_state = 100)
	classifier2Prob = svm.SVC(kernel = 'linear', random_state = 100, probability = True)
	classifier3 = LogisticRegression(random_state = 100)
	classifier4 = KNeighborsClassifier(n_neighbors = 2, weights = 'distance')
	# classifier5 = MLPClassifier(random_state = 100)
	# print("Models Created . . .")

	scaling = MinMaxScaler(feature_range = (-1,1)).fit(x_train)
	x_trainSVM = scaling.transform(x_train)
	x_testSVM = scaling.transform(x_test)

	classifier1.fit(x_train, y_train.ravel())
	classifier2.fit(x_trainSVM, y_train.ravel())
	classifier2Prob.fit(x_trainSVM, y_train.ravel())
	classifier3.fit(x_train, y_train.ravel())
	classifier4.fit(x_train, y_train.ravel())
	# classifier5.fit(x_train, y_train.ravel())
	# print("Models Trained")

	y_pred1 = classifier1.predict(x_test)
	y_pred2 = classifier2.predict(x_testSVM)
	y_pred3 = classifier3.predict(x_test)
	y_pred4 = classifier4.predict(x_test)
	# y_pred5 = classifier5.predict(x_test)
	# print("Models Tested - Normal")

	y_proba_pred1 = classifier1.predict_proba(x_test)[:,1]
	y_proba_pred2 = classifier2Prob.predict_proba(x_testSVM)[:,1]
	y_proba_pred3 = classifier3.predict_proba(x_test)[:,1]
	y_proba_pred4 = classifier4.predict_proba(x_test)[:,1]
	# y_proba_pred5 = classifier5.predict_proba(x_test)[:,1]
	# print("Models Tested - Probability")

	for i in range(len(x_test)):
		final_preds1.append(y_pred1[i])
		final_preds2.append(y_pred2[i])
		final_preds3.append(y_pred3[i])
		final_preds4.append(y_pred4[i])
		# final_preds5.append(y_pred5[i])

		proba_preds1.append(y_proba_pred1[i])
		proba_preds2.append(y_proba_pred2[i])
		proba_preds3.append(y_proba_pred3[i])
		proba_preds4.append(y_proba_pred4[i])
		# proba_preds5.append(y_proba_pred5[i])

		final_truths.append(y_test[i])



# creates ROC curve and calculates AUROC
fpr1, tpr1, thresholds1 = metrics.roc_curve(final_truths, proba_preds1)
fpr2, tpr2, thresholds1 = metrics.roc_curve(final_truths, proba_preds2)
fpr3, tpr3, thresholds1 = metrics.roc_curve(final_truths, proba_preds3)
fpr4, tpr4, thresholds1 = metrics.roc_curve(final_truths, proba_preds4)
# fpr5, tpr5, thresholds1 = metrics.roc_curve(final_truths, proba_preds5)

auc1 = metrics.roc_auc_score(final_truths, final_preds1)
auc2 = metrics.roc_auc_score(final_truths, final_preds2)
auc3 = metrics.roc_auc_score(final_truths, final_preds3)
auc4 = metrics.roc_auc_score(final_truths, final_preds4)
# auc5 = metrics.roc_auc_score(final_truths, final_preds5)

plt.plot(fpr1, tpr1, label = '%s (area = %0.2f)' % ('Random Forest', auc1))
plt.plot(fpr2, tpr2, label = '%s (area = %0.2f)' % ('Support Vector Machine', auc2))
plt.plot(fpr3, tpr3, label = '%s (area = %0.2f)' % ('Logistic Regression', auc3))
plt.plot(fpr4, tpr4, label = '%s (area = %0.2f)' % ('K-Nearest Neighbors', auc4))
# plt.plot(fpr5, tpr5, label = '%s ROC (area = %0.2f)' % ('Multi-Layer Perceptron', auc5))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curves for Various Models')
plt.legend(loc = "lower right")
plt.show()


# fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (60, 15), dpi = 400)
# tree.plot_tree(classifier.estimators_[0], feature_names = labels, class_names = classes, filled = True, impurity = False, precision = 1, fontsize = 10)
# fig.savefig('decisionTree.png')