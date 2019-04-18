import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import itertools



# import CSV data
data = pd.read_csv("creditcard.csv")


# Normalize the 'Amount' feature and remove unwanted 'Time' features
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)


# Solving sample imbalance problem by using downsampling strategy
X = data.ix[:, data.columns != 'Class']
Y = data.ix[:, data.columns == 'Class']


# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)


# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index


# Out of the indices we picked, randomly select "x" number(number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)


# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])


# Under sample dataset
under_sample_data = data.iloc[under_sample_indices, :]


X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
Y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# Split data into trainging set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


print("")
print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total member of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
print("")
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(X_undersample, Y_undersample, test_size=0.3, random_state=0)
print("Number transactions undersample train dataset: ", len(X_train_undersample))
print("Number transactions undersample test dataset: ", len(X_test_undersample))
print("Total member of undersample transactions: ", len(X_train_undersample)+len(X_test_undersample))
print("")

# Recall = TP/(TP+FN)
# K-cross validation to choose the best k
def printing_KFold_scores(X_train_data, Y_train_data):
	kf = KFold(5, shuffle=False)

	# Different C parameters
	c_param_range = [0.01, 0.1, 1, 10, 100]

	result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
	result_table['C_parameter'] = c_param_range

	# the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
	j = 0
	for c_param in c_param_range:
		print("--------------------------------------------------")
		print("C parameter: ", c_param)
		print("--------------------------------------------------")
		print("")

		result_accs = []
		for iteration, indices in enumerate(kf.split(Y_train_data), start=1):

			# Call the logistic regeression model with a certain C parameter
			lr = LogisticRegression(C=c_param, solver='liblinear', penalty='l2')

			# Use the training data to fit the model. In this case, we use the portion of the fold to train the model
			# with indices[0]. We then predict on the portion assigned as the test cross validation with indices[1]
			lr.fit(X_train_data.iloc[indices[0], :], Y_train_data.iloc[indices[0], :].values.ravel())

			# Predict values using the test indices in the training data
			Y_pred_undersample = lr.predict(X_train_data.iloc[indices[1], :].values)

			# Calculate the recall score and append it to a list for recall scores representing the current c_parameter
			recall_acc = recall_score(Y_train_data.iloc[indices[1], :].values, Y_pred_undersample)
			result_accs.append(recall_acc)
			print('Iteration', iteration, ': recall score = ', recall_acc)

		# The mean value of those recall scores in the metric we want to save and get hold of.
		result_table.ix[j, 'Mean recall score'] = np.mean(result_accs)
		j += 1
		print("")
		print("Mean recall score: ", np.mean(result_accs))
		print("")

	result_table['Mean recall score'] = result_table['Mean recall score'].astype('float64')
	best_c = result_table.loc[result_table['Mean recall score'].idxmax()]['C_parameter']

	# Finally, we can check with C parameter is the best among the chosen
	print("********************************************************************************")
	print("Best model to choose from cross validation is with C parameter = ", best_c)
	print("********************************************************************************")

	return best_c


# Get beat Regularization parameter
best_c = printing_KFold_scores(X_train_undersample, Y_train_undersample)
print(best_c)
print("")


# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	:param cm:
	:param classes:
	:param title:
	:param cmap:
	:return:
	"""
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment='center',
				 color='white' if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel("True Label")
	plt.xlabel("Predicted Label")


lr = LogisticRegression(C=best_c, solver='liblinear', penalty='l2')
lr.fit(X_train_undersample, Y_train_undersample.values.ravel())
Y_pred_undersample = lr.predict(X_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test_undersample, Y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset(undersample test set): ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

# Plot non-normalized confusion matrix(under sample test set)
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix,
					classes=class_names,
					title='Confusion matrix')
plt.show()


print("-----------------------------------------------------------------------")


lr = LogisticRegression(C=best_c, solver='liblinear', penalty='l2')
lr.fit(X_train_undersample, Y_train_undersample.values.ravel())
Y_pred = lr.predict(X_test.values)

# Compare confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset(whole test set): ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix,
					  classes=class_names,
					  title='Confusion matrix')
plt.show()


print("------------------------------------------------------------------------")


best_c = printing_KFold_scores(X_train, Y_train)
print(best_c)

lr = LogisticRegression(C=best_c, solver='liblinear', penalty='l2')
lr.fit(X_train, Y_train.values.ravel())
Y_pred_undersample = lr.predict(X_test.values)


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix,
						classes=class_names,
						title='Confusion matrix')

plt.show()


print("-----------------------------------------------------------------------")


# impact of threshold in logistic regression
print("")
lr = LogisticRegression(C=0.01, solver='liblinear', penalty='l2')
lr.fit(X_train_undersample, Y_train_undersample.values.ravel())
Y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(20, 20))

j = 1
for i in thresholds:
	Y_test_predictions_high_recall = Y_pred_undersample_proba[:, 1] > i

	plt.subplot(3, 3, j)
	j += 1

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(Y_test_undersample, Y_test_predictions_high_recall)
	np.set_printoptions(precision=2)

	print("Recall metric in the testing dataset(undersample): ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

	# Plot non-normalized confusion matrix
	class_names = [0, 1]
	plot_confusion_matrix(cnf_matrix,
						  classes=class_names,
						  title='Threshold >= %s'%i)

plt.show()


print("--------------------------------------------------------------------")


# Oversampling method to process data sets
credit_cards = pd.read_csv("creditcard.csv")

columns = credit_cards.columns

# The labels are in the last column("Class"). Simply remove it to obtain features columns
features_columns = columns.delete(len(columns)-1)

features = credit_cards[features_columns]
labels = credit_cards['Class']


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)

oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_sample(features_train, labels_train)

print(len(os_labels[os_labels == 1]))

print("---------------------------------------------------------------------------")

os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
best_c = printing_KFold_scores(os_features, os_labels)
print(best_c)

print("---------------------------------------------------------------------------")

lr = LogisticRegression(C=best_c, solver='liblinear', penalty='l2')
lr.fit(os_features, os_labels.values.ravel())
Y_pred = lr.predict(features_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_test, Y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset(all dataset): ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix,
					  classes=class_names,
					  title='Confusion matrix')

plt.show()