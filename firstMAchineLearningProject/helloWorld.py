# Link to the site where I got this code and lesson
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/


import sys, scipy, numpy, matplotlib, pandas, sklearn

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# print('Python: {}'.format(sys.version))
# print('Python: {}'.format(scipy.__version__))
# print('Python: {}'.format(numpy.__version__))

# Downgrade matplotlib to 3.5.3 to remove
# pip uninstall matplotlib --> pip install matplotlib==3.5.3 IN PYCHARM --> Settings | Project:{NAME} : Python Interpreter | Manage Packages
# Disble -> File | SETTINGS | TOOLS | PYTHON SCIENTIFIC | SHOW PLOTS IN TOOL WINDOW   <-- This resolved the problem
print('Python: {}'.format(matplotlib._get_version()))


# print('Python: {}'.format(pandas.__version__))
# print('Python: {}'.format(sklearn.__version__))

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape --> 150 instances and 5 attributes
print(dataset.shape)

# head (eyeball your data) --> see the first 20 rows of the data
print(dataset.head(20))


# A summary of each attribute -> count, mean, min, max, and some percentiles
print(dataset.describe())

# Class distribution
print(dataset.groupby('class').size())

# Univariate Plots - whisker plots
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# Histogram
#dataset.hist()
# pyplot.show()

# Multivariate Plots: Look at the interactions between the variables
# scatter plot matrix
#scatter_matrix(dataset)
# pyplot.show()

# Evaluate Some Algorithms
# 1. Separate out a validation dataset
# 2. Set-up the test harness to use 10-fold validation
# 3. Build multiple different models to predict species from flower measurement
# 4. Select the best model

# Create a validation dataset
# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Splut our dataset into 10 parts, train on 9 of them and test on 1 and repeat for all combinations of train-test splits

# Build Models
...
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Running the above code gives us the following
# LR: 0.960897 (0.052113)
# LDA: 0.973974 (0.040110)
# KNN: 0.957191 (0.043263)
# CART: 0.957191 (0.043263)
# NB: 0.948858 (0.056322)
# SVM: 0.983974 (0.032083) SNV has the largest estimated accuracy at about 0.98 OR 98%

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# We see that our accuracy is 0.966 or about 96%
# 0.9666666666666667
# [[11  0  0]
#  [ 0 12  1]
#  [ 0  0  6]]
#                  precision    recall  f1-score   support
#
#     Iris-setosa       1.00      1.00      1.00        11
# Iris-versicolor       1.00      0.92      0.96        13
#  Iris-virginica       0.86      1.00      0.92         6
#
#        accuracy                           0.97        30
#       macro avg       0.95      0.97      0.96        30
#    weighted avg       0.97      0.97      0.97        30

