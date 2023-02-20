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
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
