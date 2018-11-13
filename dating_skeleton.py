import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import utils
from mlxtend.plotting import plot_decision_regions

# Create your df here:

df = pd.read_csv("profiles.csv")

df = df.replace(np.nan, '', regex=True)
df.loc[df['religion'].str.contains('agnosticism', case=False), 'religion'] = 'agnosticism'
df.loc[df['religion'].str.contains('other', case=False), 'religion'] = 'other'
df.loc[df['religion'].str.contains('atheism', case=False), 'religion'] = 'atheism'
df.loc[df['religion'].str.contains('christianity', case=False), 'religion'] = 'christianity'
df.loc[df['religion'].str.contains('judaism', case=False), 'religion'] = 'judaism'
df.loc[df['religion'].str.contains('catholicism', case=False), 'religion'] = 'catholicism'
df.loc[df['religion'].str.contains('islam', case=False), 'religion'] = 'islam'
df.loc[df['religion'].str.contains('buddhism', case=False), 'religion'] = 'buddhism'


religion_map = {'agnosticism':0, 'other':1, 'atheism':2, 'christianity':3, 'judaism':4, 'catholicism':5, 'islam':6, 'buddhism':7}
df['religion_code'] = df['religion'].map(religion_map)

drink_map = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df['drink_code'] = df['drinks'].map(drink_map)

sex_map = {'m': 0, 'f': 1}
df['sex_code'] = df['sex'].map(sex_map)

# create df from all data with just the data i need
feature_data = df[['religion_code', 'drink_code', 'sex_code']]
feature_data.dropna(inplace=True)
#convert to np array
x = feature_data.values
#normalize
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

x = feature_data[['religion_code', 'drink_code']]
y = feature_data['sex_code']

h=.02 # step size in the mesh
#create the training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

classifier = SVC(kernel='poly', gamma=5)
classifier.fit(x_train.values, y_train.values)
# create a mesh to plot in
x_min, x_max = x_train.values[:,0].min()-1, x_train.values[:,0].max()+1
y_min, y_max = x_train.values[:,1].min()-1, x_train.values[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.axis('tight')
plt.scatter(x_train.values[:,0], x_train.values[:,1], c=y_train.values, cmap=plt.cm.coolwarm)
plt.show()

print(classifier.score(x_test.values, y_test.values))

'''
#encode float so it can be used by classifier
lab_enc = preprocessing.LabelEncoder()
encoded_y_train = lab_enc.fit_transform(y_train.values)
encoded_y_test = lab_enc.fit_transform(y_test.values)

x_list = []
y_list = []

classifier = KNeighborsClassifier(n_neighbors = 30, weights='distance')
classifier.fit(x_train, encoded_y_train)
valid_acc_y = classifier.score(x_test, encoded_y_test)


plt.plot(x_list, y_list)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title('Sex Qualifier Accuracy')
plt.show()
guess = classifier.predict(x_test)
print(guess)
'''





