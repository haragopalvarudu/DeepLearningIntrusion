import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os

labels = ['dos' 'probe' 'r2l' 'u2r']

dataset = pd.read_csv('Dataset/KDD.csv')
dataset.fillna(0, inplace = True)

labels,count = np.unique(dataset['label'],return_counts=True)
print(labels)
print(count)

cols = ['protocol_type','service','flag','label']
le = LabelEncoder()
dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
dataset[cols[2]] = pd.Series(le.fit_transform(dataset[cols[2]].astype(str)))
dataset[cols[3]] = pd.Series(le.fit_transform(dataset[cols[3]].astype(str)))

labels = np.unique(dataset['label'])
print(labels)

print(dataset.head())
print(dataset.shape)

Y = dataset['label'].tolist()
Y = np.asarray(Y)
dataset.drop(['label'], axis = 1,inplace=True)

corr_features = set()

# create the correlation matrix
corr_matrix = dataset.corr()
for i in range(len(corr_matrix .columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)
            
dataset.drop(labels=corr_features, axis=1, inplace=True)
print(dataset.shape)

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
print(X)
print(Y)


indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
print(Y.shape)

X = X.reshape((X.shape[0],X.shape[1],1,1))
print(X.shape)

if os.path.exists('model/crf_model.json'):
    with open('model/crf_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/crf_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
else:
    classifier = Sequential()
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
    classifier.save_weights('model/crf_weights.h5')            
    model_json = classifier.to_json()
    with open("model/crf_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/crf_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    




