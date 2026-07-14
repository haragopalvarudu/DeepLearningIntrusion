from tkinter import messagebox
from tkinter import *
import tkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import Sequential, model_from_json

import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import webbrowser

main = tkinter.Tk()
main.title("A Deep Learning Approach for Effective Intrusion Detection in Wireless Networks using CNN")
main.geometry("1300x1200")

filename = ""
X = None
Y = None
dataset = None
output = ""

labels = ['dos', 'probe', 'r2l', 'u2r']
accuracy = []


def uploadDataset():
    global filename
    global dataset

    filename = filedialog.askopenfilename(initialdir="Dataset")

    if filename == "":
        return

    pathlabel.config(text=filename)

    text.delete('1.0', END)

    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)

    text.insert(END, filename + " loaded\n\n")
    text.insert(END, str(dataset.head()))

    if 'label' in dataset.columns:
        label = dataset.groupby('label').size()
        label.plot(kind="bar")
        plt.show()
    else:
        text.insert(END, "\n\n'label' column not found in dataset")


def preprocessDataset():
    global dataset

    if dataset is None:
        messagebox.showerror("Error", "Please upload dataset first")
        return

    text.delete('1.0', END)

    cols = ['protocol_type', 'service', 'flag', 'label']

    le = LabelEncoder()

    for col in cols:
        if col in dataset.columns:
            dataset[col] = pd.Series(
                le.fit_transform(dataset[col].astype(str))
            )

    text.insert(END, "Dataset preprocessing completed\n\n")
    text.insert(END, str(dataset.head()))


def createCNNModel(input_shape, output_classes):

    classifier = Sequential()

    classifier.add(
        Conv2D(
            32,
            (1, 1),
            input_shape=input_shape,
            activation='relu'
        )
    )

    classifier.add(MaxPooling2D(pool_size=(1, 1)))

    classifier.add(
        Conv2D(
            32,
            (1, 1),
            activation='relu'
        )
    )

    classifier.add(MaxPooling2D(pool_size=(1, 1)))

    classifier.add(Flatten())

    classifier.add(Dense(256, activation='relu'))

    classifier.add(Dense(output_classes, activation='softmax'))

    classifier.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return classifier


def CNNFullFeatures():

    global output
    global accuracy
    global dataset
    global X, Y

    if dataset is None:
        messagebox.showerror("Error", "Please upload dataset first")
        return

    if 'label' not in dataset.columns:
        messagebox.showerror("Error", "'label' column not found")
        return

    text.delete('1.0', END)

    data_copy = dataset.copy()

    Y = data_copy['label'].values

    X = data_copy.drop(['label'], axis=1).values

    text.insert(
        END,
        "Total records found in dataset : " + str(X.shape[0]) + "\n"
    )

    text.insert(
        END,
        "Total features found in dataset : " + str(X.shape[1]) + "\n"
    )

    text.insert(
        END,
        "Types of attacks/intruders found in dataset : "
        + str(labels) + "\n\n"
    )

    indices = np.arange(X.shape[0])

    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    Y1 = to_categorical(Y)

    X = X.reshape((X.shape[0], X.shape[1], 1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y1,
        test_size=0.2,
        random_state=42
    )

    model_json_path = 'model/full_model.json'
    model_weight_path = 'model/full_weights.weights.h5'

    if os.path.exists(model_json_path):

        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()

        classifier = model_from_json(loaded_model_json)

        classifier.load_weights(model_weight_path)

        classifier.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    else:

        classifier = createCNNModel(
            (X.shape[1], X.shape[2], X.shape[3]),
            Y1.shape[1]
        )

        hist = classifier.fit(
            X_train,
            y_train,
            batch_size=16,
            epochs=10,
            shuffle=True,
            verbose=2
        )

        classifier.save_weights(model_weight_path)

        model_json = classifier.to_json()

        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)

        with open('model/full_history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    predict = classifier.predict(X_test)

    predict = np.argmax(predict, axis=1)

    y_test_labels = np.argmax(y_test, axis=1)

    p = precision_score(
        y_test_labels,
        predict,
        average='macro'
    ) * 100

    r = recall_score(
        y_test_labels,
        predict,
        average='macro'
    ) * 100

    f = f1_score(
        y_test_labels,
        predict,
        average='macro'
    ) * 100

    a = accuracy_score(
        y_test_labels,
        predict
    ) * 100

    text.insert(
        END,
        'CNN Full Features Accuracy  : ' + str(a) + "\n"
    )

    text.insert(
        END,
        'CNN Full Features Precision : ' + str(p) + "\n"
    )

    text.insert(
        END,
        'CNN Full Features Recall    : ' + str(r) + "\n"
    )

    text.insert(
        END,
        'CNN Full Features FMeasure  : ' + str(f) + "\n\n"
    )

    accuracy.clear()
    accuracy.append(a)

    unique_test, counts_test = np.unique(
        y_test_labels,
        return_counts=True
    )

    unique_pred, counts_pred = np.unique(
        predict,
        return_counts=True
    )

    output = '<html><body>'
    output += '<h2>CNN Comparison Table</h2>'
    output += '<table border=1 cellpadding=10>'
    output += '<tr>'
    output += '<th>Algorithm Name</th>'

    for lab in labels:
        output += '<th>' + lab + '</th>'

    output += '</tr>'

    output += '<tr>'
    output += '<td>CNN with Full Features</td>'

    for i in range(min(len(counts_pred), len(counts_test))):

        acc = counts_pred[i] / counts_test[i]

        text.insert(
            END,
            labels[i] + " : Accuracy = " + str(acc) + "\n"
        )

        output += '<td>' + str(round(acc, 4)) + '</td>'

    output += '</tr>'


def CNNCRF():

    global output
    global dataset
    global X, Y

    if filename == "":
        messagebox.showerror("Error", "Please upload dataset first")
        return

    text.delete('1.0', END)

    dataset = pd.read_csv(filename)

    dataset.fillna(0, inplace=True)

    cols = ['protocol_type', 'service', 'flag', 'label']

    le = LabelEncoder()

    for col in cols:
        if col in dataset.columns:
            dataset[col] = pd.Series(
                le.fit_transform(dataset[col].astype(str))
            )

    Y = dataset['label'].values

    dataset = dataset.drop(['label'], axis=1)

    corr_features = set()

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if abs(corr_matrix.iloc[i, j]) > 0.8:

                colname = corr_matrix.columns[i]

                corr_features.add(colname)

    dataset.drop(
        labels=corr_features,
        axis=1,
        inplace=True
    )

    data = dataset.values

    X = data

    indices = np.arange(X.shape[0])

    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    Y = to_categorical(Y)

    X = X.reshape((X.shape[0], X.shape[1], 1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    text.insert(
        END,
        "Total records found in dataset : "
        + str(data.shape[0]) + "\n"
    )

    text.insert(
        END,
        "Total features found in dataset after applying CRF-LCFS : "
        + str(data.shape[1]) + "\n"
    )

    text.insert(
        END,
        "Types of attacks/intruders found in dataset : "
        + str(labels) + "\n\n"
    )

    model_json_path = 'model/crf_model.json'
    model_weight_path = 'model/crf_weights.weights.h5'

    if os.path.exists(model_json_path):

        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()

        classifier = model_from_json(loaded_model_json)

        classifier.load_weights(model_weight_path)

        classifier.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    else:

        classifier = createCNNModel(
            (X.shape[1], X.shape[2], X.shape[3]),
            Y.shape[1]
        )

        hist = classifier.fit(
            X_train,
            y_train,
            batch_size=16,
            epochs=10,
            shuffle=True,
            verbose=2
        )

        classifier.save_weights(model_weight_path)

        model_json = classifier.to_json()

        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)

        with open('model/crf_history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    predict = classifier.predict(X_test)

    predict = np.argmax(predict, axis=1)

    y_test_labels = np.argmax(y_test, axis=1)

    p = precision_score(
        y_test_labels,
        predict,
        average='macro'
    ) * 100

    r = recall_score(
        y_test_labels,
        predict,
        average='macro'
    ) * 100

    f = f1_score(
        y_test_labels,
        predict,
        average='macro'
    ) * 100

    a = accuracy_score(
        y_test_labels,
        predict
    ) * 100

    text.insert(
        END,
        'CNN with CRF-LCFS Features Accuracy  : '
        + str(a) + "\n"
    )

    text.insert(
        END,
        'CNN with CRF-LCFS Features Precision : '
        + str(p) + "\n"
    )

    text.insert(
        END,
        'CNN with CRF-LCFS Features Recall    : '
        + str(r) + "\n"
    )

    text.insert(
        END,
        'CNN with CRF-LCFS Features FMeasure  : '
        + str(f) + "\n\n"
    )

    accuracy.append(a)

    unique_test, counts_test = np.unique(
        y_test_labels,
        return_counts=True
    )

    unique_pred, counts_pred = np.unique(
        predict,
        return_counts=True
    )

    output += '<tr>'
    output += '<td>CNN with CRF-LCFS</td>'

    for i in range(min(len(counts_pred), len(counts_test))):

        acc = counts_pred[i] / counts_test[i]

        text.insert(
            END,
            labels[i] + " : Accuracy = " + str(acc) + "\n"
        )

        output += '<td>' + str(round(acc, 4)) + '</td>'

    output += '</tr>'
    output += '</table></body></html>'


def graph():

    if len(accuracy) == 0:
        messagebox.showerror(
            "Error",
            "Please train models first"
        )
        return

    height = accuracy

    bars = []

    if len(accuracy) >= 1:
        bars.append('CNN Full Features Accuracy')

    if len(accuracy) >= 2:
        bars.append('CNN CRF-LCFS Features Accuracy')

    y_pos = np.arange(len(height))

    plt.figure(figsize=(8, 5))

    plt.bar(y_pos, height)

    plt.xticks(y_pos, bars)

    plt.title(
        'CNN Full Features VS CRF-LCFS Accuracy Comparison Graph'
    )

    plt.ylabel('Accuracy')

    plt.show()


def comparisonTable():

    global output

    if output == "":
        messagebox.showerror(
            "Error",
            "Please train the models first"
        )
        return

    f = open("output.html", "w")

    f.write(output)

    f.close()

    webbrowser.open("output.html", new=1)


font = ('times', 16, 'bold')

title = Label(
    main,
    text='A Deep Learning Approach for Effective Intrusion Detection in Wireless Networks using CNN',
    anchor=W,
    justify=CENTER
)

title.config(bg='yellow4', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')

upload = Button(
    main,
    text="Upload KDD-CUP Dataset",
    command=uploadDataset
)

upload.place(x=10, y=500)

upload.config(font=font1)

pathlabel = Label(main)

pathlabel.config(bg='yellow4', fg='white')

pathlabel.config(font=font1)

pathlabel.place(x=400, y=500)

preprocessButton = Button(
    main,
    text="Preprocess Dataset",
    command=preprocessDataset
)

preprocessButton.place(x=10, y=550)

preprocessButton.config(font=font1)

fullcnnButton = Button(
    main,
    text="Train CNN on Full Features",
    command=CNNFullFeatures
)

fullcnnButton.place(x=350, y=550)

fullcnnButton.config(font=font1)

cnncrfButton = Button(
    main,
    text="Train CNN with CRF-LCFS",
    command=CNNCRF
)

cnncrfButton.place(x=650, y=550)

cnncrfButton.config(font=font1)

graphButton = Button(
    main,
    text="Accuracy Comparison Graph",
    command=graph
)

graphButton.place(x=10, y=600)

graphButton.config(font=font1)

tableButton = Button(
    main,
    text="Comparison Table",
    command=comparisonTable
)

tableButton.place(x=350, y=600)

tableButton.config(font=font1)

font1 = ('times', 12, 'bold')

text = Text(main, height=20, width=120)

scroll = Scrollbar(text)

text.configure(yscrollcommand=scroll.set)

text.place(x=10, y=100)

text.config(font=font1)

main.config(bg='magenta3')

main.mainloop()