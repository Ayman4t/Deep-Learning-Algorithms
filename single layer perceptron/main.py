import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


data = pd.read_excel('Dry_Bean_Dataset.xlsx')
cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
data.fillna(value=data[data["Class"] == "BOMBAY"]['MinorAxisLength'].mean(), inplace=True)

scaler = StandardScaler()

# Fit the scaler on the selected columns
scaler.fit(data[cols])

# Transform the selected columns using the scaler
data[cols] = scaler.transform(data[cols])




def feature_selection(X_train,y_train):
    k = 2
    selector = SelectKBest(f_classif, k=k)
    selector.fit_transform(X_train, y_train)
    cols = selector.get_support(indices=True)
    selected_columns = X_train.iloc[:, cols].columns.tolist()
    print('selected columns:', selected_columns)



def test(x_train,y_train,f1,f2, weights,algo):
    tClass1 = 0
    fClass1 = 0
    tClass2 = 0
    fClass2 = 0
    Y_predict = np.zeros(len(x_train))
    for i in range(0, len(x_train)):

        net = x_train[f1][i] * weights[1] + x_train[f2][i] * weights[2]

        if (weights[0] != 0):
            net = net + weights[0]
        if(algo=="perceptron"):
            if (net < 0):
                y = -1
            else:
                y = 1
        else:
            y = int(net)


        Y_predict[i] = y
        d = y_train[i]

        ypredict = y
        yTrue = d
        if yTrue == 1:
            if ypredict == yTrue:
                tClass1 = tClass1 + 1
            else:
                fClass1 = fClass1 + 1

        else:
            if ypredict == yTrue:
                tClass2 = tClass2 + 1
            else:
                fClass2 = fClass2 + 1

    TotalTrue = tClass1 + tClass2
    Total = TotalTrue + fClass1 + fClass2
    accuracy = (TotalTrue) / (Total)
    print("Confusion Matrix")
    print("True Class1:", tClass1)
    print("False Class1:", fClass1)
    print("True Class2:", tClass2)
    print("False Class2:", fClass2)

    print("Accuracy: ", accuracy * 100)
    ax = plt.axes()
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    data = x_train.merge(y_train, left_index=True, right_index=True)
    dfClass1 = data[data['Class'] == 1]
    if(algo=="perceptron"):
        dfClass2 = data[data['Class'] == -1]
    else:
        dfClass2 = data[data['Class'] == 0]

    plt.scatter(dfClass1[f1], dfClass1[f2], c="g")
    plt.scatter(dfClass2[f1], dfClass2[f2], c="b")

    bias = weights[0]
    xStart = data.iloc[:, 1].min()
    xEnd = data.iloc[:, 1].max()
    linePointStart = -(weights[1] * data.iloc[:, 1].min() + bias) / weights[2]
    linePointEnd = -(weights[1] * data.iloc[:, 1].max() + bias) / weights[2]

    plt.plot([xStart, xEnd], [linePointStart, linePointEnd], c='r')
    ax.plot()
    plt.show()


def fit_perceptron(c1, c2, f1, f2, lr, epochs, useBias):
    x_train = pd.concat([c1, c2], ignore_index=True)[['Area','Perimeter','MajorAxisLength','MinorAxisLength','roundnes']]
    y_train = pd.concat([c1['Class'], c2['Class']], ignore_index=True)
    feature_selection(x_train, y_train)
    bias = 0.0
    if useBias:
        bias = 1.0
    # Create Weights [bias,w1,w2] w random 0->1
    weights = np.random.uniform(low=0, high=1, size=(3,))
    weights[0] = bias*weights[0]

    r = c1.iloc[0:1, -1]
    t = c2.iloc[0:1, -1]
    r = str(r.iloc[0])
    t = str(t.iloc[0])
    c1['Class'].replace(r, 1, inplace=True)
    c2['Class'].replace(t, -1, inplace=True)
    c1_train, c1_test = train_test_split(c1, test_size=0.40, random_state=1, shuffle=True)
    c2_train, c2_test = train_test_split(c2, test_size=0.40, random_state=1, shuffle=True)

    x_train = pd.concat([c1_train, c2_train], ignore_index=True)[[f1,f2]]
    y_train = pd.concat([c1_train['Class'], c2_train['Class']], ignore_index=True)
    x_test = pd.concat([c1_test, c2_test], ignore_index=True)[[f1, f2]]
    y_test = pd.concat([c1_test['Class'], c2_test['Class']], ignore_index=True)

    for j in range(0, epochs):
        for i in range(0, len(x_train)):

            net = x_train[f1][i] * weights[1] + x_train[f2][i] * weights[2]

            if (weights[0] != 0):
                net = net + weights[0]

            if (net < 0):
                y = -1
            else:
                y = 1

            d = y_train[i]
            error = d - y
            if (weights[0] != 0):
                weights[0] = weights[0] + (lr * error)
            weights[1] = weights[1] + (lr * error * x_train[f1][i])
            weights[2] = weights[2] + (lr * error * x_train[f2][i])
    print(weights)

    return weights,x_train,y_train,x_test,y_test


def fit_adaline(c1, c2, f1, f2, lr, epochs, mse, useBias):
    x_train = pd.concat([c1, c2], ignore_index=True)[['Area','Perimeter','MajorAxisLength','MinorAxisLength','roundnes']]
    y_train = pd.concat([c1['Class'], c2['Class']], ignore_index=True)
    feature_selection(x_train, y_train)
    bias = 0.0
    if useBias:
        bias = 1.0
    # Create Weights [bias,w1,w2] w random 0->1
    weights = np.random.uniform(low=0, high=1, size=(3,))
    weights[0] = bias*weights[0]
    r = c1.iloc[0:1, -1]
    t = c2.iloc[0:1, -1]
    r = str(r.iloc[0])
    t = str(t.iloc[0])
    c1['Class'].replace(r, 1, inplace=True)
    c2['Class'].replace(t, 0, inplace=True)
    c1_train, c1_test = train_test_split(c1, test_size=0.40, random_state=1, shuffle=True)
    c2_train, c2_test = train_test_split(c2, test_size=0.40, random_state=1, shuffle=True)
    x_train = pd.concat([c1_train, c2_train], ignore_index=True)[[f1, f2]]
    y_train = pd.concat([c1_train['Class'], c2_train['Class']], ignore_index=True)
    x_test = pd.concat([c1_test, c2_test], ignore_index=True)[[f1, f2]]
    y_test = pd.concat([c1_test['Class'], c2_test['Class']], ignore_index=True)
    for j in range(0, epochs):
        for i in range(0, len(x_train)):

            net = x_train[f1][i] * weights[1] + x_train[f2][i] * weights[2]

            if (weights[0] != 0):
                net = net + weights[0]

            y = net

            d = y_train[i]
            error = d - y
            if (weights[0] != 0):
                weights[0] = weights[0] + (lr * error)
            weights[1] = weights[1] + (lr * error * x_train[f1][i])
            weights[2] = weights[2] + (lr * error * x_train[f2][i])
            # MSE
            totaleError = 0
        for i in range(0, len(x_train)):
            net = x_train[f1][i] * weights[1] + x_train[f2][i] * weights[2]

            if (weights[0] != 0):
                net = net + weights[0]
            y = net

            d = y_train[i]
            error = d - y
            totaleError += (error ** 2)

        finalerror = (1 / len(x_train)) * (totaleError / 2)
        if (finalerror < mse):
                break
    return weights,x_train,y_train,x_test,y_test


def predict(c1, c2, f1, f2, lr, epochs, mse, bias, algo):
    if(algo=="perceptron"):
        weights,x_train,y_train,x_test,y_test=fit_perceptron(c1, c2, f1, f2, lr, epochs, bias)
    else:
        weights,x_train,y_train,x_test,y_test=fit_adaline(c1, c2, f1, f2, lr, epochs, mse, bias)

    test(x_train,y_train,f1,f2,weights,algo)
    test(x_test,y_test,f1,f2,weights,algo)





