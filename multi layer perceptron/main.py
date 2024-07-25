import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_excel('Dry_Bean_Dataset.xlsx')
cols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength']
data.fillna(value=data[data["Class"] == "BOMBAY"]['MinorAxisLength'].mean(), inplace=True)

scaler = StandardScaler()

# Fit the scaler on the selected columns
scaler.fit(data[cols])
# Transform the selected columns using the scaler
data[cols] = scaler.transform(data[cols])

c1=data[data['Class']=="BOMBAY"].copy()
c2=data[data['Class']=="CALI"].copy()
c3=data[data['Class']== "SIRA"].copy()

r = c1.iloc[0:1, -1]
t = c2.iloc[0:1, -1]
s = c3.iloc[0:1, -1]
r = str(r.iloc[0])
t = str(t.iloc[0])
s = str(s.iloc[0])
c1['Class'].replace(r, 1, inplace=True)
c2['Class'].replace(t, 0, inplace=True)
c3['Class'].replace(s, -1, inplace=True)

c1_train, c1_test = train_test_split(c1, test_size=0.40, random_state=1, shuffle=True)
c2_train, c2_test = train_test_split(c2, test_size=0.40, random_state=1, shuffle=True)
c3_train, c3_test = train_test_split(c3, test_size=0.40, random_state=1, shuffle=True)

trainData = pd.concat([c1_train, c2_train,c3_train], ignore_index=True)
testData = pd.concat([c1_test, c2_test,c3_test], ignore_index=True)


def prepareWeights(useBias, hiddenLayersNodes, numInput, numOutput):
    Bias = []
    weights = []
    weights.append(np.random.uniform(0, 1, size=(numInput, hiddenLayersNodes[0])))

    for layer in range(len(hiddenLayersNodes) - 1):
        weights.append(np.random.uniform(0, 1, size=(hiddenLayersNodes[layer], hiddenLayersNodes[layer + 1])))
        if useBias:
            Bias.append(np.random.uniform(0, 1, size=(hiddenLayersNodes[layer], 1)))

        else:
            Bias.append(np.zeros((hiddenLayersNodes[layer], 1)))
    weights.append(np.random.uniform(0, 1, size=(hiddenLayersNodes[-1],numOutput)))

    if useBias:
        Bias.append(np.random.uniform(0, 1, size=(numOutput, 1)))
    else:
        Bias.append(np.zeros((numOutput, 1)))
    return weights,Bias


def multi_layer_perceptron(traindata,testdata, lr, number_of_hidden_layers,number_of_hidden_nodes, usebias, epochs, activation_function):

    x_train=traindata.iloc[:, :-1]
    x_test=testdata.iloc[:, :-1]
    y_train=traindata['Class']
    y_test=testdata['Class']
    weights, bias = prepareWeights(usebias, number_of_hidden_nodes, 5, 3)
    fnet = [None] * (number_of_hidden_layers + 1)
    segma_net = [None] * (number_of_hidden_layers + 1)
    for j in range(0, epochs):
        for i in range(0, len(x_train)):
            sample = x_train.iloc[i].to_numpy().reshape(-1, 1)
            sampleY = y_train.iloc[i]
            sampleY = setValue(sampleY)
            # forward
            #first layer
            if activation_function == "sigmoid":
                fnet[0] = sigmoid(np.dot(weights[0].transpose(), sample).reshape(-1, 1) + bias[0])
            else:
                fnet[0] = tanh(np.dot(weights[0].transpose(), sample).reshape(-1, 1) + bias[0])
            #rest layer
            for layer in range(1, number_of_hidden_layers + 1):
                if activation_function == "sigmoid":
                    fnet[layer] = sigmoid(np.dot(weights[layer].transpose(), fnet[layer - 1]).reshape(-1, 1) + bias[layer])
                else:
                    fnet[layer] = tanh(np.dot(weights[layer].transpose(), fnet[layer - 1]).reshape(-1, 1) + bias[layer])

            # backpropagation
            for layer in reversed(range(number_of_hidden_layers+1)):
                if layer == number_of_hidden_layers:
                    if(activation_function=="sigmoid"):
                        segma_net[layer] = ((sampleY - fnet[layer]) * der_sigmoid(fnet[layer])).reshape(-1, 1)
                    else:
                        segma_net[layer] = ((sampleY - fnet[layer]) * der_tanh(fnet[layer])).reshape(-1, 1)
                else:
                    if (activation_function == "sigmoid"):
                        segma_net[layer] = (np.dot(weights[layer + 1], segma_net[layer + 1]) * der_sigmoid(fnet[layer])).reshape(-1, 1)
                    else:
                        segma_net[layer] = (np.dot(weights[layer + 1], segma_net[layer + 1]) * der_tanh(fnet[layer])).reshape(-1, 1)

            # update weights
            for layer in range(number_of_hidden_layers+1):
                if layer == 0:
                    matrix = np.zeros((sample.shape[0], segma_net[layer].shape[0]))
                    for i in range(sample.shape[0]):
                        for j in range(segma_net[layer].shape[0]):
                            matrix[i, j] = sample[i] * segma_net[layer][j]
                    weights[layer] = weights[layer] + lr * matrix
                    bias[layer] = bias[layer] + (lr * segma_net[layer] * usebias)
                else:
                    # bias[layer] = bias[layer] + lr * segma_net[layer]
                    bias[layer] = bias[layer] + (lr * segma_net[layer] * usebias)
                    matrix = np.zeros((fnet[layer - 1].shape[0], segma_net[layer].shape[0]))
                    for i in range(fnet[layer - 1].shape[0]):
                        for j in range(segma_net[layer].shape[0]):
                            matrix[i, j] = fnet[layer - 1][i] * segma_net[layer][j]
                    weights[layer] = weights[layer] + lr * matrix

    return weights,bias,x_train, y_train, x_test, y_test


def der_tanh(tanh):
    return 1-tanh*tanh


def der_sigmoid(sigmoid):
    return sigmoid * (1 - sigmoid)


def predict_multi_layer_perceptron(Weights, Bias, Xtrain, number_of_hidden_layer, activation_function):
    fnet = [None] * (number_of_hidden_layer + 1)
    sample = Xtrain
    # forward
    if activation_function == "sigmoid":
        fnet[0] = sigmoid(np.dot(Weights[0].transpose(), sample).reshape(-1, 1) + Bias[0])
    else:
        fnet[0] = tanh(np.dot(Weights[0].transpose(), sample).reshape(-1, 1) + Bias[0])

    for layer in range(1, number_of_hidden_layer + 1):
        # activation
        if activation_function == 1:
            fnet[layer] = sigmoid(
                np.dot(Weights[layer].transpose(), fnet[layer - 1]).reshape(-1, 1) + Bias[layer])
        else:
            fnet[layer] = tanh(np.dot(Weights[layer].transpose(), fnet[layer - 1]).reshape(-1, 1) + Bias[layer])

    return fnet[number_of_hidden_layer]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def setValue(inputY):
    if inputY == 1:
        return np.array([1, 0, 0]).reshape(-1, 1)
    elif inputY == 0:
        return np.array([0, 1, 0]).reshape(-1, 1)
    elif inputY == -1:
        return np.array([0, 0, 1]).reshape(-1, 1)


def test(x_test, y_test, num_of_hidden_layer, weights, bias, act_func):
    num_classes = len(np.unique(y_test))
    class_counts = np.zeros((num_classes, 2))  # True positive and false positive counts for each class

    for i in range(len(x_test)):
        Y_predict = predict_multi_layer_perceptron(weights, bias, x_test.iloc[i], num_of_hidden_layer, act_func)
        Y_predict = signum(Y_predict)

        true_class = y_test.iloc[i]
        predicted_class = Y_predict

        class_counts[true_class, 1] += 1  # Increment total count for the true class

        if predicted_class == true_class:
            class_counts[true_class, 0] += 1  # Increment true positive count for the true class

    accuracy = np.sum(class_counts[:, 0]) / np.sum(class_counts[:, 1])
    print("Confusion Matrix")
    for i in range(num_classes):
        print("True Class {}: {}".format(i, class_counts[i, 0]))
        print("False Class {}: {}".format(i, class_counts[i, 1] - class_counts[i, 0]))

    print("Accuracy: {:.2f}%".format(accuracy * 100))


def signum(x):
    maxClass = max(x)
    if maxClass == x[0]:
        return 1
    elif maxClass == x[1]:
        return 0
    elif maxClass == x[2]:
        return -1


def predict(number_hidden_layers, number_nuerons, epochs,lr, bias, act_func):
    number_nuerons.append(3)
    if bias==True:
        bias=1
    else:
        bias=0
    weights,Bias, x_train, y_train, x_test, y_test = multi_layer_perceptron(trainData,testData,lr,number_hidden_layers,number_nuerons,bias,epochs,act_func)
    print("train : ")
    test(x_train, y_train, number_hidden_layers,weights,Bias,act_func)
    print("test : ")
    test(x_test, y_test, number_hidden_layers,weights,Bias,act_func)