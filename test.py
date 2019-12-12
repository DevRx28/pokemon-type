import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.adapt import MLkNN

from keras.layers import Dense
from keras.models import Sequential
from keras.metrics import *

##########################################################
# Section 1 - Data Loading
##########################################################

# Getting feature data
finalData = np.array(pd.read_csv('D:/UIP/finaldata.csv', index_col='Name'))
biodata = finalData[:, 21:]

# Getting type data as dataframe for visualisations
pType = pd.read_csv('D:/UIP/primType.csv', index_col=0)
sType = pd.read_csv('D:/UIP/secondType.csv', index_col=0)
bTypes = pd.read_csv('D:/UIP/sparseTypes.csv', index_col=0)

# Getting features as numpy arrays for model inputs
primType = np.array(pType)
secType = np.array(sType)
bothTypes = np.array(bTypes)

# Get splitted data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(finalData, bothTypes, test_size=0.2, random_state=12345)
XtrainPrim, XtestPrim, YtrainPrim, YtestPrim = train_test_split(finalData, primType, test_size=0.2, random_state=12345)
XtrainSec, XtestSec, YtrainSec, YtestSec = train_test_split(finalData, secType, test_size=0.2, random_state=12345)

# Get splitted biodata
XtrainBio, XtestBio, YtrainBio, YtestBio = train_test_split(biodata, bothTypes, test_size=0.2, random_state=12345)
XtrainPrimBio, XtestPrimBio, YtrainPrimBio, YtestPrimBio = train_test_split(biodata, primType, test_size=0.2, random_state=12345)
XtrainSecBio, XtestSecBio, YtrainSecBio, YtestSecBio = train_test_split(biodata, secType, test_size=0.2, random_state=12345)


##########################################################
# Section 2 - Data Visualisation
##########################################################

# Visualising class distribution for Pokemon type
def visualiseTypeDist(typeData, nat):

    # Type Categories
    categories = list(typeData.columns.values)
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(categories, typeData.sum().values)

    # Axis labels
    if nat == 1:
        plt.title("Distribution of Primary Pokemon Types", fontsize=14)
    elif nat == 2:
        plt.title("Distribution of Secondary Pokemon Types", fontsize=14)
    else:
        plt.title("Distribution of Pokemon Types (single and dual)", fontsize=14)

    plt.ylabel('Pokemon of that Type', fontsize=14)
    plt.xlabel('Pokemon Type', fontsize=14)
    rects = ax.patches
    labels = typeData.sum().values

    # Print hist labels
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 1,
                label, ha='center', va='bottom', fontsize=12)
    plt.show()

visualiseTypeDist(pType, 1)
visualiseTypeDist(sType, 2)
visualiseTypeDist(bTypes, 0)

# Function to re-encode output of Neural Network into one-hot encoding
def reEncode(predictions):
    newOut = np.ndarray((len(predictions), len(predictions[0])))
    for i in range(len(predictions)):
        row = predictions[i]
        m = max(row)
        for j in range(len(predictions[0])):
            if row[j] == m:
                newOut[i][j] = 1
            else:
                newOut[i][j] = 0
    return newOut

# Setting epsilon for re-encoding multiple type predictions
epsilon = 0.03

# Function to re-encode output of Neural Network into multiple-hot encoding
def reEncodeMulti(predictions):
    newOut = np.ndarray((len(predictions), len(predictions[0])))
    for i in range(len(predictions)):
        row = predictions[i]
        m = max(row)
        rowAlt = [e for e in row if e != m]
        tx = max(rowAlt)
        rowAltB = [e for e in rowAlt if e != tx]
        tb = max(rowAltB)
        for j in range(len(predictions[0])):
            if row[j] == m:
                newOut[i][j] = 1
            elif row[j] == tx:
                if (tx - tb) >= epsilon:
                    newOut[i][j] = 1
            else:
                newOut[i][j] = 0
    return newOut

# ###############################################################
# # Section 3 - Multi-class classification for Type 1 of Pokemon
# ###############################################################

# Neural Network with Softmax + Categorical Crossentropy
def test_network(Xtrain, Xtest, Ytrain, Ytest):
    model = Sequential()
    feat = len(Xtrain[0])

    # Hidden Layers
    model.add(Dense(64, activation='relu', input_dim=feat))
    # model.add(Dense(64, activation='relu'))

    # Output layer with 18 nodes using Softmax activation (we have 18 Pokemon types)
    model.add(Dense(18, activation='softmax'))

    # Running the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, epochs=40, batch_size=32)

    # Accuracy Metrics and Predictions
    score = model.evaluate(Xtest, Ytest, batch_size=16)
    predictions = model.predict(Xtest)
    return predictions, score

# # Decision Tree - (Deprecated)
# def test_tree(Xtrain, Xtest, Ytrain, Ytest):

#     # Setting tree parameters
#     classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=12345)
#     classifier.fit(Xtrain, Ytrain)

#     # Accuracy Metrics and Predictions
#     print('Accuracy Score for Decision Tree on training set: {:.2f}'.format(classifier.score(Xtrain, Ytrain)))
#     print('Accuracy Score for Decision Tree on test set: {:.2f}'.format(classifier.score(Xtest, Ytest)))
#     predictions = classifier.predict(Xtest)
#     return predictions

# K-Nearest Neighbours for Multi-Class classification
def test_knn(Xtrain, Xtest, Ytrain, Ytest):

    # Setting k = 3
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(Xtrain, Ytrain)

    # Accuracy Metrics and Predictions
    predictions = classifier.predict(Xtest)
    score = classifier.score(Xtest, Ytest)
    return predictions, score

# ######################################################################
# # Section 4 - Multi-class, Multi-label approach to Type classification
# ######################################################################

# Neural Network with Softmax + Binary Crossentropy
def test_network2(Xtrain, Xtest, Ytrain, Ytest):
    model = Sequential()
    feat = len(Xtrain[0])

    # Hidden Layers
    model.add(Dense(64, activation='relu', input_dim=feat))
    # model.add(Dense(64, activation='relu'))

    # Output layer with 18 nodes using Softmax activation (we have 18 Pokemon types)
    model.add(Dense(18, activation='softmax'))

    # Running the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(Xtrain, Ytrain, epochs=40, batch_size=32)

    # Accuracy Metrics and Predictions
    score = model.evaluate(Xtest, Ytest, batch_size=16)
    predictions = model.predict(Xtest)
    return predictions, score

# Multilabel k Nearest Neighbours (MLkNN)
def test_mlknn(Xtrain, Xtest, Ytrain, Ytest):

    # Training the classfier and making predictions
    classifier = MLkNN(k=1)
    classifier.fit(Xtrain, Ytrain)
    predictions = classifier.predict(Xtest)

    # Measuring accuracy
    scores = classifier.score(Xtest, Ytest)
    loss = metrics.hamming_loss(Ytest, predictions)
    return predictions, scores, loss

# Binary Relevance with Logistic Regression
def test_logistic(Xtrain, Xtest, Ytrain, Ytest):

    # Setting parameters for Logistic Regression
    reg = LogisticRegression(C = 1.0, solver='lbfgs', random_state=12345)

    # Initialising the Binary Relevance Pipeline
    classifier = BinaryRelevance(classifier=reg)

    # Training the classfiers and making predictions
    classifier.fit(Xtrain, Ytrain)
    predictions = classifier.predict(Xtest)

    # Measuring accuracy
    scores = classifier.score(Xtest, Ytest)
    loss = metrics.hamming_loss(Ytest, predictions)
    return predictions, scores, loss


###############################################################
# Section 5 - Getting results from models
###############################################################

typeList = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost',
            'Steel', 'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', 'Fairy']

pokemon = pd.read_csv('D:/UIP/testList.csv', header=0)['Name']

#### Section 5.1 - Predicting a Pokemon's primary type. First with bio + move data, then only biodata. ####

# Neural Network
primaryNet_predic, primaryNet_acc = test_network(XtrainPrim, XtestPrim, YtrainPrim, YtestPrim)
pd.DataFrame(reEncode(primaryNet_predic), index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/NetPredictionsPrim.csv')

primaryNet_predicBio, primaryNet_accBio = test_network(XtrainPrimBio, XtestPrimBio, YtrainPrimBio, YtestPrimBio)
pd.DataFrame(reEncode(primaryNet_predicBio), index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/NetPredictionsPrimWithoutMoves.csv')

# # Decision Tree
# primaryForest_predic = test_tree(XtrainPrim, XtestPrim, YtrainPrim, YtestPrim)
# primaryForest_predicBio = test_tree(XtrainPrimBio, XtestPrimBio, YtrainPrimBio, YtestPrimBio)

# K Nearest Neighbours
primaryKNN_predic, primaryKNN_acc = test_knn(XtrainPrim, XtestPrim, YtrainPrim, YtestPrim)
pd.DataFrame(primaryKNN_predic, index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/KNNPredictionsPrim.csv')

primaryKNN_predicBio, primaryKNN_accBio = test_knn(XtrainPrimBio, XtestPrimBio, YtrainPrimBio, YtestPrimBio)
pd.DataFrame(primaryKNN_predicBio, index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/KNNPredictionsPrimWithoutMoves.csv')

#### Section 5.2 - Predicting both types for Pokemon. First with bio + move data, then only biodata. ####

# Neural Network
primaryNet_predic2, primaryNet_acc2 = test_network2(Xtrain[:, :21], Xtest[:, :21], Ytrain, Ytest)
pd.DataFrame(reEncodeMulti(primaryNet_predic2), index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/NetPredictions.csv')

primaryNet_predicBio2, primaryNet_accBio2 = test_network2(XtrainBio, XtestBio, YtrainBio, YtestBio)
pd.DataFrame(reEncodeMulti(primaryNet_predicBio2), index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/NetPredictionsWithoutMoves.csv')

# # MLkNN
mlknn_pred, mlknn_acc, mlknn_hamloss = test_mlknn(Xtrain, Xtest, Ytrain, Ytest)
pd.DataFrame(mlknn_pred.A, index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/MLKNNtPredictions.csv')

mlknn_predBio, mlknn_accBio, mlknn_hamlossBio = test_mlknn(XtrainBio, XtestBio, YtrainBio, YtestBio)
pd.DataFrame(mlknn_predBio.A, index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/MLKNNtPredictionsWithoutMoves.csv')

# Binary Relevance - Logistic Regression
log_pred, log_acc, log_loss = test_logistic(Xtrain, Xtest, Ytrain, Ytest)
pd.DataFrame(log_pred.A, index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/LogPredictions.csv')

log_predBio, log_accBio, log_lossBio = test_logistic(XtrainBio, XtestBio, YtrainBio, YtestBio)
pd.DataFrame(log_predBio.A, index=pokemon, columns=typeList).to_csv('D:/UIP/Pred/LogPredictionsWithoutBio.csv')


###############################################################
# Section 6 - Creating Confusion Matrices
###############################################################

# Type-list for primary type
typeListB = ['Normal', 'Fighting', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel', 'Fire', 
             'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', 'Fairy', 'Flying']
    
# Creating class labels
ylabels = np.unique(YtestPrim.argmax(axis=1))

# Function to return confusion matrix
def getCMatrix(truth, predictions, typeListA, typeListB, primary):
    cm = confusion_matrix(truth.argmax(axis=1), predictions.argmax(axis=1))
    if primary == True:
        cm = np.append(cm, np.zeros((17, 1), dtype=int), axis=1)
        cm = np.append(cm, np.zeros((1, 18), dtype=int), axis=0)
        cm_df = pd.DataFrame(cm, index=typeListB, columns=typeListB)
    else:
        cm_df = multilabel_confusion_matrix(truth, predictions)
    return cm_df

# Function to plot confusion matrix for Primary types
def getVisualsPrim(data, typeList, Prim, Title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='YlGnBu', annot=True, square=True, fmt="d")

    if Prim == True:
        plt.xticks(np.arange(0, 18), typeList, rotation=45)
        plt.yticks(np.arange(0, 18), typeList, rotation=45)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title(Title)
    plt.show()

# Function to plot confusion matrix for both types
def getVisuals(data, typeList):
    for i in range(len(data)):
        cm = pd.DataFrame(data[i], index=[0, 1])
        plTitle = 'Confusion Matrix: {} Type'.format(typeList[i])
        getVisualsPrim(cm, typeList, False, plTitle)

#### 6.1 - Confusion Matrices for Neural Network ######

# Recoding output to binary vector of length 18
neuralOutPrim = reEncode(primaryNet_predic)
neuralOut = reEncodeMulti(primaryNet_predic2)
neuralOut = np.where(neuralOut >= 0.5, 1, neuralOut)

# Repeating process for Neural Network without move data
neuralOutPrimBio = reEncode(primaryNet_predicBio)
neuralOutBio = reEncodeMulti(primaryNet_predicBio2)
neuralOutBio = np.where(neuralOutBio > 0.5, 1, 0)

# Getting confusion matrices
neuralPrimCM = getCMatrix(YtestPrim, neuralOutPrim, typeList, typeListB, True)
neuralCM = getCMatrix(Ytest, neuralOut, typeList, typeListB, False)

# Visualising Heatmaps of Confusion Matrices
getVisualsPrim(neuralPrimCM, typeListB, True, 'Confusion Matrix - Neural Network')
getVisuals(neuralCM, typeList)

#### 6.2 - Confusion Matrices for KNN and MLkNN ######

# Getting confusion matrices
knnCM = getCMatrix(YtestPrim, primaryKNN_predic, typeList, typeListB, True)
mlknnCM = getCMatrix(Ytest, mlknn_pred.A, typeList, typeListB, False)

# Visualising Heatmaps of Confusion Matrices
getVisualsPrim(knnCM, typeListB, True, 'Confusion Matrix - KNN')
getVisuals(mlknnCM, typeList)


###############################################################
# Section 7 - Getting accuracy measures
###############################################################

# Function to print relevant measures
def getMeasures(ytrue, ypred, name, type):

    print("Printing accuracy measures for {} below:".format(name))
    print('Precision Score = {}'.format(metrics.precision_score(ytrue, ypred, average='macro')))
    print('Recall Score = {}'.format(metrics.recall_score(ytrue, ypred, average='macro')))
    print('F1 Macro Score = {}'.format(metrics.f1_score(ytrue, ypred, average='macro')))

    # if type == 1:
    print('Accuracy Score = {}'.format(metrics.accuracy_score(ytrue, ypred)))
    
    if type == 1:
        C = top_k_categorical_accuracy(ytrue, ypred, k=2)
    else:
        C = top_k_categorical_accuracy(ytrue, ypred, k=3)
    kscore = len([i for i in C if i == 1]) / len(C)
    print('Top-K Categorical Accuracy = {}'.format(kscore))
    print('Weighted F1 Score = {}'.format(metrics.f1_score(ytrue, ypred, average='weighted')))

# Printing the measures
getMeasures(YtestPrim, neuralOutPrim, 'NeuralNet', 1)
getMeasures(YtestPrim, neuralOutPrimBio, 'NeuralNet No Moves', 1)

getMeasures(YtestPrim, primaryKNN_predic, 'KNN', 1)
getMeasures(YtestPrim, primaryKNN_predicBio, 'KNN No Moves', 1)

getMeasures(Ytest, neuralOut, 'NeuralNet BothTypes', 1)
getMeasures(Ytest, neuralOutBio, 'NeuralNet BothTypes Bio', 2)

getMeasures(Ytest, mlknn_pred.A, 'MLkNN BothTypes', 1)
getMeasures(Ytest, mlknn_predBio.A, 'MLkNN BothTypes Bio', 1)

getMeasures(Ytest, log_pred.A, 'Logis BothTypes', 1)
getMeasures(Ytest, log_predBio.A, 'Logis BothTypes Bio', 1)
