"""
Possible Machine learning methods

https://www.simplilearn.com/10-algorithms-machine-learning-engineers-need-to-know-article

- Linear regression
- Logistic regression
- Decision tree
- SVM algorithm
- Naive Bayes algorithm - not sure about this one! (https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.htm)
- KNN algorithm
- Random forest algorithm
- Gradient Boosting - Combining multiple weak or average predictors to build a strong predictor

</br>

https://towardsdatascience.com/10-machine-learning-methods-that-every-data-scientist-should-know-3cc96e0eeee9

"The great majority of top winners of Kaggle competitions use ensemble methods of some kind. 
 The most popular ensemble algorithms are Random Forest, XGBoost and LightGBM."

- Regression
- Classification
- Clustering
- Dimensionality Reduction
- Ensemble Methods
- Neural Nets and Deep Learning
- Transfer Learning
- Reinforcement Learning
- Natural Language Processing
- Word Embeddings

</br>

https://en.wikipedia.org/wiki/Outline_of_machine_learning#Machine_learning_methods

</br>
"""

import sklearn

## 1/14: Linear Regression

def linearRegression(x_train, y_train, x_test=None, y_test=None):
    model = sklearn.linear_model.LinearRegression().fit(x_train, y_train)
    if type(x_test) != type(None):
        predictions = model.predict(x_test)
        if type(y_test) != type(None):
            score = model.score(x_test, y_test)
            return model, predictions, score
        return model, predictions
    else:
        return model

## 2/14: Logistic Regression

def logRegression(x_train, y_train, x_test=None, y_test=None):
    model = sklearn.linear_model.LogisticRegression(max_iter=10000).fit(x_train, y_train)
    if type(x_test) != type(None):
        predictions = model.predict(x_test)
        if type(y_test) != type(None):
            score = model.score(x_test, y_test)
            return model, predictions, score
        return model, predictions
    else:
        return model

## 3/14: MLP

# from sklearn.neural_network import MLPClassifier as MLP
def MLP_model(x_train, y_train, x_test=None, y_test=None):
    model = sklearn.neural_network.MLPClassifier().fit(x_train, y_train)
    if type(y_test) != type(None):
        predictions = model.predict(x_test)
        if type(x_test) != type(None):
            score = model.score(x_test, y_test)
            return model, predictions, score
        return model, predictions
    else:
        return model

## 4/14: SVM

def SVM(x_train, y_train, x_test=None, y_test=None):
    model = sklearn.svm.SVC(kernel='linear').fit(x_train, y_train)
    if type(x_test) != type(None):
        predictions = model.predict(x_test)
        if type(y_test) != type(None):
            score = model.score(x_test, y_test)
            return model, predictions, score
        return model, predictions
    else:
        return model

## 5/14: Decision Tree

def decisionTree(x_train, y_train, x_test=None, y_test=None):
    model = sklearn.tree.DecisionTreeClassifier().fit(x_train, y_train)
    if type(x_test) != type(None):
        predictions = model.predict(x_test)
        if type(y_test) != type(None):
            score = model.score(x_test, y_test)
            return model, predictions, score
        return model, predictions
    else:
        return model
    
## 6/14: KNN algorithm

from sklearn.neighbors import KNeighborsClassifier
def KNN(x_train, y_train, x_test=None, y_test=None):
    model = KNeighborsClassifier().fit(x_train, y_train)
    if type(x_test) != type(None):
        predictions = model.predict(x_test)
        if type(y_test) != type(None):
            score = model.score(x_test, y_test)
            return model, predictions, score
        return model, predictions
    else:
        return model

## 7/14: Random Forest

from sklearn.ensemble import RandomForestClassifier
def randomForest(x_train, y_train, x_test=None, y_test=None):
    model = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
    if type(x_test) != type(None):
        predictions = model.predict(x_test)
        if type(y_test) != type(None):
            score = model.score(x_test, y_test)
            return model, predictions, score
        return model, predictions
    else:
        return model

## 8/14: Neural Network - FC Layer

def NN_FC():
    pass

## 9/14: Neural Network - Conv

def NN_conv():
    pass

## 10/14: Neural Network - RNN

def RNN():
    pass

## 11/14: Neural Network - LSTM

def LSTM():
    pass

## 12/14: Neural Network - Transformer

def transformer():
    pass

## 13/14: Neural Network - Autoencoder

def AE():
    pass

## 14/14: Neural Network - VAE

def VAE():
    pass
