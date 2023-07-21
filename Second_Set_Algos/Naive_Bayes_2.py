import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
class NaiveBayes:
    def __init__(self, alpha=10, epsilon=1e-9):
        self.alpha = alpha
        self.epsilon = epsilon
        self.classes = None
        self.priors = None
        self.likelihoods = None
       
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
       
        # Calculate priors
        self.priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.priors[i] = np.sum(y == c) / float(len(y))
       
        # Calculate likelihoods
        self.likelihoods = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            total_count = np.sum(X_c, axis=0) + self.alpha
            self.likelihoods[i, :] = total_count / ((np.sum(X_c) + self.alpha * n_features) + self.epsilon)

   
    def predict(self, X):
        # Calculate posterior for each class and predict the class with highest posterior
        posteriors = np.zeros((len(X), len(self.classes)))
        for i, c in enumerate(self.classes):
            likelihood = np.prod(self.likelihoods[i, :] ** X * (1 - self.likelihoods[i, :]) ** (1 - X), axis=1)
            posteriors[:, i] = self.priors[i] * likelihood
        return self.classes[np.argmax(posteriors, axis=1)]
   
    def prior(self, c):
        return self.priors[c]
   
    def posterior(self, x, c):
        likelihood = np.prod(self.likelihoods[c, :] ** x * (1 - self.likelihoods[c, :]) ** (1 - x))
        return self.priors[c] * likelihood
   
    def likelihood(self, x, c):
        return self.likelihoods[c, :]
   
    def precision(self, X, y):
        y_pred = self.predict(X)
        true_positives = np.sum((y_pred == 1) & (y == 1))
        false_positives = np.sum((y_pred == 1) & (y == 0))
        if (true_positives + false_positives) == 0:
            return 0
        else:
            return true_positives / float(true_positives + false_positives + self.epsilon)
   
    def recall(self, X, y):
        y_pred = self.predict(X)
        true_positives = np.sum((y_pred == 1) & (y == 1))
        false_negatives = np.sum((y_pred == 0) & (y == 1))
        if (true_positives + false_negatives) == 0:
            return 0
        else:
            return true_positives / float(true_positives + false_negatives + self.epsilon)
   
    def f1_score(self, X, y):
        prec = self.precision(X, y)
        rec = self.recall(X, y)
        if (prec + rec) == 0:
            return 0
        else:
            return 2 * (prec * rec) / (prec + rec)
   
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / float(len(y))
    
# Load data
df = pd.read_csv('adult.csv')
df = df.replace(" ?", np.nan)

# print(df)
df = df.dropna()
df['O'].replace({' <=50K':0, ' >50K':1},inplace = True)
print(df)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes
print(df)
df3 = df
df.drop('O',axis=1)
df.drop('C',axis=1)
df.drop('J',axis=1)
df = (df-df.mean())/df.std()
df['J'] = df3['J']
df['O'] = df3['O']
X = df.drop('O',axis=1)
y = df['O']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=45)

print(X_test,"sdfgjkl;\n")
print("\n\n\n\n\n",len(X_test),"\n\n\n\n\n")
# print(df)

nb = NaiveBayes()
nb.fit(X_train, y_train)
train_acc = nb.accuracy(X_train, y_train)
test_acc = nb.accuracy(X_test, y_test)
nb_precision = nb.precision(X_test, y_test)
nb_recall = nb.recall(X_test, y_test)
nb_f1_score = nb.f1_score(X_test, y_test)
print('Naive Bayes Classifier')
print('Training accuracy: ', train_acc)
print('Testing accuracy: ', test_acc)
print('Precision: ', nb_precision)
print('Recall: ', nb_recall)
print('F1 Score: ', nb_f1_score)
print('')

knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Use the classifier to predict the labels of the test data
y_pred = knn.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print('KNN Classifier accuracy: ', accuracy)
print('')

logistic_reg = LogisticRegression()

# Fit the model on the training data
logistic_reg.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = logistic_reg.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression accuracy: ", accuracy)
