from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43],[160, 60, 38],[154,54,37],[166,65,40], [190,90,47],[175,64,39],[177,70,40],[159,57,38],[171,75,42],[181,85,43]]

Y =['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

#Decision Tree classifier
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X,Y)
prediction = classifier.predict([[190,70,43]])

print(prediction)

#GaussianNB
classifierNaiveB = GaussianNB()
classifierNaiveB.fit(X, Y)
NBprediction = classifierNaiveB.predict([[190,70,43]])
print("According to Naive Bayes classifier:",NBprediction)

#KNN classifier
Kclassifier = KNeighborsClassifier()
Kclassifier.fit(X,Y)
KNNprediction = Kclassifier.predict([[190,70,43]])
print("According to K Nearest Neighbors:", KNNprediction)

#Random Forest Classifier
RFclassifier = RandomForestClassifier()
RFclassifier.fit(X,Y)
RFprediction = RFclassifier.predict([[190, 70, 43]])
print("Accroding to Random Forest Classifier:", RFprediction)

#The Classifiers KNN and Naive Bayes are most accurate ones.

