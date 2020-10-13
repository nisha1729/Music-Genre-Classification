#################
# 2.1 kNN
#################

from data_processor import *
import sklearn.neighbors
from sklearn.metrics import accuracy_score
import sklearn.neighbors


X_train, X_test, y_train, y_test = data_process()
print(np.argwhere(np.isnan(X_train)))
print(X_train[np.argwhere(np.isnan(X_train)),:])
cls = sklearn.neighbors.KNeighborsClassifier()
cls.fit(X_train , y_train)
y_pred = cls.predict(X_train)
print('Accuracy : ' ,accuracy_score(y_test , y_pred)*100 )

