#####################
# 2.3 Research - SVM
#####################

from data_processor import *
from sklearn import svm
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = data_process()
model = svm.SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : ', accuracy_score(y_test, y_pred) * 100)