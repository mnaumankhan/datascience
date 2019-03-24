from sklearn import tree
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn.metrics import accuracy_score

# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 140], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female',
     'male', 'male', 'male', 'female',
     'male', 'female', 'male']

clf = tree.DecisionTreeClassifier()
etc = tree.ExtraTreeClassifier()
rclf = ensemble.RandomForestClassifier()
gpclf = gaussian_process.GaussianProcessClassifier()

clf.fit(X, Y)
etc.fit(X, Y)
rclf.fit(X, Y)
gpclf.fit(X, Y)

_X = [[184, 84, 44], [198, 92, 48], [183, 83, 44], [166, 47, 36], [170, 60, 38], [172, 64, 39], [182, 80, 42],
      [180, 80, 43]]
_Y = ['male', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

prediction = clf.predict(_X)
acc_dectree = accuracy_score(_Y, prediction)

print("prediction for decision tree is:" + str(prediction) + " and accuracy is:" + str(acc_dectree))

etPrediction = etc.predict(_X)
acc_extree = accuracy_score(_Y, etPrediction)

print("prediction for extra tree is:" + str(etPrediction) + " and accuracy is:" + str(acc_extree))

rcPrediction = rclf.predict(_X)
acc_ranfor = accuracy_score(_Y, rcPrediction)

print("prediction for random forest is:" + str(rcPrediction) + " and accuracy is:" + str(acc_ranfor))

gpPrediction = gpclf.predict(_X)
acc_gaupro = accuracy_score(_Y, gpPrediction)

print("prediction for gaussian process is:" + str(gpPrediction) + " and accuracy is:" + str(acc_gaupro))
