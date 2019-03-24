from sklearn import tree
from sklearn import ensemble
from sklearn import gaussian_process

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

prediction = clf.predict([[190, 70, 43]])
etPrediction = etc.predict([[190, 70, 43]])
rcPrediction = rclf.predict([[190, 70, 43]])
gpPrediction = gpclf.predict([[190, 70, 43]])

print(prediction)
print(etPrediction)
print(rcPrediction)
print(gpPrediction)
