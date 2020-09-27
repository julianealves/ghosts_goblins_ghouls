import csv
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

x = []
y = []
ids = []
x_test = []
y_test = []
monsters = []
temp1 = []
temp2 = []
tests = []
cont = 0

#Read train file
with open('train.csv','r') as text:
    fileReader = csv.reader(text, delimiter=',')
    for row in fileReader:
        monsters.append(row)

#Read test file
with open('test.csv','r') as text:
    fileReadert = csv.reader(text, delimiter=',')
    for row in fileReadert:
        tests.append(row)

#Create X and y to train
for monster in monsters:
    if cont == 0:
        cont = cont + 1
        continue
    else:
        y.append(monster[6])
        x.append(monster[1:5])
        temp1.append(monster[5])

#Create X and y to test
cont = 0
for test in tests:
    if cont == 0:
        cont = cont + 1
        continue
    else:
        ids.append(test[0])
        x_test.append(test[1:5])
        temp2.append(test[5])
#Encode feature Color in train data
le = preprocessing.LabelEncoder()
le.fit(temp1)
result1 = le.transform(temp1)

#Encode feature Color in train data
le.fit(temp2)
result2 = le.transform(temp2)

result1 = np.float32(np.array(result1).reshape((len(result1),1)))
scaler = MinMaxScaler()

#Scale feature Color between 0 and 1 in train data
scaler.fit(result1)
minMax1 = scaler.transform(result1)
#Scale feature Color between 0 and 1 in test data
result2 = np.float32(np.array(result2).reshape((len(result2),1)))
scaler.fit(result2)
minMax2 = scaler.transform(result2)

#Prepair X and Y to train and adding feature color in X
X_train = np.float32(np.array(x))
X_train = np.insert(X_train,1,minMax1[:,0], axis=1)
y_train = np.array(y)

#Prepair X and Y to test and adding feature color in X
X_test = np.float32(np.array(x_test))
X_test = np.insert(X_test,1,minMax2[:,0], axis=1)
#Splitting the train in train and test to evaluate
train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

#Create MLP model
model = MLPClassifier(solver='adam', activation='logistic', alpha=1e-4,
                        batch_size=16, hidden_layer_sizes=(200,), learning_rate='adaptive', learning_rate_init=0.0001, max_iter=850, verbose=True)
model.fit(train_x,train_y)
#Make prediction on test of test file
pred = model.predict(X_test)
#Make prediction on test to evaluate
pred2 = model.predict(test_x)

print("Accuracy: ", accuracy_score(test_y, pred2))
print("Precision: ", precision_score(test_y, pred2, average='micro'))
print("Recall: ", recall_score(test_y, pred2, average='micro'))

#Create submission fileReader
with open('submission.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(pred)):
        writer.writerow({'id': int(ids[i]), 'type': pred[i]})
