from neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score
import numpy as np

#Consider a pattern: If the MSB and the LSB are 1 then the output is 1. Else, the output is 0

Xtrain = np.array([[np.random.randint(0, 2) for i in range(10)] for j in range(250)])
ytrain = [[1] if i[0] == 1 and i[-1] == 1 else [0] for i in Xtrain]

model = NeuralNetwork(10, 2, 1)

model.train(Xtrain, ytrain, 15000)

Xtest = np.array([[np.random.randint(0, 2) for i in range(10)] for j in range(50)])
ytest = [1 if i[0] == 1 and i[-1] == 1 else 0 for i in Xtest]

ypred = [model.predict(i) for i in Xtest]

for i, j, k in zip(Xtest, ypred, ytest):

    print(f"{i} ---> {j}\t(Actual: {k})")

print(f"\nAccuracy: {accuracy_score(ytest, ypred)*100}%")