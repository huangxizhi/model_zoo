# -*- coding:utf-8 -*-
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 500)
np.random.shuffle(X)
Y = 0.5 * X + 0.2 + np.random.normal(0, 0.05, (500,))

X_train, Y_train = X[:400], Y[:400]
X_val, Y_val = X[400:], Y[400:]

plt.scatter(X, Y)
plt.show()

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

print('begin to train...')
for step in range(500):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost:', cost)

cost = model.evaluate(X_val, Y_val, batch_size=100)
print('test cost', cost)
W, b = model.layers[0].get_weights()
print('weight=', W, 'b=', b)


Y_pred = model.predict(X_val)
plt.scatter(X_val, Y_val)
plt.plot(X_val, Y_pred)
plt.show()

