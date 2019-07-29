from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_train[:5]

A = (x>32).astype(int)
n = A.shape[0]


fig, ax = plt.subplots(1, n, figsize = (15, 10))

for i in range(n):
    ax[i].imshow(A[i], cmap='binary')
plt.show()



### -1~1로 변환
X = (2*A-1)



### W 계산
W = np.zeros((784, 784))

for i in range(n):
    W += X[i].reshape(-1, 1).dot(X[i].reshape(-1, 1).T)
    
W -= n*np.eye(len(W))



### Threshord 계산
threshold = -(np.sum(W, axis=1)/n).reshape(-1, 1)



### 활성화 함수
def update(Y, u):
    new_Y = np.zeros(u.shape)
    for i in range(len(new_Y)):
        if u[i]>0:
            new_Y[i] = 1
        elif u[i]<0:
            new_Y[i] = 0
        else:
            new_Y[i] = Y[i]
    return new_Y




### 샘플
noise = 0.2

B = A[4]*(1-noise) + np.random.rand(28, 28)*noise
plt.imshow(B.reshape(28, 28), cmap='binary')
plt.show()



### 결과
Y = (B).reshape(-1, 1)

u = W.dot(Y)+threshold
Y = update(Y, u)

plt.imshow(Y.reshape(28, 28), cmap='binary')
plt.show()