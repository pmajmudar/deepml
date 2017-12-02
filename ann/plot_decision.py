import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y):
    h = 0.02
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))
    Z = model(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.figure(figsize=(10, 10))
    #plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired)
    plt.contourf(xx1, xx2, Z)
    #plt.axis('off')
    plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap=plt.cm.Paired, edgecolors='black')
    plt.show()
