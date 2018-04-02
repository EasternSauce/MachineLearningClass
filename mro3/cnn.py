import os

from metric_learn import LMNN
from scipy import misc
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import CondensedNearestNeighbour



path = "."
image= misc.imread(os.path.join(path,'input2.bmp'), flatten= False)



mapped_colors = []
total_points = 0

for i in range(0, len(image)):
    for j in range(0, len(image[0])):
        mapped_row = []
        color = 0
        if image[i][j].tolist() == [255,0,0]:
            color = 1.0
            total_points+=1
        if image[i][j].tolist() == [0, 255, 0]:
            color = 2.0
            total_points+=1
        if image[i][j].tolist() == [0, 0, 255]:
            color = 3.0
            total_points+=1
        if color != 0:
            mapped_row.append(float(j))
            mapped_row.append(float(len(image[0])-i))
            mapped_row.append(color)
            mapped_colors.append(mapped_row)



def draw_knn(k, metric):
    names = ['x', 'y', 'color']

    df = pd.DataFrame(mapped_colors, columns=names)
    # print(df.head())

    X = np.array(df.ix[:, 0:2])
    y = np.array(df['color'])

    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if metric == 'mahalanobis':
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, metric_params={'V': np.cov(np.transpose(X))})
    else:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    err = 1 - accuracy_score(y_test, pred)
    print('\nThe error is ' + str(err*100))



    h = .02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"
              % k)


def draw_knn_with_lmnn(k, metric):
    names = ['x', 'y', 'color']

    df = pd.DataFrame(mapped_colors, columns=names)
    # print(df.head())

    X = np.array(df.ix[:, 0:2])
    y = np.array(df['color'])

    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(X, y)
    X_lmnn = lmnn.transform()

    X = X_lmnn

    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if metric == 'mahalanobis':
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, metric_params={'V': np.cov(np.transpose(X))})
    else:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    err = 1 - accuracy_score(y_test, pred)
    print('\nThe error is ' + str(err*100))

    h = .02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"
              % k)


def draw_cnn(k, metric):
    names = ['x', 'y', 'color']

    df = pd.DataFrame(mapped_colors, columns=names)
    # print(df.head())

    originalX = np.array(df.ix[:, 0:2])
    originaly = np.array(df['color'])

    cnn = CondensedNearestNeighbour(n_neighbors=k, return_indices=True)
    X_resampled, y_resampled, idx_resampled = cnn.fit_sample(originalX, originaly)

    X = X_resampled
    y = y_resampled

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if metric == 'mahalanobis':
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, metric_params={'V': np.cov(np.transpose(X))})
    else:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    err = 1 - accuracy_score(y_test, pred)
    print('\nThe error is ' + str(err*100))
    print('\nPercentage points left after CNN: ' + str(len(idx_resampled)/total_points*100))

    h = .02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    cmap_black = ListedColormap(['#FFFFFF','#FFFFFF','#FFFFFF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(originalX[:, 0], originalX[:, 1], c=originaly, cmap=cmap_black,
                edgecolor='k', s=20)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"
              % k)





#print(image)

draw_knn(1,'euclidean')
draw_cnn(1, 'euclidean')
draw_knn(3, 'euclidean')
draw_cnn(3, 'euclidean')



plt.show()


