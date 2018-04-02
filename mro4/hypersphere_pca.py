import random
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn import decomposition
import numpy as np



def belongs_to_sphere(radius, point):
    sum = 0
    for x in range(0, len(point)):
        sum += point[x] ** 2
    return sum <= radius ** 2


def generate_point(radius, dimensions):
    point = []

    for x in range(0, dimensions):
        point.append(random.uniform(-radius, radius))
    return point


sphere_radius = 1.0
num_of_points = 100


for dim in (3, 4, 5, 7, 13):
    columns = []
    df = pd.DataFrame(columns=columns)

    for i in range(1, dim + 1):
        columns.append("dim" + str(i))

    columns.append('color')


    for roll in itertools.product([-1, 1], repeat=dim):
        point = []
        point.extend(roll)
        point.append('red')
        df = df.append(pd.Series(point, index=columns), ignore_index=True)


    for c in range(0, num_of_points):
        point = generate_point(sphere_radius, dim)
        if belongs_to_sphere(sphere_radius, point):
            point.append('green')
        else:
            point.append('blue')
        df = df.append(pd.Series(point, index=columns), ignore_index=True)

    print(df[columns])

    X = np.array(df.ix[:, 1:len(columns)+1])
    y = np.array(df['color'])

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    x_val = [i[0] for i in X]
    y_val = [i[1] for i in X]


    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("PCA for " + str(dim) + " dimensions")
    for i in range(0, len(x_val)):
        plt.scatter(x_val[i], y_val[i], color=y[i])


plt.show()

