from sklearn.cluster import KMeans
import os
from scipy import misc, random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import fowlkes_mallows_score, homogeneity_score, adjusted_mutual_info_score, v_measure_score, \
    silhouette_score

#np.set_printoptions(threshold=np.nan)

path = "."
image= misc.imread(os.path.join(path,'input.bmp'), flatten= False)



mapped_colors = []

for i in range(0, len(image)):
    for j in range(0, len(image[0])):
        mapped_row = []
        color = ''
        if image[i][j].tolist() == [255,0,0]:
            mapped_row.append(float(j))
            mapped_row.append(float(len(image[0])-i))
            mapped_colors.append(mapped_row)

df = pd.DataFrame(mapped_colors, columns=["x", "y"])

X = np.array(df.ix[:, 0:2])

#print(X)

#print(df)



def perform_kmeans(init, iteration, seed):
    init_array = []

    im_width = len(image[0])
    im_height = len(image) + len(image[0]) / 2

    for i in range(0, 9):
        init_array.append([random.uniform(0, im_width - 1), random.uniform(0, im_height - 1)])
    init_array = np.array(init_array)

    kmeans_part = KMeans(n_clusters=9, init=init_array, max_iter=1, n_init=1)
    data_copy = df.ix[:, 0:2]
    kmeans_part.fit(data_copy)
    init_array_part = kmeans_part.cluster_centers_

    target_data = df.ix[:, 0:2]
    target_kmeans = KMeans(n_clusters=9, init='random')
    target_kmeans.fit(target_data)



    data = df.ix[:, 0:2]
    #print(im_width, ' ', im_height)



    if init == 'random':
        kmeans = KMeans(n_clusters=9, init=init_array, max_iter=iteration, n_init=1, random_state=seed)
    elif init == 'forgy':
        kmeans = KMeans(n_clusters=9, init='random', max_iter=iteration, random_state=seed)
    elif init == 'kmeans++':
        kmeans = KMeans(n_clusters=9, init='k-means++', max_iter=iteration, n_init=1, random_state=seed)
    elif init == 'randompartition':
        kmeans = KMeans(n_clusters=9, init=init_array_part, max_iter=iteration, n_init=1, random_state=seed)

    kmeans.fit(data)

    print('score for init: ' , init , ' is ' , v_measure_score(target_kmeans.labels_, kmeans.labels_))
    '''

    h = .02

    x_min, x_max = data.values[:, 0].min() - 1, data.values[:, 0].max() + 1
    y_min, y_max = data.values[:, 1].min() - 1, data.values[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)


    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(data.values[:, 0], data.values[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('k-means with init: ' + init)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    return adjusted_mutual_info_score(data, kmeans.labels_, metric='euclidean')
'''

seeds = []

for z in range(0, 20):
    seeds.append(int(random.uniform(1,100)))

mean_results = []
std_results = []

for i in range(1,11):
    attempts = []
    for attempt in range(0, 5):
        attempts.append(perform_kmeans('forgy', i, seeds[attempt]))

    mean_results.append(np.mean(attempts))
    std_results.append(np.std(attempts))





plt.plot(range(1, 11), mean_results, 'ro', markersize=1)
plt.xlabel('iteration')
plt.ylabel('score')
plt.errorbar(range(1, 11), mean_results, yerr=std_results, ls='none')




plt.show()