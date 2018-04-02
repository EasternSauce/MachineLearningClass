import os

from scipy import misc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
import numpy as np
from sklearn.preprocessing import StandardScaler



path = "."
image= misc.imread(os.path.join(path,'input.bmp'), flatten= False)


mapped_colors = []

for i in range(0, len(image)):
    for j in range(0, len(image[0])):
        mapped_row = []
        color = ''
        if image[i][j].tolist() == [255,0,0]:
            color = 'red'
        if image[i][j].tolist() == [0, 128, 128]:
            color = 'green'
        if image[i][j].tolist() == [0, 0, 255]:
            color = 'blue'
        if image[i][j].tolist() == [255, 255, 0]:
            color = 'yellow'
        if color != '':
            mapped_row.append(float(j))
            mapped_row.append(float(len(image[0])-i))
            mapped_row.append(color)
            mapped_colors.append(mapped_row)


df = pd.DataFrame(mapped_colors,columns=["x", "y", "color"])


X = np.array(df.ix[:, 0:2])
X_std = StandardScaler().fit_transform(X)
y = np.array(df['color'])

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)




pca = decomposition.PCA()
pca.fit(X)
pca_X = pca.transform(X)

kernel_cosine_pca = decomposition.KernelPCA(kernel='cosine')
kernel_cosine_pca.fit(X)
kernel_cosine_X = kernel_cosine_pca.transform(X)


kernel_rbf_pca = decomposition.KernelPCA(kernel='rbf')
kernel_rbf_pca.fit(X)
kernel_rbf_X = kernel_rbf_pca.transform(X)

kernel_rbf_pca_gamma5 = decomposition.KernelPCA(kernel='rbf', gamma=5)
kernel_rbf_pca_gamma5.fit(X)
kernel_rbf_X_gamma5 = kernel_rbf_pca_gamma5.transform(X)

kernel_rbf_pca_gamma10 = decomposition.KernelPCA(kernel='rbf', gamma=10)
kernel_rbf_pca_gamma10.fit(X)
kernel_rbf_X_gamma10 = kernel_rbf_pca_gamma10.transform(X)

kernel_rbf_pca_gamma15 = decomposition.KernelPCA(kernel='rbf', gamma=15)
kernel_rbf_pca_gamma15.fit(X)
kernel_rbf_X_gamma15 = kernel_rbf_pca_gamma15.transform(X)


x_val = [i[0] for i in X]
y_val = [i[1] for i in X]
pca_x_val = [i[0] for i in pca_X]
pca_y_val = [i[1] for i in pca_X]
kernel_cosine_x_val = [i[0] for i in kernel_cosine_X]
kernel_cosine_y_val = [i[1] for i in kernel_cosine_X]
kernel_rbf_x_val = [i[0] for i in kernel_rbf_X]
kernel_rbf_y_val = [i[1] for i in kernel_rbf_X]
kernel_rbf_gamma5_x_val = [i[0] for i in kernel_rbf_X_gamma5]
kernel_rbf_gamma5_y_val = [i[1] for i in kernel_rbf_X_gamma5]
kernel_rbf_gamma10_x_val = [i[0] for i in kernel_rbf_X_gamma10]
kernel_rbf_gamma10_y_val = [i[1] for i in kernel_rbf_X_gamma10]
kernel_rbf_gamma15_x_val = [i[0] for i in kernel_rbf_X_gamma15]
kernel_rbf_gamma15_y_val = [i[1] for i in kernel_rbf_X_gamma15]

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA")

for i in range(0, len(pca_x_val)):
    plt.scatter(pca_x_val[i], pca_y_val[i], color=y[i], s=1)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("Before PCA (with eigenvectors)")


for i in range(0, len(x_val)):
    plt.scatter(x_val[i], y_val[i], color=y[i], s=1)

for i in eig_vecs:
    plt.quiver(i[0]*eig_vals[0], i[1]*eig_vals[1], angles='xy', scale_units='xy', scale=1)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with cosine kernel")

for i in range(0, len(kernel_cosine_x_val)):
    plt.scatter(kernel_cosine_x_val[i], kernel_cosine_y_val[i], color=y[i], s=1)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel")

for i in range(0, len(kernel_rbf_x_val)):
    plt.scatter(kernel_rbf_x_val[i], kernel_rbf_y_val[i], color=y[i], s=1)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel (gamma=5)")


for i in range(0, len(kernel_rbf_gamma5_x_val)):
    plt.scatter(kernel_rbf_gamma5_x_val[i], kernel_rbf_gamma5_y_val[i], color=y[i], s=1)



plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel (gamma=10)")

for i in range(0, len(kernel_rbf_gamma10_x_val)):
    plt.scatter(kernel_rbf_gamma10_x_val[i], kernel_rbf_gamma10_y_val[i], color=y[i], s=1)

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel (gamma=15)")

for i in range(0, len(kernel_rbf_gamma15_x_val)):
    plt.scatter(kernel_rbf_gamma15_x_val[i], kernel_rbf_gamma15_y_val[i], color=y[i], s=1)



path = "."
image= misc.imread(os.path.join(path,'input2.bmp'), flatten= False)


mapped_colors = []

for i in range(0, len(image)):
    for j in range(0, len(image[0])):
        mapped_row = []
        color = ''
        if image[i][j].tolist() == [255,0,0]:
            color = 'red'
        if image[i][j].tolist() == [0, 128, 128]:
            color = 'green'
        if image[i][j].tolist() == [0, 0, 255]:
            color = 'blue'
        if image[i][j].tolist() == [255, 255, 0]:
            color = 'yellow'
        if color != '':
            mapped_row.append(float(j))
            mapped_row.append(float(len(image[0])-i))
            mapped_row.append(color)
            mapped_colors.append(mapped_row)

df = pd.DataFrame(mapped_colors,columns=["x", "y", "color"])


X = np.array(df.ix[:, 0:2])
X_std = StandardScaler().fit_transform(X)
y = np.array(df['color'])

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)




pca = decomposition.PCA()
pca.fit(X)
pca_X = pca.transform(X)

kernel_cosine_pca = decomposition.KernelPCA(kernel='cosine')
kernel_cosine_pca.fit(X)
kernel_cosine_X = kernel_cosine_pca.transform(X)


kernel_rbf_pca = decomposition.KernelPCA(kernel='rbf')
kernel_rbf_pca.fit(X)
kernel_rbf_X = kernel_rbf_pca.transform(X)


kernel_rbf_pca_gamma5 = decomposition.KernelPCA(kernel='rbf', gamma=5)
kernel_rbf_pca_gamma5.fit(X)
kernel_rbf_X_gamma5 = kernel_rbf_pca_gamma5.transform(X)

kernel_rbf_pca_gamma10 = decomposition.KernelPCA(kernel='rbf', gamma=10)
kernel_rbf_pca_gamma10.fit(X)
kernel_rbf_X_gamma10 = kernel_rbf_pca_gamma10.transform(X)

kernel_rbf_pca_gamma15 = decomposition.KernelPCA(kernel='rbf', gamma=15)
kernel_rbf_pca_gamma15.fit(X)
kernel_rbf_X_gamma15 = kernel_rbf_pca_gamma15.transform(X)


x_val = [i[0] for i in X]
y_val = [i[1] for i in X]
pca_x_val = [i[0] for i in pca_X]
pca_y_val = [i[1] for i in pca_X]
kernel_cosine_x_val = [i[0] for i in kernel_cosine_X]
kernel_cosine_y_val = [i[1] for i in kernel_cosine_X]
kernel_rbf_x_val = [i[0] for i in kernel_rbf_X]
kernel_rbf_y_val = [i[1] for i in kernel_rbf_X]
kernel_rbf_gamma5_x_val = [i[0] for i in kernel_rbf_X_gamma5]
kernel_rbf_gamma5_y_val = [i[1] for i in kernel_rbf_X_gamma5]
kernel_rbf_gamma10_x_val = [i[0] for i in kernel_rbf_X_gamma10]
kernel_rbf_gamma10_y_val = [i[1] for i in kernel_rbf_X_gamma10]
kernel_rbf_gamma15_x_val = [i[0] for i in kernel_rbf_X_gamma15]
kernel_rbf_gamma15_y_val = [i[1] for i in kernel_rbf_X_gamma15]


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA")

for i in range(0, len(pca_x_val)):
    plt.scatter(pca_x_val[i], pca_y_val[i], color=y[i], s=1)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("Before PCA (with eigenvectors)")


for i in range(0, len(x_val)):
    plt.scatter(x_val[i], y_val[i], color=y[i], s=1)

for i in eig_vecs:
    plt.quiver(i[0]*eig_vals[0], i[1]*eig_vals[1], angles='xy', scale_units='xy', scale=1)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with cosine kernel")

for i in range(0, len(kernel_cosine_x_val)):
    plt.scatter(kernel_cosine_x_val[i], kernel_cosine_y_val[i], color=y[i], s=1)


plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel")

for i in range(0, len(kernel_rbf_x_val)):
    plt.scatter(kernel_rbf_x_val[i], kernel_rbf_y_val[i], color=y[i], s=1)

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel (gamma=5)")


for i in range(0, len(kernel_rbf_gamma5_x_val)):
    plt.scatter(kernel_rbf_gamma5_x_val[i], kernel_rbf_gamma5_y_val[i], color=y[i], s=1)



plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel (gamma=10)")

for i in range(0, len(kernel_rbf_gamma10_x_val)):
    plt.scatter(kernel_rbf_gamma10_x_val[i], kernel_rbf_gamma10_y_val[i], color=y[i], s=1)

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title("PCA with rba kernel (gamma=15)")

for i in range(0, len(kernel_rbf_gamma15_x_val)):
    plt.scatter(kernel_rbf_gamma15_x_val[i], kernel_rbf_gamma15_y_val[i], color=y[i], s=1)



plt.show()