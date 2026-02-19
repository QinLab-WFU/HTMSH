import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 数据
hash_code_1 = np.array([2.3933e-01, -2.3704e-01, 2.1899e-01, -1.2896e-01, -1.7266e-01,
-2.2546e-01, -2.1872e-01, 1.7051e-01, 2.8337e-01, 4.3529e-01,
-3.9596e-01, -3.0276e-01, -7.1987e-02, 2.4458e-01, -2.4647e-01,
1.4290e-01])

hash_code_2 = np.array([0.4223, -0.3162, 0.0844, -0.0993, -0.0259, -0.1858, -0.0909, 0.2478,
0.1931, 0.2874, -0.4886, -0.2959, -0.1185, 0.2324, -0.0731, 0.2835])

# 合并数据以进行降维
data = np.vstack((hash_code_1, hash_code_2))

# t-SNE降维
tsne = TSNE(n_components=2, random_state=0)
data_tsne = tsne.fit_transform(data)

# PCA降维
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# 绘制t-SNE降维结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data_tsne[0, 0], data_tsne[0, 1], color='red', label='With CB')
plt.scatter(data_tsne[1, 0], data_tsne[1, 1], color='blue', label='Without CB')
plt.title('t-SNE Dimensionality Reduction')
plt.legend()
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# 绘制PCA降维结果
# plt.subplot(1, 2, 2)
# plt.scatter(data_pca[0, 0], data_pca[0, 1], color='red', label='With CB')
# plt.scatter(data_pca[1, 0], data_pca[1, 1], color='blue', label='Without CB')
# plt.title('PCA Dimensionality Reduction')
# plt.legend()
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
#
plt.show()