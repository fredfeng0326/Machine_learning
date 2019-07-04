from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])

sk_kmeans = KMeans(n_clusters=3)
sk_kmeans.fit(data2)
sk_C = sk_kmeans.predict(data2)


def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c

# 多加一列C
data_with_c = combine_data_C(data2, sk_C)

# print(data_with_c)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
plt.show()