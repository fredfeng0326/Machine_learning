import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex7data1.mat')
# print(mat.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'X'])

data1 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
# print(data1.head())
"""
         X1        X2
0  3.381563  3.389113
1  4.527875  5.854178
2  2.655682  4.411995
3  2.765235  3.715414
4  2.846560  4.175506
"""

sns.set(context="notebook", style="white")

# fit_reg If True, estimate and plot a regression model relating the x and y variables.
sns.lmplot('X1', 'X2', data=data1, fit_reg=False)
plt.show()