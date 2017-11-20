# -*- coding: utf-8 -*-
"""
@author: Rajat
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import tqdm
import pickle

column_names = ['index','callfailure']
data2 = pd.read_csv('simlifiedX.csv')
y2 = pd.read_csv('yy.csv', header = None, names = column_names)

X = data2.iloc[:, [1,2,3,4,5]].values
y = y2.iloc[:,[1]]

y[y.callfailure != 0] = 1
cnt = (y['callfailure']==1).sum()
y=y.iloc[:,0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import sys
import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

plt.style.use('bmh')

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False


def _dtw_distance(ts_a, ts_b, d = lambda x,y: abs(x-y)):
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxsize * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window),
                       min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window 
    return cost[-1, -1]

# Identify the k nearest neighbors
#        knn_idx = dm.argsort()[:, :self.n_neighbors]
#
#        # Identify k nearest labels
#        knn_labels = self.l[knn_idx]
#        
#        # Model Label
#        mode_data = mode(knn_labels, axis=1)
#        mode_label = mode_data[0]
#        mode_proba = mode_data[1]/self.n_neighbors
#
#        return mode_label.ravel(), mode_proba.ravel()

class ProgressBar:
    """This progress bar was taken from PYMC
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print ('\r', self,)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
    
n_neighbors = 1
max_warping_window = 10
subsample_step = 1
xdot = X_train[::1000]
l = y_train[::1000]
#label, proba = m.predict(X_test[::10])
dm_count = 0     

x = X_train[::1]
y = xdot
x_s = x.shape

dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
# Compute condensed distance matrix (upper triangle) of pairwise dtw distances
# when x and y are the same array
if(np.array_equal(x, y)):
            
    p = ProgressBar(shape(dm)[0])
            
    for i in range(0, x_s[0] - 1):
        for j in range(i + 1, x_s[0]):
            dm[dm_count] = _dtw_distance(x[i, ::subsample_step],
              y[j, ::subsample_step])
                    
            dm_count += 1
            p.animate(dm_count)
            
        # Convert to squareform
        dm = squareform(dm)        
    # Compute full distance matrix of dtw distnces between x and y
else:
    x_s = np.shape(x)
    y_s = np.shape(y)
    dm = np.zeros((x_s[0], y_s[0])) 
    dm_size = x_s[0]*y_s[0]
            
    p = ProgressBar(dm_size)
    
    for i in range(0, x_s[0]):
        for j in range(0, y_s[0]):
            dm[i, j] = _dtw_distance(x[i, ::subsample_step],
                                      y[j, ::subsample_step])
        # Update progress bar
            dm_count += 1
            p.animate(dm_count)


df1=pd.DataFrame(dm)
print (pd.DataFrame(df1).to_csv('dm'))

x=X_test[::1]
y=xdot
x_s = x.shape
dm_count = 0
dm2 = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
# Compute condensed distance matrix (upper triangle) of pairwise dtw distances
# when x and y are the same array
if(np.array_equal(x, y)):
            
    p = ProgressBar(shape(dm2)[0])
            
    for i in range(0, x_s[0] - 1):
        for j in range(i + 1, x_s[0]):
            dm2[dm_count] = _dtw_distance(x[i, ::subsample_step],
              y[j, ::subsample_step])
                    
            dm_count += 1
            p.animate(dm_count)
            
        # Convert to squareform
        dm2 = squareform(dm2)        
    # Compute full distance matrix of dtw distnces between x and y
else:
    x_s = np.shape(x)
    y_s = np.shape(y)
    dm2 = np.zeros((x_s[0], y_s[0])) 
    dm_size = x_s[0]*y_s[0]
            
    p = ProgressBar(dm_size)
    
    for i in range(0, x_s[0]):
        for j in range(0, y_s[0]):
            dm2[i, j] = _dtw_distance(x[i, ::subsample_step],
                                      y[j, ::subsample_step])
        # Update progress bar
            dm_count += 1
            p.animate(dm_count)

df2=pd.DataFrame(dm2)
print (pd.DataFrame(df2).to_csv('dm2'))


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0,gamma=500,C=5)
classifier.fit(dm, y_train)

# Predicting the Test set results
y_pred = classifier.predict(dm2)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

