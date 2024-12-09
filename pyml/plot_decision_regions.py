# copyright unknown :) believe it is public domain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier=None, resolution=0.02):
    
    labels = np.unique(y)
    markers = [ 's', 'o', '^', 'v', 'x' ]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_map = ListedColormap( colors[:len(labels)] )
    
    x1_min, x2_min = X.min(0) - 1
    x1_max, x2_max = X.max(0) + 1
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    
    # decision areas
    if classifier:
        x1r = np.arange(x1_min, x1_max, resolution)
        x2r = np.arange(x2_min, x2_max, resolution)
        xx1, xx2 = np.meshgrid(x1r, x2r)
        ar = np.array([xx1.ravel(), xx2.ravel()]).T
        Z = classifier.predict(ar).reshape(xx1.shape)
        plt.contourf( xx1, xx2, Z, alpha=0.3, cmap=color_map)
        
    
    
    # samples
    for k, cl in enumerate(labels):
        idx = ( y == cl )
        plt.scatter(x=X[idx,0], y=X[idx,1],
                    alpha=0.8, c=colors[k], marker=markers[k],
                    label=cl, edgecolor='black')