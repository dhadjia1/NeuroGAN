import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plt_init():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def points2d(ax, x, y, color='r'):
    ax.scatter(x,y,c=color)

def plot_pc_with_eigenvector(xy, eigenvalues, eigenvectors, fig=None, ax=None, first=True, show=True):
    if first:
        fig, ax = plt_init()
    points2d(ax, xy[:,0], xy[:,1],color='r')
    zero = np.zeros(eigenvalues.shape[0])
    sqrteigen = np.zeros(eigenvalues.shape[0])
    
    sqrteigen[0] = eigenvalues[0] ** 0.5
    sqrteigen[1] = eigenvalues[1] ** 0.5
    
    x1, x2 = zero[0], sqrteigen[0]*eigenvectors[0,0]
    y1, y2 = zero[1], sqrteigen[0]*eigenvectors[0,1]

    ax.plot([x1, x2], [y1, y2], 'b-')


    x1, x2 = zero[0], sqrteigen[1]*eigenvectors[1,0]
    y1, y2 = zero[1], sqrteigen[1]*eigenvectors[1,1]


    ax.plot([x1, x2], [y1, y2], 'b-')
    if show:
        plt.show()
    else:
        return fig, ax  
        
def plot_bounds(ub, lb, fig=None, ax=None, first=True, show=True):
    if first:
        fig, ax = plt_init()
    x1, x2 = ub[0], ub[0]
    y1, y2 = ub[1], lb[1]
    
    ax.plot([x1, x2], [y1, y2], 'g-')

    x1, x2 = lb[0], lb[0]
    y1, y2 = ub[1], lb[1]

    ax.plot([x1, x2], [y1, y2], 'g-')
    
    x1, x2 = ub[0], lb[0]
    y1, y2 = ub[1], ub[1]
  
    ax.plot([x1, x2], [y1, y2], 'g-')
  
    x1, x2 = ub[0], lb[0]
    y1, y2 = lb[1], lb[1]

    ax.plot([x1, x2], [y1, y2], 'g-')

    if show:
        plt.show()
    else:
        return fig, ax