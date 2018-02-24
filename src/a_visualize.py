import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import sys

movie_ages = np.loadtxt('../data/movie_ages.txt', delimiter=' ').astype(int)
movie_names = np.loadtxt('../data/movies.txt', delimiter='\t', dtype='str',
                             usecols=[1])
def main(V):
    # 20's, 30's, ..., 90's
    decades = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]
    decade_indices = [[] for i in range(8)]
    
    for j, movie in enumerate(movie_ages):
        id = j + 1
        year = movie[1]
        for i in range(7, -1, -1):
            if year > decades[i]:
                decade_indices[i].append(id)
                break
    
    x_means = []
    y_means = []
    x_stds = []
    y_stds = []
    labels = []
   
    for i, decade in enumerate(decade_indices):
        labels.append(str(decades[i]) + '\'s')
        
        # get x and y values for the movie groups
        x = []
        y = []
        for di in decade:
            v = V[di - 1]
            x.append(v[0])
            y.append(v[1])
            
        # get means and stds
        x_means.append(np.mean(x))
        x_stds.append(np.std(x))
        y_means.append(np.mean(y))
        y_stds.append(np.std(y))
        
    # plot
    ells = [Ellipse(xy=(x_means[i], y_means[i]), width=x_stds[i], height=y_stds[i]) for i in range(7, -1, -1)]
    
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    colors = ['black', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
    
    for i, e in enumerate(ells):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)
        e.set_facecolor(colors[i])
    
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)
    ps = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(8)]
    plt.legend(handles=ps)
    plt.show()

if __name__ == "__main__":
    # usage: mf_visualize.py [filename].
    # V is imported from [filename]
    V = np.loadtxt(sys.argv[1], delimiter=',')
    main(V)
    
