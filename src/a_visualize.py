import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

movie_ages = np.loadtxt('../data/movie_ages.txt', delimiter=' ')
movie_names = np.loadtxt('../data/movies.txt', delimiter='\t', dtype='str',
                             usecols=[1])
    
    # 20's, 30's, ..., 90's
    decades = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]
    decade_indices = [[] for i in range(8)]
    
    for movie in movie_ages:
        id = movie[0]
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
    
    for i, decade in enumerate(decades):
        labels.append(str(decade) + '\'s')
        
        # get x and y values for the movie groups
        x = []
        y = []
        for di in decade_indices:
            id = sorted_ids[di - 1]
            v = V[id]
            x.append(v[0])
            y.append(v[1])
            
        # get means and stds
        x_means.append(np.mean(x))
        x_stds.append(np.std(x))
        y_means.append(np.mean(y)
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
    
    plt.show()

if __name__ == "__main__":
    # usage: mf_visualize.py [filename].
    # V is imported from [filename]
    V = np.loadtxt(sys.argv[1], delimiter=',')
    type = sys.argv[2]
    if type == 'popular':
        most_popular(V)
    elif type == 'best':
        best_movies(V)
    elif type == 'genre':
        all_in_genre(V, sys.argv[3])
    elif type == 'handpicked':
        handpicked(V)
    
