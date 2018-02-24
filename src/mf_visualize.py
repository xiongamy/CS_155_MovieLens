import numpy as np
import matplotlib.pyplot as plt
import sys

MOVIES_FILE = '../data/movies.txt'
DATA_FILE = '../data/data.txt'
    
def normalize(l):
    return (np.array(l) - np.mean(l)) / np.std(l)
    
def dict_of_ratings(all_data):
    # create dictionary of ratings, where the key is the movie ID
    # and the value is a list of all the ratings
    m_ratings = {}
    for d in all_data:
        movie = d[1]
        rating = d[2]
        if movie in m_ratings:
            m_ratings[movie].append(rating)
        else:
            m_ratings[movie] = [rating]
            
    return m_ratings
    
def most_popular(V):
    m_ratings = dict_of_ratings(np.loadtxt(DATA_FILE).astype(int))
    movie_names = np.loadtxt(MOVIES_FILE, delimiter='\t', dtype='str',
                             usecols=[1])
            
    # sort dictionary of ratings by the number of ratings each movie has
    sorted_ratings = sorted(m_ratings.items(), key=lambda x: len(x[1]),
                            reverse=True)
    sorted_ids = [s[0] for s in sorted_ratings]
    
    # get x and y values for the 10 most popular movies
    x = []
    y = []
    for i in range(10):
        id = sorted_ids[i]
        v = V[id]
        x.append(v[0])
        y.append(v[1])
        
    # normalize coordinates
    x = normalize(x)
    y = normalize(y)
        
    # plot
    plt.scatter(x, y)
    for i in range(10):
        id = sorted_ids[i]
        name = movie_names[id - 1]
        plt.annotate(name, xy=(x[i], y[i]), xytext=(-25, 5), textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    
    plt.title('10 most popular movies')
    plt.show()
    
def average_rating(ratings):
    # computes the average of the list of ratings
    return float(sum(ratings)) / len(ratings)
    
def best_movies(V):
    m_ratings = dict_of_ratings(np.loadtxt(DATA_FILE).astype(int))
    movie_names = np.loadtxt(MOVIES_FILE, delimiter='\t', dtype='str',
                             usecols=[1])
    
    # sort dictionary of ratings by each movie's average rating
    sorted_ratings = sorted(m_ratings.items(),
                            key=lambda x: float(sum(x[1])) / len(x[1]),
                            reverse=True)
    sorted_ids = [s[0] for s in sorted_ratings]
    
    # get x and y values for the 10 best movies
    x = []
    y = []
    for i in range(10):
        id = sorted_ids[i]
        v = V[id]
        x.append(v[0])
        y.append(v[1])
        
    # normalize coordinates
    x = normalize(x)
    y = normalize(y)
        
    # plot
    plt.scatter(x, y)
    for i in range(10):
        id = sorted_ids[i]
        name = movie_names[id - 1]
        plt.annotate(name, xy=(x[i], y[i]), xytext=(-25, 5), textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
                     
    plt.title('10 best movies')
    plt.show()
    
def all_in_genre(V, genre):
    genre_ids = {'unknown': 1, 'action': 2, 'adventure': 3, 'animation': 4,
                 'childrens': 5, 'comedy': 6, 'crime': 7, 'documentary': 8,
                 'drama': 9, 'fantasy': 10, 'film-noir': 11, 'horror': 12,
                 'musical': 13, 'mystery': 14, 'romance': 15, 'sci-fi': 16,
                 'thriller': 17, 'war': 18, 'western': 19}
    cols = [0] + list(range(2, 21))
    movie_data = np.loadtxt(MOVIES_FILE, delimiter='\t',
                            usecols=cols).astype(int)
    
    # get the IDs of all the movies in the genre
    g_id = genre_ids[genre]
    filtered_movies = [elem[0] for elem in movie_data if elem[g_id] == 1]
    
    # get x and y values for all the movies in the genre
    x = []
    y = []
    for id in filtered_movies:
        v = V[id]
        x.append(v[0])
        y.append(v[1])
        
    # normalize coordinates
    x = normalize(x)
    y = normalize(y)
        
    # plot
    plt.scatter(x, y)
    plt.title('All ' + genre + ' movies')
    plt.show()

def handpicked(V):
    movie_names = np.loadtxt(MOVIES_FILE, delimiter='\t', dtype='str',
                             usecols=[1])
            
    # get x and y values for the 10 most popular movies
    x = []
    y = []
    for i in [8, 32, 42, 57, 129, 149, 234, 263, 273, 1242]:
        id = sorted_ids[i - 1]
        v = V[id]
        x.append(v[0])
        y.append(v[1])
        
    # normalize coordinates
    x = normalize(x)
    y = normalize(y)
        
    # plot
    plt.scatter(x, y)
    for i in range(10):
        id = sorted_ids[i]
        name = movie_names[id - 1]
        plt.annotate(name, xy=(x[i], y[i]), xytext=(-25, 5), textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))
    
    plt.title('10 handpicked movies')
    plt.show()
    
if __name__ == "__main__":
    # usage: mf_visualize.py [filename] [type] [opt: genre].
    # V is imported from [filename]
    # [type] is one of 'popular', 'best', 'genre'.
    # The genre argument is the name of one of the 19 genres (see function
    # all_in_genre for the list of appropriate genres), and is
    # only used if [type] is 'genre'.
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
    
