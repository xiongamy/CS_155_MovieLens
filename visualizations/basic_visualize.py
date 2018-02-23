import numpy as np
import matplotlib.pyplot as plt
import sys
    
def all_movies(all_data):
    ratings = data[:, 2]
    plt.hist(ratings, range(1, 7), align='left', zorder=3)
    plt.title('Ratings of all movies')
    plt.xlabel('Rating')
    plt.ylabel('Counts')
    plt.grid(True, axis='y', zorder=0)
    plt.show()
    
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
    
def most_popular(all_data):
    m_ratings = dict_of_ratings(all_data)
            
    # sort dictionary of ratings by the number of ratings each movie has
    sorted_ratings = sorted(m_ratings.values(), key=len, reverse=True)
    
    # append all the ratings from the 10 most popular movies together
    popular_ratings = []
    for i in range(10):
        popular_ratings += sorted_ratings[i]
        
    # plot
    plt.hist(popular_ratings, range(1, 7), align='left', zorder=3)
    plt.title('Ratings of the 10 most popular movies')
    plt.xlabel('Rating')
    plt.ylabel('Counts')
    plt.grid(True, axis='y', zorder=0)
    plt.show()
    
def average_rating(ratings):
    # computes the average of the list of ratings
    return float(sum(ratings)) / len(ratings)
    
def best_movies(all_data):
    m_ratings = dict_of_ratings(all_data)
    
    # sort dictionary of ratings by each movie's average rating
    sorted_ratings = sorted(m_ratings.values(), key=average_rating,
                            reverse=True)
    
    # append all the ratings from the 10 best rated movies together
    best_ratings = []
    for i in range(10):
        best_ratings += sorted_ratings[i]
        
    # plot
    plt.hist(best_ratings, range(1, 7), align='left', zorder=3)
    plt.title('Ratings of the 10 best movies')
    plt.xlabel('Rating')
    plt.ylabel('Counts')
    plt.grid(True, axis='y', zorder=0)
    plt.show()
    
def all_in_genre(all_data, genre):
    genre_ids = {'unknown': 1, 'action': 2, 'adventure': 3, 'animation': 4,
                 'childrens': 5, 'comedy': 6, 'crime': 7, 'documentary': 8,
                 'drama': 9, 'fantasy': 10, 'film-noir': 11, 'horror': 12,
                 'musical': 13, 'mystery': 14, 'romance': 15, 'sci-fi': 16,
                 'thriller': 17, 'war': 18, 'western': 19}
    cols = [0] + list(range(2, 21))
    movie_data = np.loadtxt('../data/movies.txt', delimiter='\t',
                            usecols=cols).astype(int)
    
    # get the IDs of all the movies in the genre
    g_id = genre_ids[genre]
    filtered_movies = [elem[0] for elem in movie_data if elem[g_id] == 1]
    
    # get the ratings of all the movies in the genre
    m_ratings = dict_of_ratings(all_data)
    genre_ratings = []
    for m_id in filtered_movies:
        genre_ratings += m_ratings[m_id]
        
    # plot
    plt.hist(genre_ratings, range(1, 7), align='left', zorder=3)
    plt.title('Ratings of all ' + genre + ' movies')
    plt.xlabel('Rating')
    plt.ylabel('Counts')
    plt.grid(True, axis='y', zorder=0)
    plt.show()
    
if __name__ == "__main__":
    # usage: basic_visualize.py [type] [opt: genre].
    # [type] is one of 'all', 'popular', 'best', 'genre'.
    # The genre argument is the name of one of the 19 genres (see function
    # all_in_genre for the list of appropriate genres), and is
    # only used if [type] is 'genre'.
    data = np.loadtxt('../data/data.txt').astype(int)
    type = sys.argv[1]
    if type == 'all':
        all_movies(data)
    elif type == 'popular':
        most_popular(data)
    elif type == 'best':
        best_movies(data)
    elif type == 'genre':
        all_in_genre(data, sys.argv[2])
    
