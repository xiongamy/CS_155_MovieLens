import numpy as np
import pandas as pd
import re

movie_cats = ['movie_id', 'movie_title', 'unknown', 'action', 'adventure',
			  'animation', 'childrens', 'comedy', 'crime', 'documentary',
			  'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
			  'romance', 'scifi', 'thriller', 'war', 'western']

movies = pd.read_csv('../data/movies.txt', sep='\t', names=movie_cats, encoding='latin-1')
titles = np.array(movies)[:,1]

# Finds year (beginning with 19--) of movie
for i, t in enumerate(titles):
	pattern = re.compile('\((19\d\d)\)')
	year = pattern.findall(t)
	if len(year) == 0:
		titles[i] = -1
	else:
		titles[i] = int(year[0])

movies['year'] = titles

id_year = np.array(movies)[:, [0, 21]]

np.savetxt('../data/movie_ages.txt', id_year, fmt='%i', delimiter=" ")