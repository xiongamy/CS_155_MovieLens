# Necessary to be in 2.7 to use graphlab.
import numpy as np
import pandas as pd
import graphlab

names = ['user_id', 'movie_id', 'rating']
train = graphlab.SFrame(pd.read_csv('../data/train.txt', sep='\t', names=names, encoding='latin-1'))
test = graphlab.SFrame(pd.read_csv('../data/test.txt', sep='\t', names=names, encoding='latin-1'))
movie_cats = ['movie_id', 'movie_title', 'unknown', 'action', 'adventure',
			  'animation', 'childrens', 'comedy', 'crime', 'documentary',
			  'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
			  'romance', 'scifi', 'thriller', 'war', 'western']
movies = graphlab.SFrame(pd.read_csv('../data/movies.txt', sep='\t', names=movie_cats, encoding='latin-1'))

sim_model = graphlab.factorization_recommender.create(train, user_id='user_id', item_id='movie_id', target='rating',
														num_factors=20, regularization=0.1, max_iterations=4000)

print "\n"
print "Test set info:"
rec = sim_model.evaluate(test)

coef = sim_model.get('coefficients').get('movie_id').select_column('factors').to_numpy()

R, Sigma, S = np.linalg.svd(np.transpose(coef))
tildeV = np.matmul(coef, R[:, :2])

np.savetxt('../results/v_graphlab.csv', tildeV, delimiter=',')

#view = sim_model.views.overview(validation_set=test, item_data=movies, item_name_column='movie_title')
#view.show()
#i = raw_input("Press enter to exit.")