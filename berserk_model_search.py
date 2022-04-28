import pandas as pd
import numpy as np
from enum import IntEnum
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
import joblib

t_total = time()
t = time()
print('Processing games...')

# We encoded white and black berserk data in one column in the csv to save space
class Berserk(IntEnum):
    NEITHER = 0
    WHITE = 1
    BLACK = 2
    BOTH = 3
# Same deal with Termination, we used int instead of string
class Termination(IntEnum):
    NORMAL = 0
    TIME_FORFEIT = 1

## Outside this file, we combined all csvs into one using csvkit:
## $ csvstack *.csv  > all.csv
## 
## Then: 
##
# games = pd.read_csv(
#     '../lichess_games/all.csv', 
#     usecols=[
#         'white_rating', 
#         'black_rating', 
#         'time', 
#         'berserk', 
#         'result'
#     ],
#     dtype={ # np.ushort = uint16, np.ubyte = uint8
#         'white_rating': np.ushort, 
#         'black_rating': np.ushort, 
#         'time': np.ubyte, 
#         'berserk': np.ubyte, 
#         'result': np.ubyte
#     }
# )
# joblib.dump(games, '../lichess_games/all.pkl')

games = joblib.load('../lichess_games/all.pkl')

games['white_berserked'] = (games['berserk'] == Berserk.WHITE) | (games['berserk'] == Berserk.BOTH)
games['black_berserked'] = (games['berserk'] == Berserk.BLACK) | (games['berserk'] == Berserk.BOTH)
games['white_berserked'] = games['white_berserked'].astype(np.ubyte)
games['black_berserked'] = games['black_berserked'].astype(np.ubyte)
games.drop(columns=['berserk'], inplace=True) 

time_control = 3 # the time control we want (in min). the data has only 1+0 and 3+0 games
games = games[(games['time'] == time_control)] 
games.reset_index(inplace=True, drop=True)
games.drop(columns=['time'], inplace=True) 

min_rating = 600 # n.b. lichess doesn't let ratings go below 600
max_rating = 3200 if time_control == 1 else 3000 # there are too few games where either player had a higher rating

games = games[
    (games['white_rating'] >= min_rating) & (games['white_rating'] < max_rating) &
    (games['black_rating'] >= min_rating) & (games['black_rating'] < max_rating) 
]

# work with a subset of the data so everything can fit in RAM
games = games.sample(frac=1.0, random_state=0) if time_control == 1 else games.sample(frac=0.65, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(
    games[['white_rating', 'black_rating']],
    games[['white_berserked', 'black_berserked']],
    test_size=0.25,
    random_state=0
)

del games

print(
    'Took {:.0f} seconds. Using {:,} games for training and {:,} for test'
    .format(time() - t, x_train.shape[0], x_test.shape[0])
)

# poly = PolynomialFeatures( 
#     2, 
#     interaction_only=True, 
#     include_bias=False 
# )
spline = SplineTransformer(
    include_bias=True
)
scale = StandardScaler() # scale features to zero mean and unit variance
clf = LogisticRegression(
    fit_intercept=True,
    solver='saga', # only solver that can handle elasticnet, l1, and l2
    penalty='elasticnet',
    C = 0.0001,
    l1_ratio=0.5,
    random_state=0,
    verbose=1, # 1 prints a line for each epoch for each job
    max_iter=200
)

clf = MultiOutputClassifier(clf, n_jobs=1)
pipe = Pipeline(steps=[('spline', spline), ('scale', scale), ('clf', clf)])

pipe.fit(x_train, y_train)

print(f'\nSaving search results as pickles/{time_control}_berserk_model_search.pkl')
joblib.dump(pipe, f'pickles/{time_control}_berserk_model_search.pkl')
print('Search took {:.2f} hours in total'.format((time() - t_total)/60/60))




