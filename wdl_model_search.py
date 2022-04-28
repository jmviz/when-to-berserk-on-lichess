import pandas as pd
import numpy as np
from enum import IntEnum
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
    games[['white_rating', 'black_rating', 'white_berserked', 'black_berserked']],
    games['result'],
    test_size=0.25,
    random_state=0
)

del games

print(
    'Took {:.0f} seconds. Using {:,} games for training and {:,} for test'
    .format(time() - t, x_train.shape[0], x_test.shape[0])
)

poly = PolynomialFeatures( 
    3, 
    interaction_only=False, 
    include_bias=False # we include the intercept in the Logistic Regression instead
)
scale = StandardScaler() # scale features to zero mean and unit variance
logreg = LogisticRegression(
    fit_intercept=True,
    penalty='elasticnet',
    random_state=0,
    solver='saga', # only solver that can handle elasticnet, l1, and l2
    multi_class='ovr', # binary one-over-all model fit to each class 
    # n_jobs=3, # put the model for each class on a separate thread
    n_jobs=1, # try less for large model
    verbose=1, # 1 prints a line for each epoch for each job
    max_iter=200
)
pipe = Pipeline(steps=[('poly', poly), ('scale', scale), ('logreg', logreg)])
param_grid = {
    'logreg__C': [0.0001], # greater C means less regularization 
    'logreg__l1_ratio': [0.5] # 1 = l1, 0 = l2
    ## Previous grid searches for models with deg 2 and 3 poly features for both 1+0 and 3+0 data 
    ## with the below parameters all landed on or near the above parameters
    # 'logreg__C': np.logspace(-4, 6, 5), # greater C means less regularization 
    # 'logreg__l1_ratio': np.linspace(0, 1, 5) # 1 = l1, 0 = l2
} 
search = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid,
    # scoring='neg_log_loss', # more penalization the less confident you were in the correct classification
    refit=True, # refit estimator using best found params on all of the data
    cv=4, # stratified cv-fold cv. this will result in cv*(num param combos) fits
    return_train_score=True, # cv_results_ will include training scores 
    verbose=3 
)
search.fit(x_train, y_train)

print(f'\nSaving search results as pickles/{time_control}_wdl_model_search.pkl')
joblib.dump(search, f'pickles/{time_control}_wdl_model_search.pkl')
print('Search took {:.2f} hours in total'.format((time() - t_total)/60/60))

print('\nAverage score over folds for each combination of parameters:')
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, search.cv_results_['params']):
    print('{:.6f} (+/-{:.06f}) for {}'.format(mean, std * 2, params))

print('\nBest combination of parameters found:')
print(search.best_params_)

print('\nCoefficients for best model:')
best = search.best_estimator_
feature_names = best.named_steps['poly'].get_feature_names_out()
feature_names = np.concatenate((['intercept'], feature_names))
coefs = best.named_steps['logreg'].coef_
intercept = best.named_steps['logreg'].intercept_
coefs = np.concatenate((np.atleast_2d(intercept).T, coefs), axis=1)
coefs = pd.DataFrame({
    'feature': feature_names,
    '0': coefs[0],
    '1': coefs[1],
    '2': coefs[2],
})
print(coefs)

print('\nClassification report for best model on training data:')
y_pred = search.predict(x_train)
print(classification_report(y_train, y_pred, digits=6, zero_division=0))

print('\nConfusion matrix for best model on training data (rows=true, cols=pred; normalized to true):')
confusion = confusion_matrix(y_train, y_pred, normalize='true')
confusion = pd.DataFrame(data=confusion)
print(confusion)

print('\nClassification report for best model on test data:')
y_pred = search.predict(x_test)
print(classification_report(y_test, y_pred, digits=6, zero_division=0))

print('\nConfusion matrix for best model on test data (rows=true, cols=pred; normalized to true):')
confusion = confusion_matrix(y_test, y_pred, normalize='true')
confusion = pd.DataFrame(data=confusion)
print(confusion)

