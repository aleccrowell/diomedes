#%%
import pandas as pd
import os
from sklearn import preprocessing
import numpy as np

os.chdir('/Users/user4574/projects/diomedes/')
os.getcwd()

"""
Need to uniquely identify:
course
year
constructor
driver
result
laps(?)
completed(?)

So need to link raceID to year and course
"""

# %%
results = pd.read_csv('data/results.csv')
races = pd.read_csv('data/races.csv')
drivers = pd.read_csv('data/drivers.csv')
constructors = pd.read_csv('data/constructors.csv')
keep_list = ['raceId', 'driverId', 'constructorId','milliseconds','year']
results = results[[i for i in results.columns if i in keep_list]]
races = races[[i for i in races.columns if i in keep_list]]
proc = results.merge(races,on='raceId')
proc = proc.merge(drivers[['driverId','code']],on='driverId')
proc.replace('\\N',np.nan,inplace=True)
proc['milliseconds'] = proc[['raceId', 'milliseconds']].groupby('raceId').milliseconds.transform(lambda x: preprocessing.scale(x.astype(float)))
proc.rename(columns={'milliseconds':'z_score'},inplace=True)
proc = proc[~proc.z_score.isna()]
proc['constructor_year'] = proc.apply(lambda row: str(int(row.constructorId)) + '_' + str(row.year),axis=1)
proc['cyd'] = preprocessing.OrdinalEncoder().fit_transform(
    proc.constructor_year.values.reshape(-1, 1)).astype(int)
proc.drop('constructor_year',axis=1,inplace=True)
proc['dd'] = preprocessing.OrdinalEncoder().fit_transform(
    proc.driverId.values.reshape(-1, 1)).astype(int)
#Need to make driverId a sequence with max = n_drivers
proc = proc.merge(constructors[['constructorId','name']],on='constructorId')

proc.to_csv('data/processed.csv',index=False)

# %%
results = pd.read_csv('data/results.csv')
races = pd.read_csv('data/races.csv')
drivers = pd.read_csv('data/drivers.csv')
constructors = pd.read_csv('data/constructors.csv')
keep_list = ['raceId', 'driverId', 'constructorId', 'milliseconds', 'year', 'round']
results = results[[i for i in results.columns if i in keep_list]]
races = races[[i for i in races.columns if i in keep_list]]

grid = results.merge(races, on='raceId')
grid = grid.merge(drivers[['driverId', 'code']], on='driverId')
grid.replace('\\N', np.nan, inplace=True)
grid['milliseconds'] = grid[['raceId', 'milliseconds']].groupby(
    'raceId').milliseconds.transform(lambda x: preprocessing.scale(x.astype(float)))
grid.rename(columns={'milliseconds': 'z_score'}, inplace=True)
grid = grid[~grid.z_score.isna()]
grid['dd'] = preprocessing.OrdinalEncoder().fit_transform(
    grid.driverId.values.reshape(-1, 1)).astype(int)
grid['cd'] = preprocessing.OrdinalEncoder().fit_transform(
    grid.constructorId.values.reshape(-1, 1)).astype(int)
grid = grid.merge(grid[['year', 'round']].groupby(['year', 'round']).size().reset_index().reset_index().rename(
    columns={'index': 'race_ind'})[['race_ind', 'year', 'round']], on=['year', 'round'])
grid = grid.sort_values('race_ind')


# %%
result_grid = grid.pivot(index='race_ind',columns='dd',values='z_score').values


# %%
#need to make constructors x drivers matrix
constructor_grid = grid.pivot(index='race_ind', columns='dd', values='cd').values


# %%
constructor_grid = np.zeros((np.max(grid.race_ind)+1,np.max(grid.dd)+1,np.max(grid.cd)+1))
constructor_grid[tuple(grid[['race_ind', 'dd', 'cd']].values.T)] = 1

# %%
