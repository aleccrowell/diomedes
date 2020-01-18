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
keep_list = ['raceId', 'driverId', 'constructorId','grid','milliseconds','year','round','circuitId']
results = results[[i for i in results.columns if i in keep_list]]
races = races[[i for i in races.columns if i in keep_list]]
proc = results.merge(races,on='raceId')
proc.replace('\\N',np.nan,inplace=True)
proc['milliseconds'] = proc[['raceId', 'milliseconds']].groupby('raceId').milliseconds.transform(lambda x: preprocessing.scale(x.astype(float)))

proc.to_csv('processed.csv')
# %%
