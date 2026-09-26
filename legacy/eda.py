#%%
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/Users/user4574/projects/diomedes/')
os.getcwd()

from src.model import DiomedesModel, fitDiomedesModel

tfd = tfp.distributions
tfb = tfp.bijectors

#%%
data = pd.read_csv('data/processed.csv')

num_drivers = data['dd'].nunique()
num_cy = data['cyd'].nunique()
num_observations = data.shape[0]

#%%


def get_value(dataframe, key, dtype):
    return dataframe[key].values.astype(dtype)


features_train = {
    k: get_value(data, key=k, dtype=np.int32)
    for k in ['dd', 'cyd']}
labels_train = get_value(data, key='z_score', dtype=np.float32)



#%%
dio_jointdist = DiomedesModel()
dio_train = dio_jointdist(features_train)

#%%
fit_res = fitDiomedesModel(dio_train, labels_train)


# %%
plt.plot(fit_res[3])
plt.ylabel(r'Loss $-\log$ $p(y\mid\mathbf{x})$')
plt.xlabel('Iteration')
plt.show()

# %%
[
    effect_driver_mean,
    effect_cy_mean,
] = [
    np.mean(x, axis=0).astype(np.float32) for x in [
        fit_res[0],
        fit_res[1],
    ]
]

# Get the pseudo-posterior predictive distribution
(*posterior_conditionals, ratings_posterior), _ = dio_train.sample_distributions(
    value=(
        effect_driver_mean,
        effect_cy_mean,
    ))

ratings_prediction = ratings_posterior.mean()


# %%
plt.title("Histogram of Driver Effects")
plt.hist(effect_driver_mean, 75)
plt.show()


# %%
plt.title("Histogram of Constructor Year Effects")
plt.hist(effect_cy_mean, 75)
plt.show()


# %%
driver_inf = pd.DataFrame(effect_driver_mean).reset_index()
driver_inf.columns = ['dd', 'driver_effect']
data = data.merge(driver_inf, on='dd')

# %%
cy_inf = pd.DataFrame(effect_cy_mean).reset_index()
cy_inf.columns = ['cyd', 'cy_effect']
data = data.merge(cy_inf, on='cyd')


# %%
drivers = pd.read_csv('data/drivers.csv')
result = data.merge(
    drivers[['driverId', 'forename', 'surname']], on='driverId')

# %%
result = result[['forename', 'surname', 'year', 'name', 'driver_effect', 'cy_effect']].groupby(
    ['forename', 'surname', 'year']).first().reset_index().sort_values(['driver_effect', 'cy_effect'])
result.to_csv('data/output.csv')

#%%
effects = result[['year', 'driver_effect', 'cy_effect']].drop_duplicates()

g = (sns.jointplot(effects['driver_effect'],
                   effects['cy_effect'], kind='hex').set_axis_labels('Driver Effect (SD)', 'Constructor Year Effect (SD)'))
# %%
