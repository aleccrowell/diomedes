#%%
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

#%%
data = pd.read_csv('data/processed.csv')

num_drivers = data['driverId'].nunique()
num_cy = data['cyd'].nunique()
num_observations = data.shape[0]

#%%
def get_value(
    dataframe, key, dtype): return dataframe[key].values.astype(dtype)


features_train = {
    k: get_value(data, key=k, dtype=np.int32)
    for k in ['driverId', 'cyd']}
labels_train = get_value(data, key='z_score', dtype=np.float32)

#%%
class LinearMixedEffectModel(tf.Module):
  def __init__(self):
    # Set up fixed effects and other parameters.
    # These are free parameters to be optimized in E-steps
    self._intercept = tf.Variable(
        0., name="intercept")            # alpha in eq
    self._stddev_drivers = tfp.util.TransformedVariable(
        1., bijector=tfb.Exp(), name="stddev_drivers")            # sigma in eq
    self._stddev_instructors = tfp.util.TransformedVariable(
        1., bijector=tfb.Exp(), name="stddev_cys")         # sigma in eq

  def __call__(self, features):
    model = tfd.JointDistributionSequential([
        # Set up random effects.
        tfd.MultivariateNormalDiag(
            loc=tf.zeros(num_drivers),
            scale_identity_multiplier=self._stddev_drivers),
        tfd.MultivariateNormalDiag(
            loc=tf.zeros(num_cy),
            scale_identity_multiplier=self._stddev_cys),
        # This is the likelihood for the observed.
        lambda effect_cys, effect_drivers: tfd.Independent(
            tfd.Normal(
                loc=(tf.gather(effect_drivers, features["drivers"], axis=-1) +
                     tf.gather(effect_cys, features["cys"], axis=-1) +
                     self._intercept),
                scale=1.),
            reinterpreted_batch_ndims=1)
    ])

    # To enable tracking of the trainable variables via the created distribution,
    # we attach a reference to `self`. Since all TFP objects sub-class
    # `tf.Module`, this means that the following is possible:
    # LinearMixedEffectModel()(features_train).trainable_variables
    # ==> tuple of all tf.Variables created by LinearMixedEffectModel.
    model._to_track = self
    return model


lmm_jointdist = LinearMixedEffectModel()
# Conditioned on feature/predictors from the training data
lmm_train = lmm_jointdist(features_train)


# %%
