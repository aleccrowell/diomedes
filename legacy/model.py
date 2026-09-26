
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


tfd = tfp.distributions
tfb = tfp.bijectors

class DiomedesModel(tf.Module):
  def __init__(self):
    # Set up fixed effects and other parameters.
    # These are free parameters to be optimized in E-steps
    self._intercept = tf.Variable(
        0., name="intercept")            # alpha in eq
    self._stddev_drivers = tfp.util.TransformedVariable(
        1., bijector=tfb.Exp(), name="stddev_drivers")            # sigma in eq
    self._stddev_cys = tfp.util.TransformedVariable(
        1., bijector=tfb.Exp(), name="stddev_cys")         # sigma in eq

  def __call__(self, features):
    num_drivers = len(np.unique(features['dd']))
    num_cy = len(np.unique(features['cyd']))
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
                loc=(tf.gather(effect_drivers, features["dd"], axis=-1) +
                     tf.gather(effect_cys, features["cyd"], axis=-1) +
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
    model.num_drivers = num_drivers
    model.num_cy = num_cy
    return model


def fitDiomedesModel(model, targets, num_warmup_iters=1000, num_iters=5000):
    # Set up E-step (MCMC).
    @tf.function(autograph=False, experimental_compile=True)
    def one_e_step(current_state, kernel_results):
        next_state, next_kernel_results = hmc.one_step(current_state=current_state, previous_kernel_results=kernel_results)
        return next_state, next_kernel_results
    # Set up M-step (gradient descent).
    @tf.function(autograph=False, experimental_compile=True)
    def one_m_step(current_state):
        with tf.GradientTape() as tape:
            loss = -target_log_prob_fn(*current_state)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss
    num_accepted = 0
    effect_driver_samples = np.zeros([num_iters, model.num_drivers])
    effect_cy_samples = np.zeros([num_iters, model.num_cy])
    loss_history = np.zeros([num_iters])
    optimizer = tf.optimizers.Adam(learning_rate=.01)
    target_log_prob_fn = lambda *x: model.log_prob(x + (targets,))
    trainable_variables = model.trainable_variables
    current_state = model.sample()[:-1]
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.015,
        num_leapfrog_steps=3)
    kernel_results = hmc.bootstrap_results(current_state)
    # Run warm-up stage.
    for t in range(num_warmup_iters):
        current_state, kernel_results = one_e_step(current_state, kernel_results)
        num_accepted += kernel_results.is_accepted.numpy()
        if t % 500 == 0 or t == num_warmup_iters - 1:
            print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(
                t, num_accepted / (t + 1)))
    num_accepted = 0  # reset acceptance rate counter
    # Run training.
    for t in range(num_iters):
        # run 5 MCMC iterations before every joint EM update
        for _ in range(5):
            current_state, kernel_results = one_e_step(current_state, kernel_results)
        loss = one_m_step(current_state)
        effect_driver_samples[t, :] = current_state[0].numpy()
        effect_cy_samples[t, :] = current_state[1].numpy()
        num_accepted += kernel_results.is_accepted.numpy()
        loss_history[t] = loss.numpy()
        if t % 500 == 0 or t == num_iters - 1:
            print("Iteration: {:>4} Acceptance Rate: {:.3f} Loss: {:.3f}".format(
                t, num_accepted / (t + 1), loss_history[t]))
    return effect_driver_samples, effect_cy_samples, current_state, loss_history
 
