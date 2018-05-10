import pymc3 as pm
from numpy import ones, array

# Samples for each dose level
n = 5 * ones(4, dtype=int)
# Log-dose
dose = array([-.86, -.3, -.05, .73])

with pm.Model() as model:

    # Logit-linear model parameters
    alpha = pm.Normal('alpha', 0, sd=100.)
    beta = pm.Normal('beta', 0, sd=1.)

    # Calculate probabilities of death
    theta = pm.Deterministic('theta', pm.math.invlogit(alpha + beta * dose))

    # Data likelihood
    deaths = pm.Binomial('deaths', n=n, p=theta, observed=[0, 1, 3, 5])


def run(n=1000):
    if n == "short":
        n = 50
    with model:
        pm.sample(n, tune=1000)

if __name__ == '__main__':
    run()
