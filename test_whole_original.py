
import sys
sys.path.append('/Users/chris/anaconda/envs/env_pymc3/lib/python3.6/')
sys.path.append('/Users/chris/anaconda/envs/env_pymc3/lib/python3.6/site-packages/')
import pymc3 as pm

import numpy as np

## Generate Data
# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

### Instantiate model
basic_model2 = pm.Model()
with basic_model2:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=1.0, observed=Y)

    step = pm.Metropolis()
    trace = pm.sample(5000,step=step)

print(np.mean(trace['alpha']))
print(np.mean(trace['beta']))
