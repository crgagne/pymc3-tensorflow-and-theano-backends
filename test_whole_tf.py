
import sys
import os
import numpy as np

os.environ["PYMC_SYMB_BACKEND"] = "tensorflow"

sys.path.append('./pymc3-master')
import pymc3 as pm


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

Y=Y.astype('float32')

### Instantiate model
basic_model2 = pm.Model()
with basic_model2:
    alpha = pm.Normal(name='alpha',initial_value=0.0, mu=0.0, sd=10.0) # make sure these are float
    beta = pm.Normal(name='beta',initial_value=np.array([1.0,1.0],dtype='float32'), mu=0.0, sd=10.0, shape=2)
    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    #import pdb; pdb.set_trace()
    Y_obs = pm.Normal(name='Y_obs',initial_value=np.ones_like(Y), mu=mu, sd=1.0, observed=Y)

    step = pm.Metropolis()
    #import pdb; pdb.set_trace()
    trace = pm.sample(5000,step=step,cores=1,chains=1)


print(np.mean(trace['alpha:0']))
print(np.mean(trace['beta:0'],axis=0))
