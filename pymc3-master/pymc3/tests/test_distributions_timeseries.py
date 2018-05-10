from __future__ import division

from ..model import Model
from ..distributions.continuous import Flat, Normal
from ..distributions.timeseries import EulerMaruyama, AR, GARCH11
from ..sampling import sample, sample_ppc
from ..theanof import floatX

import numpy as np

def test_AR():
    # AR1
    data = np.array([0.3,1,2,3,4])
    phi = np.array([0.99])
    with Model() as t:
        y = AR('y', phi, sd=1, shape=len(data))
        z = Normal('z', mu=phi*data[:-1], sd=1, shape=len(data)-1)
    ar_like = t['y'].logp({'z':data[1:], 'y': data})
    reg_like = t['z'].logp({'z':data[1:], 'y': data})
    np.testing.assert_allclose(ar_like, reg_like)

    # AR1 + constant
    with Model() as t:
        y = AR('y', [0.3, phi], sd=1, shape=len(data), constant=True)
        z = Normal('z', mu=0.3 + phi*data[:-1], sd=1, shape=len(data)-1)
    ar_like = t['y'].logp({'z':data[1:], 'y': data})
    reg_like = t['z'].logp({'z':data[1:], 'y': data})
    np.testing.assert_allclose(ar_like, reg_like)

    # AR2
    phi = np.array([0.84, 0.10])
    with Model() as t:
        y = AR('y', phi, sd=1, shape=len(data))
        z = Normal('z', mu=phi[0]*data[1:-1]+phi[1]*data[:-2], sd=1, shape=len(data)-2)
    ar_like = t['y'].logp({'z':data[2:], 'y': data})
    reg_like = t['z'].logp({'z':data[2:], 'y': data})
    np.testing.assert_allclose(ar_like, reg_like)



def test_GARCH11():
    # test data ~ N(0, 1)
    data = np.array([-1.35078362, -0.81254164,  0.28918551, -2.87043544, -0.94353337,
                     0.83660719, -0.23336562, -0.58586298, -1.36856736, -1.60832975,
                     -1.31403141,  0.05446936, -0.97213128, -0.18928725,  1.62011258,
                     -0.95978616, -2.06536047,  0.6556103 , -0.27816645, -1.26413397])
    omega = 0.6
    alpha_1 = 0.4
    beta_1 = 0.5
    initial_vol = np.float64(0.9)
    vol = np.empty_like(data)
    vol[0] = initial_vol
    for i in range(len(data)-1):
        vol[i+1] = np.sqrt(omega + beta_1*vol[i]**2 + alpha_1*data[i]**2)

    with Model() as t:
        y = GARCH11('y', omega=omega, alpha_1=alpha_1, beta_1=beta_1,
                    initial_vol=initial_vol, shape=data.shape)
        z = Normal('z', mu=0, sd=vol, shape=data.shape)
    garch_like = t['y'].logp({'z':data, 'y': data})
    reg_like = t['z'].logp({'z':data, 'y': data})
    np.testing.assert_allclose(garch_like, reg_like)



def _gen_sde_path(sde, pars, dt, n, x0):
    xs = [x0]
    wt = np.random.normal(size=(n,) if isinstance(x0, float) else (n, x0.size))
    for i in range(n):
        f, g = sde(xs[-1], *pars)
        xs.append(
            xs[-1] + f * dt + np.sqrt(dt) * g * wt[i]
        )
    return np.array(xs)


def test_linear():
    lam = -0.78
    sig2 = 5e-3
    N = 300
    dt = 1e-1
    sde = lambda x, lam: (lam * x, sig2)
    x = floatX(_gen_sde_path(sde, (lam,), dt, N, 5.0))
    z = x + np.random.randn(x.size) * sig2
    # build model
    with Model() as model:
        lamh = Flat('lamh')
        xh = EulerMaruyama('xh', dt, sde, (lamh,), shape=N + 1, testval=x)
        Normal('zh', mu=xh, sd=sig2, observed=z)
    # invert
    with model:
        trace = sample(init='advi+adapt_diag', chains=1)

    ppc = sample_ppc(trace, model=model)
    # test
    p95 = [2.5, 97.5]
    lo, hi = np.percentile(trace[lamh], p95, axis=0)
    assert (lo < lam) and (lam < hi)
    lo, hi = np.percentile(ppc['zh'], p95, axis=0)
    assert ((lo < z) * (z < hi)).mean() > 0.95
