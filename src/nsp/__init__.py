"""nsp: network spectral predictor.

A network of small linear predictors on a gated graph. One map per body (K) and one per spring (M) are
shared across the network and assembled into a single linear step:

    x_v(t+1) = K x_v(t) - M sum_u g_uv (x_v(t) - x_u(t))   =>   x(t+1) = (I_n (x) K - L(g) (x) M) x(t)

Every module takes d, the number of position coordinates per body (1 for the planar pendulums of step 1,
2 for the spherical pendulums of stage 1.5). The core needs only NumPy and SciPy; `nsp.viz` adds matplotlib.
"""

__version__ = "0.1.0"
