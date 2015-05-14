# IMPORT LIBRARY
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy
from joblib import Parallel, delayed
import multiprocessing



# GLOBAL SETTING
n = 10000
p = 0.5
l = 0.2
k = n/4        # number of subsets
diameter = 2
epsilon = 0.01

# DATA SAMPLING
def wb_sample(n, p, l):
    return l * np.random.weibull(p, n)


# WEIBULL LINEAR LEAST SQUARE ESTIMATOR
# Y = AX + B + err
# RETURN (A, B)
def wb_llse(data, bound_A = 5, bound_B = (-10, 10) ):
    n = len(data)
    sdata = sorted(data)
    X = []
    Y = []
    for i in range(1, len(sdata) + 1):
        # http://www.wuj.pl/UserFiles/File/Schedae%2020%20N/Schedae%20Informaticae_20_3p.pdf
        Fi = (i) * 1.0 / (n + 1)                      # adjusted mean rank
        # we, currently got bias problem here!!!, SOLVE IT PLEASE

        Y.append( np.log ( - np.log ( 1 - Fi ) ) )
        X.append( np.log(sdata[i - 1]) )
    A = np.vstack([X, np.ones(len(X))]).T
    slop, intercept = np.linalg.lstsq(A, Y)[0]
    if slop > 5:
        slop = bound_A
    return (slop, intercept)

def wb_mle(data):
    # from paper: http://www.math.uiuc.edu/~pjohnson/Johnson_etal_BMLE_Paper.pdf
    pp = stats.exponweib.fit(data, floc = 0, f0=1)
    b1,b2 =  wb_mle_bias(len(data), pp[1], pp[3])
    return ( pp [1] - b1 , pp[3] - b2 )

def wb_mle_bias(n, p1, p2):
    # return [0,0]
    z3 = scipy.special.zetac(3) + 1
    b1 = 18.0 * p1 * (np.pi**2 -  2* (z3)) / n / np.pi**4
    gam = 0.577215664902 # np.euler_gamma
    d2 = np.pi**4 * ( -1 + 2* p1) - 6 * np.pi**2 * ( 1 + gam**2 + 5*p1 - 2*gam*(1 + 2*p1)) - 72 * ( -1 + gam)* p1 * (z3)
    b2 = -1.0 * ( p2 * d2  ) / (2 * n * np.pi**4 * p1**2)
    return [b1, b2]



# ADDING LAPLACE NOISE FOR GIVING DIFFERENTIAL PRIVACY
def Laplace_mechanism(value, magnitude):
    print "errr", magnitude
    d = len(value)
    r = np.random.randn(d)
    r = r / np.linalg.norm(r)

    rnd = r * np.random.gamma(d, magnitude)
    return value + rnd

data =  wb_sample(n, p, l)


pp = stats.exponweib.fit(data, floc = 0, f0=1)
pp = [  pp [1], pp[3] ]
print "true params", pp

# true_param =  wb_llse(data)
# print true_param

subsets = np.array_split(data, k)
params = []
c = 0


def compute_mle(data):
    [p, l] = wb_mle(data)
    if p > 1: p = 1.0
    if l > 1: l = 1.0
    if p < 0: p = 0.0
    if l < 0: l = 0.0
    return [p, l]

num_copres = multiprocessing.cpu_count()

results = Parallel (n_jobs=num_copres)(delayed(compute_mle)(subsets[i]) for i in range(k))


# for s in subsets:
#     c = c+ 1
#
#     pr = wb_mle( s )
#     params.append ( pr )
#     print c, np.average( params, axis = 0)

avg_param = np.average( results, axis = 0)
para = Laplace_mechanism(avg_param, diameter*1.0 / k / epsilon)

print para, np.linalg.norm ( pp - para )
