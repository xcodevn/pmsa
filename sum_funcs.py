import math
import numpy as np
import matplotlib.pyplot as plt
import re

txt = open("grace1000.dat").readlines()
data = []
for l in txt[1:]:
    row =   re.split('[ ]+', l)
    data.append( (float(row[2]), float(row[3]) ) )

# print data

# exit()

def fun (l, p, ti, di = 1.0):
    return di * np.log (p * l**p * ti**(p-1)) - (l * ti)**p

def sfun (l, p, data):
    return np.sum( [ fun(l, p, ti, di) for ti, di in data ] )

def grad(l, p, ti, di = 1.0):
    dl = -ti**p * p * l**(p-1) + di * p / l
    dp = -(l * ti)**p * np.log(l * ti) + di * (1.0/p + np.log(l) + np.log(ti))
    return np.asarray ( [dl, dp] )

def sgrad (l, p, data):
    return np.sum( [ grad(l, p, ti, di) for ti, di in data ], axis=0)

p = np.asarray( [0.00539251, 0.15416494] )
c = 1
while True:
    c = c + 1
    # print p
    gr = 0.0000001 * sgrad(p[0], p[1], data)
    print gr
    vl = sfun(p[0], p[1], data)
    p = p + gr
    if c == 10: break
    print p, vl


l = 0.00539251
p = 0.15416494
t = np.zeros( 200 )
print sgrad(0.00539251, 0.15416494, data)
for ti, di in data:
    f = [ fun(l, p, ti, di) for l in np.linspace(0.001, 6, 200) ]
    t = t + f
    plt.plot(np.linspace(0.001, 6, 200), f)
plt.plot(np.linspace(0.001, 6, 200), t)
plt.show()



