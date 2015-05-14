import math
import numpy as np
import matplotlib.pyplot as plt

A, B = 8000, 10000
epsilon = 0.1

def smooth_mechanism(a, b, epsilon):

    delta = 1.0 /B
    print delta
    alpha = epsilon / 2
    beta  = epsilon / 2 / math.log(1/delta)
    count = 1.0
    value = a*1.0/b

    # current maximum sensitivity
    max_sen = 0
    while b > count:
        l = (a - count)/(b + count)
        r = (a + count)/(b - count)

        if l <= 0 : l = value

        sen = max ( [ value - l, r - value ] )
        t_sen = math.exp ( - count * beta ) * sen
        print "#", sen, t_sen

        if t_sen > max_sen: max_sen = t_sen

        count = count + 1

    print "# smooth_mechanism"
    print max_sen
    rvalue = value + np.random.laplace() * max_sen / alpha
    print math.fabs(rvalue - value)
    return rvalue


def staircase_mechanism(dist, eps):
    l = r = dist[0][1]

    mean = dist[0][1]
    count = 1
    prob_list = []
    while count + 1 < len(dist):
        nl = dist[count][1]
        nr = dist[count+1][1]
        level = dist[count][0]
        d = nr - r + l - nl
        if d < 0: break
        # print level
        prob = math.exp ( - epsilon * level ) * d
        prob_list.append(prob)
        # l, r = nl, nr
        count = count + 2
        if count > 600: break
    prob_list = np.asarray(prob_list)/ sum(prob_list)
    print prob_list
    return prob_list

#
# f(x; \mu, \lambda) = \frac{1}{2\lambda} \exp\left(-\frac{|x - \mu|}{\lambda}\right).
#

def laplace_mechanism(mean, sensitivity, epsilon, size=1):
    scale = sensitivity / epsilon
    n = mean + np.random.laplace(0,  scale, size)
    return n

dist =  smooth_mechanism(A, B, epsilon)

# pr = staircase_mechanism(dist, epsilon)
# cpr = [0]
# for i in pr:
#     cpr.append( cpr[-1] + i)
#     if cpr[-1] > 0.9: break
#
#
# print "# Staircase #"
# print cpr
# print dist [ len(cpr)*2 ][1]
# print dist[0][1]
# print dist [ len(cpr)*2 ][1] - dist[0][1]


# [x, y] = zip(*dist)
# plt.scatter ( y, x )
# plt.show()

print "# Laplace mechanism #"
ra = laplace_mechanism(A, 1, epsilon/2)
rb = laplace_mechanism(B, 1, epsilon/2)
print A, B
print ra, rb, ra/rb, np.abs(A*1.0/B - ra/rb)

#plt.text(0, 0, r'$f(x; \mu, \lambda) = \frac{1}{2\lambda} \exp\left(-\frac{|x - \mu|}{\lambda}\right)$')
#plt.show()

