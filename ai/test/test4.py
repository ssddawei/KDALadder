# import numpy as np
# import matplotlib.pyplot as plt

# v = 4
# theta = np.pi
# g = 9.8

# vx = v*np.sin(theta)
# vy = v*np.sin(theta)

# t = 2 * vy/g

# nt = np.linspace(0, t, 10)
# nx = vx*nt
# ny = abs((vy-g*nt)**2 - vy**2) / (2*g)

# plt.plot(nx, ny)
# plt.show()

#refs https://www.jianshu.com/p/80c40a8cf2b5
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# import csv
# #%matplotlib

# with open("test.csv") as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     frames = []
#     ox, oy = [], []
#     list1 = []
#     for row in readCSV:
#         list1.append(row)
#     for i in range(1 , len(list1)):
#         frames += [int(list1[i][0])]
#         ox += [int(float(list1[i][2]))]
#         oy += [int(float(list1[i][3]))]

# ox = np.array([i for i in ox], np.float)
# oy = np.array([1000-i for i in oy], np.float)

# x = np.array([i for i in ox[320:360]], np.float)
# y = np.array([i for i in oy[320:360]], np.float)
# x = x[x!=0]
# y = y[y!=1000]

# # x = np.array([-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0.,
# #          1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.], np.float)
# # y = np.array([0.890928, 0.904723, 0.921421, 0.935007, 0.94281 , 0.949828,
# #        0.966265, 0.975411, 0.978693, 0.97662 , 0.974468, 0.967101,
# #        0.957691, 0.958369, 0.949841, 0.932791, 0.9213  , 0.901874,
# #        0.879374, 0.868257], np.float)

# A = np.zeros((len(x), 3)) #
# A[:,0] = 1
# A[:,1] = x
# A[:,2] = x**3
# # A[:,3] = x**3

# c = np.matmul(A.T, A)
# d = np.matmul(A.T, y)

# _,result = cv.solve(c,d)

# y2 = result[0] + result[1]*x + result[2] * (x**2)

# x2 = np.array([i for i in range(0,1000)])
# y2 = result[0] + result[1]*x2 + result[2] * (x2**3) 


# plt.grid(True)
# plt.xlabel("angle")
# plt.ylabel("matchVal")
# plt.scatter(ox, oy)
# plt.plot(x2,y2)
# # plt.plot( (-result[1]/result[2]/2,-result[1]/result[2]/2), (0.9,1))
# plt.show()
# print("f(x) = {} + ({}*x) + ({}*x^2)".format(result[0],result[1],result[2]))

# https://blog.csdn.net/ouening/article/details/80673288
import matplotlib.pyplot as plt
from numpy import sin, cos, linspace, pi
from scipy.integrate import odeint, solve_ivp
import numpy as np
# import torchdiffeq 
# import torch
plt.close()


def f1(y,t):
    '''
    dy/dt = func(y, t, ...)
    '''
    return sin(t**2)

def f2(t,y):
    return cos(t**2)

def solve_first_order_ode():
    t1 = linspace(-10,10,1000)
    y0 = [10] 
    
    y1 = odeint(f1,y0,t1)
    y2 = odeint(f2,y0,t1,tfirst=True) 
    y3 = solve_ivp(f2,(-10.0,10.0), y0, method='LSODA', t_eval=t1) 
    # y4 = torchdiffeq.odeint(f2,torch.tensor(y0,dtype=torch.float32),torch.tensor(t1))
    # print(y4.shape)
    
    plt.subplot(221)
    plt.plot(t1,y1,'b--',label='scipy odeint')
    plt.legend()
    plt.subplot(222)
    plt.plot(t1,y2,'r-', label='odeint with tfirst')
    plt.legend()
    plt.subplot(223)
    plt.plot(y3.t,y3.y[0],'g', label='solve_ivp')
    plt.legend()
    # plt.subplot(224)
    # plt.plot(t1, y4.numpy(), label='torchdiffeq odeint')
    # plt.legend()
    plt.show()
    
solve_first_order_ode()
