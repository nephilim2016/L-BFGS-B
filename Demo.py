#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:51:16 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot
from Optimizer import para,options,L_BFGS_B

def func(x):
    f=100*(x[0]**2-x[1])**2+(x[0]-1)**2
    g=np.array([400*x[0]*(x[0]**2-x[1])+2*(x[0]-1),-200*(x[0]**2-x[1])])
    return f,g

if __name__=='__main__':
    fun1=lambda x,y: 100*(x**2-y)**2+(x-1)**2
    x0=np.array([5,5])
    fig=pyplot.figure()  
    ax3=pyplot.axes(projection='3d')
    xx=np.arange(-5,5.1,0.1)
    yy=np.arange(-5,5.1,0.1)
    X,Y=np.meshgrid(xx,yy)
    Z=fun1(X,Y)
    ax3.plot_surface(X,Y,Z,cmap='rainbow',alpha=0.2)
    low=np.array([-5,-5])
    up=np.array([2,2])
    
    optimizer=L_BFGS_B(func,x0,low,up)
    info=optimizer.l_bfgs_b()
    # x=OptimizerGrad(fun1,gfun,x0)
    
    # fun=lambda x: 100*(x[0]**2-x[1])**2+(x[0]-1)**2
    # gfun=lambda x: np.array([400*x[0]*(x[0]**2-x[1]),-200*(x[0]**2-x[1])])
    # x=OptimizerAdadelta(gfun,x0)
    
    x_idx=[]
    y_idx=[]
    z_idx=[]
    x=info['x_history']
    for idx in range(len(x)):
        x_idx.append(x[idx][0])
        y_idx.append(x[idx][1])
        z_idx.append(fun1(x[idx][0],x[idx][1]))
    ax3.plot(x_idx,y_idx,z_idx,'b-o')
