#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:16:50 2018

@author: nsage
"""

import numpy as np

class SOM(object):
    
    def __init__(self, l=[2,2], inputs=[4,4],
                 sigma=1, t_step=0, learn_c=0.02,
                 tau_sigma=2500, tau_c=1500): # [row, col]
        self.l = l
        self.inputs = inputs
        self.sigma=sigma
        self.t_step = t_step
        self.learn_c=learn_c
        self.tau_sigma=tau_sigma
        self.tau_c=tau_c
        self.initWeights()
        
    def initWeights(self):
        self.l = np.random.uniform(-1,1,[self.l[0],
                                         self.l[1],
                                         self.inputs[0],
                                         self.inputs[1]]).astype(np.float16)
    
    def out(self,input_mat):
        m = np.shape(self.l)[0]
        n = np.shape(self.l)[1]
        o = np.zeros( (m,n) )
        
        for i in range(m):
            for j in range(n):
                o[i][j] = self.sigmoid( np.sum(input_mat*self.l[i][j]) )
        return o
        
    def sigmoid(self, x):
        return 1/( 1+np.exp(-x) )
    
    def learn(self,input_mat):
        # computing the Euclidean distance of the neurons
        # with respect to their inputs
        c = self.compete(input_mat)
        # indices of the winning neuron
        i,j=np.where(c==np.min(c))

        # compute neighborhood effects and update weights
        self.update((i,j), input_mat)
        
        # timestep
        self.t_step += 1
        self.learn_decay()
        self.sigma_decay()
    
    def discrim(self, input_mat, weights): # discrimination function
        d_j = 0
        for e in input_mat:
            for w in weights:
        
                d_j += sum((e-w)**2)
                
        return d_j
    
    def compete(self, input_mat):
        discrim_mat = np.zeros((len(self.l),len(self.l[0])))
        
        for i in range(len(self.l)):
            for j in range(len(self.l[i])):
                discrim_mat[i][j] = self.discrim(input_mat, self.l[i][j])
                
        return discrim_mat
    
    def update(self,winner_i, input_mat):
        n_mat = np.zeros((len(self.l),len(self.l[0])))
        
        for k in range(len(self.l)):
            for l in range(len(self.l[k])):
                print("k,l: ", (k,l))
                print("w: ", winner_i)
                print("mat: ", n_mat[k][l])
                print("n: ", self.neighbors( (k,l), winner_i ))
                print("")
                n_mat[k][l] = self.neighbors( (k,l), winner_i )
        
        for m in range(len(self.l)):
            for n in range(len(self.l[m])):
                for o in range(len(self.l[m][n])):
                    self.l[m][n][o] = self.learn_c*n_mat[m][n]*np.sum(input_mat-self.l[m][n])
                    
    def learn_decay(self):
        self.learn_c = self.learn_c*np.exp(-self.t_step/self.tau_c)
        
    def neighbors(self,n,winner): # topological neighborhood
        # lateral distance
        s = self.distance(n, winner)
        t = np.exp( (-s**2 )/( 2*self.sigma**2) )
        return t
        
    def sigma_decay(self):
        self.sigma = self.sigma*np.exp(-self.t_step/self.tau_sigma)
    
    def distance(self,a,b): # euclidean distance between two vectors
        return np.sqrt( ( a[0]-b[0] )**2 + ( a[1]-b[1] )**2 )
    
    def reset(self): # resets params after a learning cycle
        self.t_step=0
        self.sigma=1
        
        
def test():
    s=SOM([2,2],[4,4])
    
    t_0 = [[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]
    t_1 = [[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]]
    t_2 = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,0]]
    t_3 = [[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]]
    
    for i in range(0,100):
        for j in range(0,15):
            s.learn(t_0)
        for k in range(0,15):
            s.learn(t_1)
        for l in range(0,15):
            s.learn(t_2)
        for m in range(0,15):
            s.learn(t_3)
            
    print(s.out(t_0))
    print(s.out(t_1))
    print(s.out(t_2))
    print(s.out(t_3))
    
    
    return s