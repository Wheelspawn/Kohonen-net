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
                 tau_sigma=2500, tau_l=2500): # [row, col]
        self.l = l
        self.inputs = inputs
        self.sigma=sigma
        self.t_step = t_step
        self.learn_c=learn_c
        self.tau_sigma=tau_sigma
        self.tau_l=tau_l
        self.initWeights()
        
    def initWeights(self):
        self.l = np.random.uniform(-1,1,[self.l[0],
                                         self.l[1],
                                         self.inputs[0],
                                         self.inputs[1]]).astype(np.float16)
    
    def feed(self,input_mat):
        # computing the Euclidean distance of the neurons
        # with respect to their inputs
        c = self.compete(input_mat)
        # indices of the winning neuron
        i,j=np.where(c==np.min(c))

        # compute neighborhood effects
        self.compute_neighbors((i,j))
        
        # timestep
        self.t_step += 1
    
    def discrim(self, input_mat, weights): # discrimination function
        d_j = 0
        for e in input_mat:
            for w in weights:
                print(e)
                print(w)
        
                d_j += sum((e-w)**2)
                
        print(d_j)
        return d_j
    
    def compete(self, input_mat):
        discrim_mat = np.zeros((len(self.l),len(self.l[0])))
        
        for i in range(len(self.l)):
            for j in range(len(self.l[i])):
                discrim_mat[i][j] = self.discrim(input_mat, self.l[i][j])
                
        return discrim_mat
    
    def update():
        pass
    
    def compute_neighbors(self,winner_i):
        for k in range(len(self.l)):
            for l in range(len(self.l[k])):
                if (k,l) != winner_i:
                    self.neighbors( (k,l), winner_i )
    
    def neighbors(self,n,winner): # topological neighborhood
        # lateral distance
        s = np.sqrt( (n[0]-winner[0])**2 - (n[1]-winner[1])**2 )
        t = np.exp( (-s**2 )/( 2*self.sigma**2) )
        
        pass
    def sigma_decay(self):
        self.sigma = self.sigma*np.exp(-self.t_step/self.tau_sigma)
    
    def index():
        pass
    
    def distance(self,a,b): # distance between two vectors
        return np.sqrt( ( a[0]-b[0] )**2 + ( a[1]-b[1] )**2 )
        pass
    
    def reset(self):
        self.t_step=0

t_0 = [[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]
t_1 = [[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]]
t_2 = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,0]]
t_3 = [[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]]
