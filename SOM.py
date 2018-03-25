#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:16:50 2018

@author: nsage
"""

import numpy as np
import random

class SOM(object):
    
    def __init__(self, inputs=[4,4], l=[2,2],
                 sigma=1, t_step=0, learn_c=0.04,
                 tau_sigma=10**7, tau_c=10**7): # [row, col]
        self.l = l
        self.inputs = inputs
        self.sigma=sigma
        self.t_step = t_step
        self.learn_c=learn_c
        self.tau_sigma=tau_sigma
        self.tau_c=tau_c
        self.initWeights()
        
    def initWeights(self): # initialize weight tensor
        self.l = np.random.uniform(-1,1,[self.l[0],
                                         self.l[1],
                                         self.inputs[0],
                                         self.inputs[1]]).astype(np.float16)
    
    def out(self,input_mat): # feedforward
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
        
        # print("i: ", np.round(input_mat,2))
        # print("")
        # print(input_mat)
        c = self.compete(input_mat)
        # print("w: ", np.round(c,2))
        
        # indices of the winning neuron
        # print(c)
        i,j=np.where(c==np.min(c))

        # compute neighborhood effects and update weights
        self.update((i,j), input_mat)
        
        # timestep
        self.t_step += 1
        self.learn_decay()
        self.sigma_decay()
        
        # print(self.t_step)
        # print(self.learn_c)
        # print(self.sigma)
    
    def discrim(self, input_mat, weights): # discrimination function
        d_j = 0
        for e in input_mat:
            for w in weights:
        
                d_j += sum((e-w)**2)
                
        return d_j
    
    def compete(self, input_mat): # neurons compete to see who has the highest activation
        discrim_mat = np.zeros((len(self.l),len(self.l[0])))
        
        for i in range(len(self.l)):
            for j in range(len(self.l[i])):
                discrim_mat[i][j] = self.discrim(input_mat, self.l[i][j])
                
        return discrim_mat
    
    def update(self, winner_i, input_mat): # update process
        n_mat = np.zeros((len(self.l),len(self.l[0])))
        
        for k in range(len(self.l)):
            for l in range(len(self.l[k])):
                
                '''
                print("k,l: ", (k,l))
                print("w: ", winner_i)
                print("mat: ", n_mat[k][l])
                print("n: ", self.neighbors( (k,l), winner_i ))
                print("")
                '''
                
                # print(winner_i)
                
                n_mat[k][l] = self.neighbors( (k,l), winner_i )
        
        # print("n: ")
        # print(np.round(n_mat,2))
        
        # print("")
        
        # print(n_mat)
        
        for m in range(len(self.l)):
            for n in range(len(self.l[m])):
                for o in range(len(self.l[m][n])):
                    for p in range(len(self.l[m][n][0])):
                        self.l[m][n][o][p] += self.learn_c*n_mat[m][n]*(input_mat[m][n]-self.l[m][n][o][p])
                    
    def neighbors(self,n,winner): # topological neighborhood
        # lateral distance
        s = self.distance(n, winner)
        t = np.exp( (-s**2 )/( 2*self.sigma**2) )
        # print(t)
        return t
    
    def learn_decay(self): # decay the learning constant
        if self.learn_c > 10**-10:
            self.learn_c = self.learn_c*np.exp(-self.t_step/self.tau_c)
        else:
            pass
        
    def sigma_decay(self): # decay the sigma constant
        
            if self.sigma > 10**-10:
                self.sigma = self.sigma*np.exp(-self.t_step/self.tau_sigma)
            else:
                pass
    
    def distance(self,a,b): # euclidean distance between two vectors
        return np.sqrt( ( a[0]-b[0] )**2 + ( a[1]-b[1] )**2 )
    
    def reset(self, s=1,l=0.04): # resets params after learning
        self.t_step=0
        self.sigma=s
        self.learn_c = l
        
def test():
    s=SOM([9,9],[3,3])
    
    '''
    
    t_0 = np.array([[1,1,0,0],
                    [1,1,0,0],
                    [0,0,0,0],
                    [0,0,0,0]])
    
    t_1 = np.array([[0,0,1,1],
                    [0,0,1,1],
                    [0,0,0,0],
                    [0,0,0,0]])
    
    t_2 = np.array([[0,0,0,0],
                    [0,0,0,0],
                    [1,1,0,0],
                    [1,1,0,0]])
    
    t_3 = np.array([[0,0,0,0],
                    [0,0,0,0],
                    [0,0,1,1],
                    [0,0,1,1]])
    '''
    
    t_0 = np.array([[1,1,1,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0]])
    
    t_1 = np.array([[0,0,0,1,1,1,0,0,0],
           [0,0,0,1,1,1,0,0,0],
           [0,0,0,1,1,1,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0]])
    
    t_2 = np.array([[0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0]])
    
    t_3 = np.array([[0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0]])
    
    t_4 = np.array([[0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,1,1,1,0,0,0],
           [0,0,0,1,1,1,0,0,0],
           [0,0,0,1,1,1,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0]])
    
    t_5 = np.array([[0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0]])
    
    t_6 = np.array([[0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0],
           [1,1,1,0,0,0,0,0,0]])
    
    t_7 = np.array([[0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,1,1,1,0,0,0],
           [0,0,0,1,1,1,0,0,0],
           [0,0,0,1,1,1,0,0,0]])
    
    t_8 = np.array([[0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,1,1,1],
           [0,0,0,0,0,0,1,1,1]])
    
    t_all = [t_0,t_1,t_2,t_3,t_4,t_5,t_6,t_7,t_8]
    
    # t_all = [t_0,t_1,t_2,t_3]
    
    for i in range(0,250):
        r = random.choice(t_all)
        for j in range(0,5):
            s.learn(r)
            
    for i in range(0,3):
        for j in range(0,3):
            print(s.discrim(t_all[i+j], s.l[i][j]))
            print("")
            
    return s