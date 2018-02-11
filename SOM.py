#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:16:50 2018

@author: nsage
"""

import numpy as np

class SOM(object):
    
    def __init__(self, l=[3,5],inputs=3):
        self.l = l
        self.inputs = inputs
        self.initWeights()
        
    def initWeights(self):
        self.l = np.random.uniform(-1,1,[self.l[0],self.l[1],self.inputs])
        
    def discrim(self, vect_input, weights): # discrimination function
        d_j = 0
        for i in range(len(vect_input)):
            d_j += (vect_input[i]-weights[i])**2
            
        return d_j
    
    def update():
        pass
    
    def neighbor(i,j): # topological neighborhood
        pass
    
    def index():
        pass
    
    def distance():
        pass