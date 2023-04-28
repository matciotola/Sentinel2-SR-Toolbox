# -*- coding: utf-8 -*-
"""

Created on Thu Aug 31 16:41:51 2017

@author: mass.gargiulo
"""
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
import gdal
import scipy.io as sio
#from scipy.misc import imresize
#import matplotlib.pyplot as plt
from math import fmod
from PIL import Image

class BatchNormLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
            nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).
        
        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(lasagne.init.Constant(1), shape, 'std',
                                  trainable=False, regularizable=False)
        self.beta = self.add_param(lasagne.init.Constant(0), shape, 'beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(lasagne.init.Constant(1), shape, 'gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std
        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)

def batch_norm(layer):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).
    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity)

def load_dataset(dataset_folder, patch_side, border_width,identity,num_bands, city, patch):#,ff): #, output_folder):
    
    #############
    # short names
    path, ps, r,city_name,patch0 = dataset_folder, patch_side, border_width,city, patch
    #output_path = output_folder
    print(r)
    #############
    k = identity
    dir_list = os.listdir(path)
    dir_list.sort()
#    dir_list = dir_list[:-2]
    print(dir_list)
    N = num_bands#-4
    Out = 6 
    import random
    Ts = ps #2*ps when I want to consider full resolution (10-m)
    Ts_2 = ps//2
#    k_L = num
    x_train = np.ndarray(shape=(0, N, Ts_2, Ts_2), dtype='float32')
    y_train = np.ndarray(shape=(0, Out, Ts_2, Ts_2), dtype='float32')
    x_val = np.ndarray(shape=(0, N, Ts_2, Ts_2), dtype='float32')
    y_val = np.ndarray(shape=(0, Out, Ts_2, Ts_2), dtype='float32')
#    K = 33 # 17# #100 #300
    K1 = Ts
    K2 = Ts
    K3 = Ts_2
    K4 = Ts_2
    x_test2 = np.ndarray(shape=(0,N, Ts+2*r, Ts+2*r), dtype='float32')
    y_test2 = np.ndarray(shape=(0,Out-5, Ts+2*r, Ts+2*r), dtype='float32')
    x_test = np.ndarray(shape=(0,N, Ts_2+2*r, Ts_2+2*r), dtype='float32')
    y_test = np.ndarray(shape=(0,Out-5, Ts_2+2*r, Ts_2+2*r), dtype='float32')
    num = 0 
#    for num1 in range(1):#dir_list:
#        file1 = dir_list[num1]
#        if file1.find(city_name) != -1 and file1.find("large") != -1: # and num == k_L and file[2]<str(7):
    vh_file = dir_list[0]
    print(vh_file)
    vh1_file =city_name + '_large_' + str(patch0) + '_B08D'+ vh_file[-4:]
    vh2_file =city_name + '_large_' + str(patch0) + '_B04D'+ vh_file[-4:]
    vh4_file =city_name + '_large_' + str(patch0) + '_B03D'+ vh_file[-4:]
    vh5_file =city_name + '_large_' + str(patch0) + '_B02D'+ vh_file[-4:]
#            b5 b6 b7 b8a b11 b12  

    vh6_file =city_name + '_large_' + str(patch0) + '_B'+k+'DR'+ vh_file[-4:]
    vh6f_file =city_name + '_large_' + str(patch0) + '_B'+k+'DH'+ vh_file[-4:]
    vh6r1_file =city_name + '_large_' + str(patch0) + '_B'+k+'R1'+ vh_file[-4:]
    vh7_file =city_name + '_large_' + str(patch0) + '_B'+k+ vh_file[-4:]

    dataset = gdal.Open(path + vh6r1_file, gdal.GA_ReadOnly)
    b6_r1= dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None

    
    print(vh6_file)
    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
    b8_d= dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
    b4_d = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None           
    
    
    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
    b3_d = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
#            
    dataset = gdal.Open(path + vh5_file, gdal.GA_ReadOnly)      
    b2_d = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
    
    
    dataset = gdal.Open(path + vh7_file, gdal.GA_ReadOnly)
    b6 = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None           
    
    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
    b6_r= dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh6f_file, gdal.GA_ReadOnly)
    b6_hh= dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
    vh1_file =city_name + '_large_' + str(patch0) + '_B02H1'+ vh_file[-4:] # senza H1 per immagine normale
    vh2_file = city_name + '_large_' + str(patch0) + '_B03H1'+ vh_file[-4:]
    vh3_file =city_name + '_large_' + str(patch0) + '_B04H1'+ vh_file[-4:]
    vh4_file =city_name + '_large_' + str(patch0) + '_B08H1'+ vh_file[-4:]
    
    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
    b2_h1 = dataset.ReadAsArray()
#            b2_h1 = b2_h1/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
    b3_h1 = dataset.ReadAsArray()
#            b3_h1 = b3_h1/(2**16)
    dataset = None           
    
    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
    b4_h1 = dataset.ReadAsArray()
#            b4_h1 = b4_h1/(2**16)
    dataset = None         
    
    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
    b8_h1 = dataset.ReadAsArray()
#            b8_h1 = b8_h1/(2**16)
    dataset = None
    vh1_file =city_name + '_large_' + str(patch0) + '_B05H1'+ vh_file[-4:] # senza H1 per immagine normale
    vh2_file = city_name + '_large_' + str(patch0) + '_B06H1'+ vh_file[-4:]
    vh3_file =city_name + '_large_' + str(patch0) + '_B07H1'+ vh_file[-4:]
    vh4_file =city_name + '_large_' + str(patch0) + '_B8AH1'+ vh_file[-4:]
    vh5_file = city_name + '_large_' + str(patch0) + '_B11H1'+ vh_file[-4:]
    vh6_file =city_name + '_large_' + str(patch0) + '_B12H1'+ vh_file[-4:]

    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
    b5_h1 = dataset.ReadAsArray()
#            b9 = b8/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
    b6_h1 = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None           
    
    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
    b7_h1 = dataset.ReadAsArray()
#            b11 = b11/(2**16)
    dataset = None         
    
    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
    b8_a_h1 = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
#            
    dataset = gdal.Open(path+vh5_file, gdal.GA_ReadOnly)
    b11_h1 = dataset.ReadAsArray()
#            b11 = b11/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
    b12_h1 = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
    
    vh1_file =city_name + '_large_' + str(patch0) + '_B05DR'+ vh_file[-4:] # senza H1 per immagine normale
    vh2_file = city_name + '_large_' + str(patch0) + '_B06DR'+ vh_file[-4:]
    vh3_file =city_name + '_large_' + str(patch0) + '_B07DR'+ vh_file[-4:]
    vh4_file =city_name + '_large_' + str(patch0) + '_B8ADR'+ vh_file[-4:]
    vh5_file = city_name + '_large_' + str(patch0) + '_B11DR'+ vh_file[-4:]
    vh6_file =city_name + '_large_' + str(patch0) + '_B12DR'+ vh_file[-4:]

    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
    b5_r = dataset.ReadAsArray()
#            b9 = b8/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
    b6_rr = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None           
    
    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
    b7_r = dataset.ReadAsArray()
#            b11 = b11/(2**16)
    dataset = None         
    
    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
    b8_ar = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
#            
    dataset = gdal.Open(path+vh5_file, gdal.GA_ReadOnly)
    b11_r = dataset.ReadAsArray()
#            b11 = b11/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
    b12_r = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None

    vh1_file =city_name + '_large_' + str(patch0) + '_B05DH'+ vh_file[-4:] # senza H1 per immagine normale
    vh2_file = city_name + '_large_' + str(patch0) + '_B06DH'+ vh_file[-4:]
    vh3_file =city_name + '_large_' + str(patch0) + '_B07DH'+ vh_file[-4:]
    vh4_file =city_name + '_large_' + str(patch0) + '_B8ADH'+ vh_file[-4:]
    vh5_file = city_name + '_large_' + str(patch0) + '_B11DH'+ vh_file[-4:]
    vh6_file =city_name + '_large_' + str(patch0) + '_B12DH'+ vh_file[-4:]

    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
    b5_h = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
    b6_h = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None           
    
    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
    b7_h = dataset.ReadAsArray()
#            b11 = b11/(2**16)
    dataset = None         
    
    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
    b8_a_h = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
#            
    dataset = gdal.Open(path+vh5_file, gdal.GA_ReadOnly)
    b11_h = dataset.ReadAsArray()
#            b11 = b11/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh6_file, gdal.GA_ReadOnly)
    b12_h = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None


    
    vh1_file =city_name + '_large_' + str(patch0) + '_B02DH'+ vh_file[-4:] # senza H1 per immagine normale
    vh2_file = city_name + '_large_' + str(patch0) + '_B03DH'+ vh_file[-4:]
    vh3_file =city_name + '_large_' + str(patch0) + '_B04DH'+ vh_file[-4:]
    vh4_file =city_name + '_large_' + str(patch0) + '_B08DH'+ vh_file[-4:]
    
    dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
    b2_h = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None
    
    dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
    b3_h = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None           
    
    dataset = gdal.Open(path+vh3_file, gdal.GA_ReadOnly)
    b4_h = dataset.ReadAsArray()
#            b11 = b11/(2**16)
    dataset = None         
    
    dataset = gdal.Open(path + vh4_file, gdal.GA_ReadOnly)      
    b8_h = dataset.ReadAsArray()
#            b8 = b8/(2**16)
    dataset = None

    p_test = [] 
    mask = np.zeros(b6_r.shape,dtype='float32')
#            train_patches = np.zeros(b6_r.shape,dtype='float32')
#            val_patches = np.zeros(b6_r.shape,dtype='float32')

#            mask[int(530/2):int(1000/2),int(4300/2):int(4750/2)]=1 
#            for y in range(int(530/2),int(1000/2),2*ps):
#                for x in range(int(4300/2),int(4750/2),2*ps):
#                    p_test.append([y,x])
    print(r)
    b5_h1 = np.pad(b5_h1,((r,r),(r,r)),'reflect') 
    b6_h1 = np.pad(b6_h1,((r,r),(r,r)),'reflect') 
    b7_h1 = np.pad(b7_h1,((r,r),(r,r)),'reflect') 
    b8_a_h1 = np.pad(b8_a_h1,((r,r),(r,r)),'reflect') 
    b11_h1 = np.pad(b11_h1,((r,r),(r,r)),'reflect') 
    b12_h1 = np.pad(b12_h1,((r,r),(r,r)),'reflect') 
    b2_h1 = np.pad(b2_h1,((r,r),(r,r)),'reflect') 
    b3_h1 = np.pad(b3_h1,((r,r),(r,r)),'reflect') 
    b4_h1 = np.pad(b4_h1,((r,r),(r,r)),'reflect') 
    b8_h1 = np.pad(b8_h1,((r,r),(r,r)),'reflect') 
    b6_r1 = np.pad(b6_r1,((r,r),(r,r)),'reflect') 


    b5_h2 = np.pad(b5_h,((r,r),(r,r)),'reflect') 
    b6_h2 = np.pad(b6_h,((r,r),(r,r)),'reflect') 
    b7_h2 = np.pad(b7_h,((r,r),(r,r)),'reflect') 
    b8_a_h2 = np.pad(b8_a_h,((r,r),(r,r)),'reflect') 
    b11_h2 = np.pad(b11_h,((r,r),(r,r)),'reflect') 
    b12_h2 = np.pad(b12_h,((r,r),(r,r)),'reflect') 
    b2_h2 = np.pad(b2_h,((r,r),(r,r)),'reflect') 
    b3_h2 = np.pad(b3_h,((r,r),(r,r)),'reflect') 
    b4_h2 = np.pad(b4_h,((r,r),(r,r)),'reflect') 
    b8_h2 = np.pad(b8_h,((r,r),(r,r)),'reflect') 
    b6_r2 = np.pad(b6_r,((r,r),(r,r)),'reflect') 


    p_test = [0,0]
    
    y0, x0 = p_test
    print((y0,x0))
#            y0 += r
#            x0 += r
    x_test_k = np.ndarray(shape=(len(p_test)-1, N, Ts_2+2*r,Ts_2+2*r), dtype='float32')
    y_test_k = np.ndarray(shape=(len(p_test)-1, Out-5, Ts_2+2*r,Ts_2+2*r), dtype='float32')
    x_test_k2 = np.ndarray(shape=(len(p_test)-1, N, Ts+2*r,Ts+2*r), dtype='float32')
    y_test_k2 = np.ndarray(shape=(len(p_test)-1, Out-5, Ts+2*r,Ts+2*r), dtype='float32')
    n = 0
#            for patch in p_test:
#            y0, x0 = patch[0], patch[1]     
    print(x_test_k2[n,0,:,:].shape)
    print(b5_h1[y0:y0+K1+2*r,x0:x0+K2+2*r].shape)
    x_test_k2[n,0,:,:] = b5_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,1,:,:] = b6_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,2,:,:] = b7_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,3,:,:] = b8_a_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,4,:,:] = b11_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,5,:,:] = b12_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,6,:,:] = b8_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,7,:,:] = b4_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]
    x_test_k2[n,8,:,:] = b3_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]           
    x_test_k2[n,9,:,:] = b2_h1[y0:y0+K1+2*r,x0:x0+K2+2*r]

    x_test_k[n,0,:,:] = b5_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,1,:,:] = b6_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,2,:,:] = b7_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,3,:,:] = b8_a_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,4,:,:] = b11_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,5,:,:] = b12_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,6,:,:] = b8_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,7,:,:] = b4_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    x_test_k[n,8,:,:] = b3_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]           
    x_test_k[n,9,:,:] = b2_h2[y0:y0+K3+2*r,x0:x0+K4+2*r]


    y_test_k[n,0,:,:] = b6_r2[y0:y0+K3+2*r,x0:x0+K4+2*r]
    y_test_k2[n,0,:,:] = b6_r1[y0:y0+K1+2*r,x0:x0+K2+2*r]

    x_test = np.concatenate((x_test, x_test_k))
    y_test = np.concatenate((y_test, y_test_k))
    x_test2 = np.concatenate((x_test2, x_test_k2))
    y_test2 = np.concatenate((y_test2, y_test_k2))
    [s1, s2] = b6_r.shape
    p = []
#            for y in range(1,s1-ps+1,r):
#                for x in range(1,s2-ps+1,r):
#                    mask_d0 = mask[y:y+ps,x:x+ps]
#                    s_0 =  mask_d0.sum()
#                    if s_0 == 0:
#                        p.append([y,x])
    
    p = [0,0]
    y0, x0 = p 
#            y0 += r
#            x0 += r

    p_train = p 
    p_val = p 
    
    x_train_k = np.ndarray(shape=(len(p_train)-1, N, Ts_2, Ts_2), dtype='float32')
    y_train_k = np.ndarray(shape=(len(p_train)-1, Out, Ts_2, Ts_2), dtype='float32')
    n = 0
#            for patch in p_train:
#                y0, x0 = patch[0], patch[1]
    x_train_k[n,0,:,:] = b5_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,1,:,:] = b6_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,2,:,:] = b7_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,3,:,:] = b8_a_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,4,:,:] = b11_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,5,:,:] = b12_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,6,:,:] = b8_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,7,:,:] = b4_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,8,:,:] = b3_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_train_k[n,9,:,:] = b2_h[y0:y0+Ts_2,x0:x0+Ts_2]

#                x_train_k[n,0,:,:] = b5_r[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,1,:,:] = b6_rr[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,2,:,:] = b7_r[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,3,:,:] = b8_ar[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,4,:,:] = b11_r[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,5,:,:] = b12_r[y0:y0+ps,x0:x0+ps]
#
#                x_train_k[n,0,:,:] = b6_r[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,1,:,:] = b8_d[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,2,:,:] = b4_d[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,3,:,:] = b3_d[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,4,:,:] = b2_d[y0:y0+ps,x0:x0+ps]
#
#                x_train_k[n,0,:,:] = b6_hh[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,1,:,:] = b8_h[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,2,:,:] = b4_h[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,3,:,:] = b3_h[y0:y0+ps,x0:x0+ps]
#                x_train_k[n,4,:,:] = b2_h[y0:y0+ps,x0:x0+ps]

        
    y_train_k[n, 0, :, :] = b6[y0:y0+Ts_2,x0:x0+Ts_2]#-b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
    y_train_k[n, 1, :, :] = b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
    y_train_k[n,2,:,:] = b8_d[y0:y0+Ts_2,x0:x0+Ts_2]
    y_train_k[n,3,:,:] = b4_d[y0:y0+Ts_2,x0:x0+Ts_2]
    y_train_k[n,4,:,:] = b3_d[y0:y0+Ts_2,x0:x0+Ts_2]
    y_train_k[n,5,:,:] = b2_d[y0:y0+Ts_2,x0:x0+Ts_2]
#                if (n+6)%6 == 0 and n<=len(p_train)-6:
#                    y_train_k[n, 0, :, :] = b6[y0+r:y0+Ts_2-r, x0+r:x0+ps-r]-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_train_k[n+1, 0, :, :] = b5[y0+r:y0+ps-r, x0+r:x0+ps-r]-b5_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_train_k[n+2, 0, :, :] = b7[y0+r:y0+ps-r, x0+r:x0+ps-r]-b7_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_train_k[n+3, 0, :, :] = b8_a[y0+r:y0+ps-r, x0+r:x0+ps-r]-b8_ar[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_train_k[n+4, 0, :, :] = b11[y0+r:y0+ps-r, x0+r:x0+ps-r]-b11_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_train_k[n+5, 0, :, :] = b12[y0+r:y0+ps-r, x0+r:x0+ps-r]-b12_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                n = n + 1
    x_train = np.concatenate((x_train, x_train_k))
    y_train = np.concatenate((y_train, y_train_k))
    
    x_val_k = np.ndarray(shape=(len(p_val)-1, N, Ts_2, Ts_2), dtype='float32')
    y_val_k = np.ndarray(shape=(len(p_val)-1, Out, Ts_2, Ts_2), dtype='float32')
    n = 0
#            for patch in p_val:
#                y0, x0 = patch[0], patch[1]
    x_val_k[n,0,:,:] = b5_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,1,:,:] = b6_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,2,:,:] = b7_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,3,:,:] = b8_a_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,4,:,:] = b11_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,5,:,:] = b12_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,6,:,:] = b8_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,7,:,:] = b4_h[y0:y0+Ts_2,x0:x0+Ts_2]
    x_val_k[n,8,:,:] = b3_h[y0:y0+Ts_2,x0:x0+Ts_2]                
    x_val_k[n,9,:,:] = b2_h[y0:y0+Ts_2,x0:x0+Ts_2]
    
#                x_val_k[n,0,:,:] = b5_r[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,1,:,:] = b6_rr[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,2,:,:] = b7_r[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,3,:,:] = b8_ar[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,4,:,:] = b11_r[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,5,:,:] = b12_r[y0:y0+ps,x0:x0+ps]

#                x_val_k[n,0,:,:] = b6_r[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,1,:,:] = b8_d[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,2,:,:] = b4_d[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,3,:,:] = b3_d[y0:y0+ps,x0:x0+ps]                
#                x_val_k[n,4,:,:] = b2_d[y0:y0+ps,x0:x0+ps]

#                x_val_k[n,0,:,:] = b6_hh[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,1,:,:] = b8_h[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,2,:,:] = b4_h[y0:y0+ps,x0:x0+ps]
#                x_val_k[n,3,:,:] = b3_h[y0:y0+ps,x0:x0+ps]                
#                x_val_k[n,4,:,:] = b2_h[y0:y0+ps,x0:x0+ps]

    
    y_val_k[n, 0, :, :] = b6[y0:y0+Ts_2,x0:x0+Ts_2]#-b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
    y_val_k[n, 1, :, :] = b6_r[y0:y0+Ts_2,x0:x0+Ts_2]
    y_val_k[n,2,:,:] = b8_d[y0:y0+Ts_2,x0:x0+Ts_2]
    y_val_k[n,3,:,:] = b4_d[y0:y0+Ts_2,x0:x0+Ts_2]
    y_val_k[n,4,:,:] = b3_d[y0:y0+Ts_2,x0:x0+Ts_2]                
    y_val_k[n,5,:,:] = b2_d[y0:y0+Ts_2,x0:x0+Ts_2]
#                if (n+6)%6 == 0 and n<=len(p_val)-6:
#                    y_val_k[n, 0, :, :] = b6[y0+r:y0+ps-r, x0+r:x0+ps-r]-b6_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_val_k[n+1, 0, :, :] = b5[y0+r:y0+ps-r, x0+r:x0+ps-r]-b5_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_val_k[n+2, 0, :, :] = b7[y0+r:y0+ps-r, x0+r:x0+ps-r]-b7_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_val_k[n+3, 0, :, :] = b8_a[y0+r:y0+ps-r, x0+r:x0+ps-r]-b8_ar[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_val_k[n+4, 0, :, :] = b11[y0+r:y0+ps-r, x0+r:x0+ps-r]-b11_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#                    y_val_k[n+5, 0, :, :] = b12[y0+r:y0+ps-r, x0+r:x0+ps-r]-b12_r[y0+r:y0+ps-r, x0+r:x0+ps-r]
#            n = n + 1
    x_val = np.concatenate((x_val, x_val_k))
    y_val = np.concatenate((y_val, y_val_k))
    b11, b8_d, b4_d, b3_d, b2_d, b11_r = None, None, None, None, None, None
    return x_train, y_train, x_val, y_val, x_test, y_test, x_test2, y_test2

def build_cnn(input_var=None,num_bands = None, k_1=None,k_2=None,k_3=None):#,k_4=None):
    network = BatchNormLayer(lasagne.layers.InputLayer(shape=(None,num_bands,None,None),input_var=input_var)) #Patch sizes varying between train-val and test
#    network = lasagne.layers.InputLayer(shape=(None,10,None,None),input_var=input_var)#Patch sizes varying between train-val and test
#    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(k_1,k_1),nonlinearity=lasagne.nonlinearities.rectify)#, W=lasagne.init.Normal(std=0.001,mean=0))
    network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(k_1,k_1),nonlinearity=lasagne.nonlinearities.rectify,pad='same')#, W=lasagne.init.Normal(std=0.001,mean=0))
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(k_2,k_2),nonlinearity=lasagne.nonlinearities.rectify,pad='same')#,W=lasagne.init.Normal(std=0.001,mean=0))
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(k_2,k_2),nonlinearity=lasagne.nonlinearities.rectify,pad='same')#,W=lasagne.init.Normal(std=0.001,mean=0))
#    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)#
    network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(k_3,k_3),nonlinearity=lasagne.nonlinearities.tanh,pad='same')#,W=lasagne.init.Normal(std=0.001,mean=0))#,nonlinearity=lasagne.nonlinearities.rectify)#,W=lasagne.init.Normal(std=0.001,mean=0))#,nonlinearity=lasagne.nonlinearities.tanh)#, nonlinearity=our_activation)# lasagne.nonlinearities.tanh #W=lasagne.init.Normal(std=0.001,mean=0),
    return network

#def build_cnn(input_var=None,k_1=None,k_2=None,k_3=None):
#    network = BatchNormLayer(lasagne.layers.InputLayer(shape=(None,10,None,None),input_var=input_var)) #Patch sizes varying between train-val and test
##    network = lasagne.layers.InputLayer(shape=(None,10,None,None),input_var=input_var)#Patch sizes varying between train-val and test
#    
#    network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(k_1,k_1),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01))#, W=lasagne.init.Normal(std=0.001,mean=0))
#    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(k_2,k_2),nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01))#,W=lasagne.init.Normal(std=0.001,mean=0))
##    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)#
#    network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(k_3,k_3),nonlinearity=lasagne.nonlinearities.tanh)#,W=lasagne.init.Normal(std=0.001,mean=0))#,nonlinearity=lasagne.nonlinearities.rectify)#,W=lasagne.init.Normal(std=0.001,mean=0))#,nonlinearity=lasagne.nonlinearities.tanh)#, nonlinearity=our_activation)# lasagne.nonlinearities.tanh #W=lasagne.init.Normal(std=0.001,mean=0),
#    return network
    

#    network = lasagne.layers.InputLayer(shape=(None,2,None,None),input_var=input_var) #Patch sizes varying between train-val and test
#    network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(9, 9),W=lasagne.init.Normal(std=0.001,mean=0),nonlinearity=lasagne.nonlinearities.rectify)
##    network = lasagne.layers.MaxPool2DLayer(network, 3, stride=1, pad=(0, 0), ignore_border=True)
#    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),W=lasagne.init.Normal(std=0.001,mean=0),nonlinearity=lasagne.nonlinearities.rectify)
# #==============================================================================
# #     network = lasagne.layers.LocalResponseNormalization2DLayer(network, alpha=0.0001, k=2, beta=0.75, n=5)
# #==============================================================================
#    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),W=lasagne.init.Normal(std=0.001,mean=0), nonlinearity=lasagne.nonlinearities.rectify)
#    network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(3,3),W=lasagne.init.Normal(std=0.001,mean=0))#,nonlinearity=lasagne.nonlinearities.tanh) 
#    return network

def Total_Var(input_var=None):
#    network = BatchNormLayer(lasagne.layers.InputLayer(shape=(None,5,None,None),input_var=input_var)) #Patch sizes varying between train-val and test
    network = lasagne.layers.InputLayer(shape=(None,1,None,None),input_var=input_var)#Patch sizes varying between train-val and test
    Weights=np.ndarray(shape=(2,1,2,2),dtype='float32')
    Weights[0,0,:,:]=np.asarray([[1, 0],[-1, 0]],dtype='float32')
    Weights[1,0,:,:]=np.asarray([[1, -1],[0, 0]],dtype='float32')
    network = lasagne.layers.Conv2DLayer(network, num_filters=2, filter_size=(2,2),W=lasagne.utils.create_param(Weights, (2,1,2,2), name=None), b=lasagne.utils.create_param(np.zeros((2,)), (2,), name=None))
    return network

def build_grad(input_var=None):
#    network = BatchNormLayer(lasagne.layers.InputLayer(shape=(None,1,None,None),input_var=input_var)) #Patch sizes varying between train-val and test
    network = lasagne.layers.InputLayer(shape=(None,1,None,None),input_var=input_var)#Patch sizes varying between train-val and test
    Weights=np.ndarray(shape=(4,1,2,2),dtype='float32')
    Weights[0,0,:,:]=np.asarray([[1, 0],[-1, 0]],dtype='float32')
    Weights[1,0,:,:]=np.asarray([[1, -1],[0, 0]],dtype='float32')
    Weights[2,0,:,:]=np.asarray([[1, 0],[0 ,-1]],dtype='float32')
    Weights[3,0,:,:]=np.asarray([[0 ,1],[-1, 0]],dtype='float32')
    network = lasagne.layers.Conv2DLayer(network, num_filters=4, filter_size=(2,2),W=lasagne.utils.create_param(Weights, (4,1,2,2), name=None), b=lasagne.utils.create_param(np.zeros((4,)), (4,), name=None))
    return network
    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False): #bachsize=128, Shuffle=?
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(data_folder, output_folder, identifier, num_epochs,identifierPre):#
    t0,tt0,v0,test00,rate= 0,0,0,0,(10**(-4))#(0.5)*(10**(-2))#(10**(-4))#(0.5)*(10**(-2))#(0.5)*(10**(-2))#(0.5)*(10**(-3))#(0.5)*(10**(-2))#10**(-4)
    # Hyper-parameters
    k_1 = 9 # receptive field side - layer 1
    k_2 = 5# receptive field side - layer 1
    k_3 = 5 # receptive field side - layer 1
    r = ((k_1 - 1) + (k_2 - 1) + (k_2 - 1) + (k_3 - 1)) / 2    
    num_bands = 10

    ########################################################
        # Load the dataset
    Crops = 1
    w_loss_todo = [1.0000, 0.6332, 0.4362, 0.2622, 0.0268, 0]
    
    weight_pan = [[0.8929, 0.8712, 0.846, 0.7972, 0.3508, 0],[0.933, 0.9113, 0.8861, 0.8373, 0.391, 0.0402],[0.9955, 0.9738, 0.9486, 0.8998, 0.4534, 0.1026],[0.9423, 0.964, 0.9892, 1.0, 0.5537, 0.2028]]    
    sizer = [128,256,512] # patch (linear) size
    dims = ["small","medium","large"]#["small","medium","large"]#
    zone = ["Sidney"]#,"Tokyo","Adis_Abeba","New_York","Athens"]
    band = ['05','06','07','8A','11','12']
    ps_first = 512
    patches = [3,3,3,2,3]
    for city in range(len(zone)):
        city_name = zone[city]
        for patch in range(len(patches)):
            patchette = patches[patch]
            for patch0 in range(patchette):
                for dimensions in range(len(dims)): 
                    preambol_name = dims[dimensions]
                    ps = sizer[dimensions]
                    k_p_p = 0
                    for k in band:
                        X_traina, y_traina, X_vala, y_vala, X_testa, y_testa, X_test2a, y_test2a = load_dataset(data_folder, ps_first, r, k, num_bands, city_name, patch0)
                        Ts = ps # 2*ps
                        Ts_2 = ps//2
                        X_train, y_train, X_val, y_val, X_test, y_test, X_test2, y_test2 = X_traina[:,:,:Ts_2,:Ts_2], y_traina[:,:,:Ts_2,:Ts_2], X_vala[:,:,:Ts_2,:Ts_2], y_vala[:,:,:Ts_2,:Ts_2], X_testa[:,:,:Ts_2+2*r,:Ts_2+2*r], y_testa[:,:,:Ts_2+2*r,:Ts_2+2*r], X_test2a[:,:,:Ts+2*r,:Ts+2*r], y_test2a[:,:,:Ts+2*r,:Ts+2*r]
                        pred_test2 = np.ndarray(shape=( 6,Crops*(Ts), Crops*(Ts)), dtype='float32') # 1,1,
                        pred_test = np.ndarray(shape=( 6,Crops*(Ts_2), Crops*(Ts_2)), dtype='float32') # 1,1,
                        w_loss = w_loss_todo[k_p_p]
                        rate = 10**(-5)           
                        tutto = 0
                    
                        print(y_test.shape)
                    
                        # Prepare Theano variables for inputs and targets
                        eps_var = T.constant(10**(-10))        
                        input_var = T.tensor4('inputs')
                        input1_var = T.tensor4('inputs1')
                        target_var = T.tensor4('targets2') #T.ivector('targets')
                        prediction1 = T.tensor4('targets1')
                        target_pan = T.tensor4('targetn') #T.ivector('targets')
                        ndvi0 = T.tensor4('n0')
                        # Model building
                        print("Building model and compiling functions...")
                        network = build_cnn(input_var,num_bands,k_1,k_2,k_3)#, k_4)
                    # sto caricando i dati da un preaddestramento   
                        suffix = '_ID'+identifier
                #        if gg > 1: 
                #            rate = (0.5)*rate
                        suffixPre = '_ID'+identifierPre #+ str(gg-1) + '_s4_' #+'_FineTuning2'
                        with np.load('/datasets/remotesensing/Dataset_MRI_Tomografia/Models/model'+suffixPre+k+'.npz') as f: #/home2/mass.gargiulo/Immagini/model_IDNDVI_TRIPLE_all_date_without_2.npz
                #    /home2/mass.gargiulo/PAN_BANDS_CLASSIFICATION/Codici_Test_Training_OLD/Old_Training/TRAINING/
        #                with np.load('/home2/mass.gargiulo/PAN_BANDS_CLASSIFICATION/JSTARS/Raffaele/Models_FT/model'+suffixPre+k+'.npz') as f: #/home2/mass.gargiulo/Immagini/model_IDNDVI_TRIPLE_all_date_without_2.npz
                            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                            lasagne.layers.set_all_param_values(network, param_values)
    #                    print(param_values[0].shape)
    #                    print(param_values[2].shape)
    #                    print(param_values[4].shape)
                        # Create loss for training
                        prediction = lasagne.layers.get_output(network)
        #### Begin Loss  = L1 + Old Struct + Reg                
                        prediction = prediction + ndvi0
                    
                #        T1 = build_grad(prediction - target_pan)
                        T1 = build_grad(prediction - target_var)
                        T2 = build_grad(target_var - target_pan)
                        predictionTV = Total_Var(prediction)
                #            lasagne.layers.set_all_param_values(network, param_values)
                        # Create loss for training
                        test_1 = lasagne.layers.get_output(T1, deterministic=True)
                        test_2 = lasagne.layers.get_output(T2, deterministic=True)
                        test_3 = lasagne.layers.get_output(predictionTV, deterministic=True)
                        loss1 = abs(prediction - target_var)#(2**(16)/2000)*
                        loss2 = T.sqrt(T.nonzero_values(abs(test_1 - test_2)))
                        loss3 = abs(test_3)
                        loss =  loss1.mean()#
        #                loss = (loss1.mean() + w_loss*loss2.mean() + (w_loss/2)*loss3.mean()) #
        #                loss = (loss1.mean() + loss2.mean() + loss3.mean()) #
        #### End Loss                
        #### New Loss
                        
                        # We could add some weight decay as well here, see lasagne.regularization.
                    
                        # Create update expressions for training
                        # Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum
                        params = lasagne.layers.get_all_params(network, trainable=True)
                        l_rate = T.scalar('learn_rate','float32')
                #        updates = lasagne.updates.momentum(loss, params, l_rate, momentum=0.9)
                        #adamax in the original version. 
                        updates =lasagne.updates.adamax(loss, params, l_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
                        # Create a loss expression for validation/testing. The crucial difference here is
                        # that we do a deterministic forward pass through the network, disabling dropout layers.
                        test_prediction = lasagne.layers.get_output(network, deterministic=True)
                        test_prediction = test_prediction + ndvi0
                    #    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
                        test_loss = abs(test_prediction - target_var) #(2**(16)/2000)*
                    #    test_loss = -np.log10(1-abs(test_prediction - target_var))   
                    
                        test_loss = test_loss.mean()
                    
                    
                    #    test_loss = test_loss.max()
                        # Compile a function performing a training step on a mini-batch (by giving
                        # the updates dictionary) and returning the corresponding training loss:
                #        train_fn = theano.function([input_var, ndvi0, target_var, l_rate], loss, updates=updates, allow_input_downcast=True)
        #                train_fn = theano.function([input_var, ndvi0,target_pan,target_var, l_rate], loss, updates=updates, allow_input_downcast=True)
                        train_fn = theano.function([input_var, ndvi0,target_var, l_rate], loss, updates=updates, allow_input_downcast=True)
                        # Compile a second function computing the validation loss and accuracy:
                        val_fn = theano.function([input_var,ndvi0, target_var], test_loss)  # [test_loss, test_acc])
                    
                        # Finally, launch the training loop.
                        print("Starting training...")
                        test_loss_curve = []
                        train_loss_curve = []
                        val_loss_curve = []
                        plot_period = 60.0
                        partial_time = time.time() - (plot_period + 1.0)
                        # We iterate over epochs:
                    #    fn = theano.function([input_var], prediction)
                    
                        count = 0
                        for epoch in range(num_epochs):
                            ###Controllo per arrestare l'addestramento 
                            if count <= 2: 
                         ####################################       
                                # In each epoch, we do a full pass over the training data:
                                train_err = 0
                        #        if fmod(epoch,10) == 0: rate *= 0.5
                                train_batches = 0
                        #        if rate < 0.0001: rate = 0.0001
                                start_time = time.time()
                                for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
                                    inputs, targets = batch
                                    targetn = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2],inputs.shape[3]),dtype='float32')
                                    targets2 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2],inputs.shape[3]),dtype='float32')
        
        #                            targetn = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2]-2*r,inputs.shape[3]-2*r),dtype='float32')
        #                            targets2 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2]-2*r,inputs.shape[3]-2*r),dtype='float32')
        
                        #            c,d,e,f,PANS = 1.5,1.5,0.5,0.5,4 #WmeanPAN
        #                            c,d,e,f, PANS = 1,1,1,1,4
                                    c,d,e,f = weight_pan[0][k_p_p],weight_pan[1][k_p_p],weight_pan[2][k_p_p],weight_pan[3][k_p_p]
                                    PANS = c + d + e + f #meanPAN
                        #            c,d,e,f,PANS = 1,0,0,0,1 #PAN            
                                    
                                    targetn[:,0,:,:] = ((c)*targets[:,2,:,:]+(d)*targets[:,3,:,:]+(e)*targets[:,4,:,:]+(f)*targets[:,5,:,:])/(PANS)
                        #            pred = fn(inputs)
                        #            err = abs(targets - pred)
                        #            err_train = err.mean()
                        #            print(err_train)
                                    n0 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2],inputs.shape[3]),dtype='float32')
                                    n0[:,0,:,:] = targets[:,1,:,:]    
                                    targets2[:,0,:,:] = targets[:,0,:,:]      
                                    train_err += train_fn(inputs, n0, targets2,rate)
        #                            train_err += train_fn(inputs, n0, targetn, targets2,rate)           
                                    train_batches += 1
                                
                                    # And a full pass over the validation data:
                                val_err = 0
                                val_batches = 0
                                for batch in iterate_minibatches(X_val, y_val, 1, shuffle=True):
                                    inputs, targets = batch
                                    targets2 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2],inputs.shape[3]),dtype='float32')
                                    n0 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2],inputs.shape[3]),dtype='float32')          
        
        #                            targets2 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2]-2*r,inputs.shape[3]-2*r),dtype='float32')
        #                            n0 = np.ndarray(shape = (inputs.shape[0],1,inputs.shape[2]-2*r,inputs.shape[3]-2*r),dtype='float32')          
        
                                    n0[:,0,:,:] = targets[:,1,:,:]   
                                    targets2[:,0,:,:] = targets[:,0,:,:]      
                        #            err = val_fn(inputs, n0, targets)
                                    err = val_fn(inputs, n0, targets2)
                                    val_err += err
                                    val_batches += 1
                                tempo = time.time() - start_time
                            
                                tutto += tempo
                                    # Then we print the results for this epoch:
                                print("Epoch {} of {} took {:.3f}s".format(
                                    epoch + 1, num_epochs, time.time() - start_time))
                                t = train_err / train_batches
                                v = val_err / val_batches
                                
                                print("  training loss:\t\t{:.10f}".format(train_err / train_batches))
                                print("  validation loss:\t\t{:.10f}".format(val_err / val_batches))
                                print("  modification of training loss:\t\t{:.10f}".format(t0-t))
                                print("  modification of validation loss:\t\t{:.10f}".format(v0-v))
            
            ## Conteggio per bloccare l'addestramento
            
                                if epoch > 1 and v0 < v: 
                                    print('v0 minore di v')
                                    count +=1
                                    rate = (0.5)*rate
                                elif v0 > v: 
                                    print('v0 maggiore di v')
                                    count = 0
                                    
                                t0 = t
                                v0 = v
                                train_loss_curve.append(t)
                                val_loss_curve.append(v)
                        
                                # PARTIAL OUTPUT
                                   #+'_ep'+str(len(train_loss_curve))
                                   
                                
                                sio.savemat(output_folder+city_name + '_' + preambol_name + '_' + str(patch0) + '_' + 'loss'+suffix+k+'.mat',
                                            {'train_loss': np.asarray(train_loss_curve), 'val_loss': np.asarray(val_loss_curve)})
                    #            np.savez(output_folder+'model'+suffix+k+'_FineTuning'+'.npz', *lasagne.layers.get_all_param_values(network))
                                np.savez(output_folder+city_name + '_' + preambol_name + '_' + str(patch0) + '_' + 'model'+suffix+k + '.npz', *lasagne.layers.get_all_param_values(network))
                    # time training 
                        network2 = build_cnn(input1_var,num_bands,k_1,k_2,k_3)#k_4
                        with np.load(output_folder+city_name + '_' + preambol_name + '_' + str(patch0) + '_' + 'model'+suffix+k + '.npz') as g:#output_folder
                            param_values = [g['arr_%d' % i] for i in range(len(g.files))]
                        lasagne.layers.set_all_param_values(network2, param_values)
                
                        prediction1 = lasagne.layers.get_output(network2, deterministic=True)
                
                # Compile a function performing a training step on a mini-batch (by giving
                # the updates dictionary) and returning the corresponding training loss:
                        test_fn_n = theano.function([input1_var], prediction1)#, allow_input_downcast=True)
                    
                        pred_test1 = test_fn_n(X_test)
                        pred_test12 = test_fn_n(X_test2)
                        print(X_test.shape)
                        print(y_test.shape)
                        print(pred_test1.shape)
                        print(X_test2.shape)
                        print(pred_test12.shape)
                        print(y_test2.shape)
    #                    pred_test[k_p_p,(start_ind - 4) *( Ts ):(start_ind -4 + 1)*(Ts ), (start_ind1 -4) *( Ts):(start_ind1 -4 + 1)*(Ts)] = pred_test1[0,0,r:-r,r:-r] + y_test[0,0,r:-r,r:-r]
    #                    pred_test2[k_p_p,(start_ind - 4) *( Ts ):(start_ind -4 + 1)*(Ts ), (start_ind1 -4) *( Ts):(start_ind1 -4 + 1)*(Ts)] = pred_test12[0,0,r:-r,r:-r] + y_test2[0,0,r:-r,r:-r]
                        pred_test[k_p_p,:Ts_2, :Ts_2] = pred_test1[0,0,r:-r,r:-r] + y_test[0,0,r:-r,r:-r]
                        pred_test2[k_p_p,:Ts, :Ts] = pred_test12[0,0,r:-r,r:-r] + y_test2[0,0,r:-r,r:-r]
                        k_p_p += 1
                
                    P2 = np.ndarray(shape=( Crops*(Ts_2), Crops*(Ts_2)), dtype='float32') # 1,1,
                    P = np.ndarray(shape=( Crops*(Ts), Crops*(Ts)), dtype='float32') # 1,1,
                    kappa = 0
                    for k_p in range(6):
                        
                        im20 = output_folder + city_name + '_' + preambol_name + '_' + str(patch0) + '_B' + band[k_p] +'_' +identifier +'_20.tif'
                        P2 = pred_test[kappa,:,:]
                        ndvi1_array = np.asarray(P)
                        ndvi1_array = Image.fromarray(ndvi1_array, mode='F') # float32
                        ndvi1_array.save(im20, "TIFF")
                        
                        im10 = output_folder + city_name + '_' + preambol_name + '_' + str(patch0) + '_B' + band[k_p] +'_' +identifier +'_10.tif'
                        P = pred_test2[kappa,:,:]
                        ndvi1_array = np.asarray(P)
                        ndvi1_array = Image.fromarray(ndvi1_array, mode='F') # float32
                        ndvi1_array.save(im10, "TIFF")
                        kappa += 1 
    #       
if __name__ == '__main__':
    kwargs = {}
    kwargs['data_folder'] = '/datasets/remotesensing/MAX/Dataset_MaxAntonio/'
#    kwargs['data_folder'] = '/home2/mass.gargiulo/PAN_BANDS_CLASSIFICATION/JSTARS/Raffaele/DATASET_Reunion/'
    kwargs['output_folder'] = '/datasets/remotesensing/MAX/Dataset_MaxAntonio_output/'
    kwargs['identifier'] = 'PAN_FNew_'
    kwargs['identifierPre'] = 'PAN_PSme_'
    kwargs['n_epochs'] = 1
    main(kwargs['data_folder'], kwargs['output_folder'], kwargs['identifier'], kwargs['n_epochs'], kwargs['identifierPre'])#


    #                loss2 = (abs(prediction) + eps_var )/(abs(target_pan) + eps_var )
    #                prediction1 = prediction + ndvi0
    #            
    #        #        T1 = build_grad(prediction - target_pan)
    #        #        T2 = build_grad(target_var - target_pan)
    #        #        T1 = build_grad(prediction)
    #                
    #                
    #        #        T2 = build_grad(target_pan)
    #                predictionTV = Total_Var(prediction1)
    #        #            lasagne.layers.set_all_param_values(network, param_values)
    #                # Create loss for training
    #        #        test_1 = lasagne.layers.get_output(T1, deterministic=True)
    #        #        test_2 = lasagne.layers.get_output(T2, deterministic=True)
    #                test_3 = lasagne.layers.get_output(predictionTV, deterministic=True)
    #                loss1 = abs(prediction1 - target_var) ####(2**(16)/2000)*
    #        #        loss2 = T.sqrt(T.nonzero_values(abs(test_1 - test_2)))
    #        #        loss_t_1 = abs(test_1)
    #        #        loss_t_2 = abs(test_2)
    #        #
    #        #        loss2 = (loss_t_1 + eps_var )/(loss_t_2 + eps_var )  
    #        	#loss2 = T.nonzero_values(T.sqrt(T.nonzero_values(abs(test_1))) - T.sqrt(T.nonzero_values(abs(test_2))))
    #                loss3 = abs(test_3)
    #            #    loss = lasagne.objectives.squared_error(prediction, target_var)
    #                loss1a = loss1.mean()
    #                loss2a = loss2.mean()
    #                loss3a = loss3.mean()
    #                loss =  ((0.8999)*loss1a + (0.0001)*loss2a + (0.1)*loss3a) #
    #### End New Loss
