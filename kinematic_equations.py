#!/usr/bin/env python

# useful kinematics math

import numpy as np
import tensorflow as tf
import random
import math

# transform a tensor from joint-space to cartesian space
def forward_kinematics_3(thetas,lengths):
    sins = tf.sin(tf.cumsum(thetas,1))
    coss = tf.cos(tf.cumsum(thetas,1))
    x_comps = tf.multiply(lengths,coss) # could use tensordot
    y_comps = tf.multiply(lengths,sins)

    x_final = tf.reduce_sum(x_comps,1,keepdims=True)
    y_final = tf.reduce_sum(y_comps,1,keepdims=True)
    z_final = tf.zeros(tf.shape(x_final),dtype=tf.float32)

    return tf.concat([x_final,y_final,z_final],1)

def euclidian_distance_sq(x,y):
    return tf.reduce_sum(tf.square(tf.subtract(x,y)),[1]) 

def euclidian_distance(x,y):
    return tf.sqrt(euclidian_distance_sq(x,y))

def random_configurations(num_joints,batch_size):
    return np.array([
        random.random()*2*np.pi
        for j in range(0,num_joints)
        for i in range(0,batch_size)
    ]).reshape(-1,num_joints)

def random_end_affector(max_length,batch_size):
    batch = []
    for i in range(0,batch_size):
        r = max_length*random.random()
        theta = random.random()*2*np.pi
        batch.append([r*math.cos(theta),r*math.sin(theta),0])
    return np.array(batch).reshape(-1,3)
