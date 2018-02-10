#!/usr/bin/env python

import numpy as np
import tensorflow as tf

def input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()



seg_zero_len = tf.feature_column.numeric_column(key="len0")
seg_one_len  = tf.feature_column.numeric_column(key="len1")
seg_two_len  = tf.feature_column.numeric_column(key="len2")

joint_zero_theta = tf.feature_column.numeric_column(key="theta0")
joint_one_theta  = tf.feature_column.numeric_column(key="theta1")
joint_two_theta  = tf.feature_column.numeric_column(key="theta2")

goal_x = tf.feature_column.numeric_column(key="goal_x")
goal_y = tf.feature_column.numeric_column(key="goal_y")
goal_z = tf.feature_column.numeric_column(key="goal_z")

feature_columns = [seg_zero_len,seg_one_len,seg_two_len,
                   joint_zero_theta,joint_one_theta,joint_two_theta,
                   goal_x,goal_y,goal_z]

# transform a tensor from joint-space to cartesian space
def forward_kinematics_3(thetas,lengths):
    sins = tf.sin(thetas)
    coss = tf.cos(thetas)
    x_comps = tf.mul(lengths,coss)
    y_comps = tf.mul(lengths,sins)
    x_final = tf.reduce_sum(x_comps)
    y_final = tf.reduce_sum(y_comps)
    z_final = tf.constant([0])
    return tf.concat([x_final,y_final,z_final],0)


def kin_solver_fn(features,labels,mode,params):
	# set up the input layer using the feature columns info
	# function call sets up the net for chaining later
    net = tf.feature_column.input_layer(features,params['feature_columns'])

	#set up the hidden layers
	#sequentially add layers to the net
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # output layer
    result = tf.layers.dense(net,units=params['num_joints'],activation=None)

    lengths = features.slice(0) #the lengths are a feature

    kinematics = forward_kinematics_3(result,lengths)

    #actions for prediction mode

    # define the loss layers
    # compute the distance from the desired result
    distance_loss = tf.losses.mean_squared_error(
        labels=labels,
        predictions=kinematics,
        reduction=None)

    angular_loss = 0 # compute the difference from the calculated state to the previous state

    #actions for training mode

    #actions for eval mode
	
    pass

num_joints = 3
kinematics_solver = tf.estimator.Estimator(
    model_fn=kin_solver_fn,
    model_dir="~/Documents/Yonder/kiNNematics/models/{}seg".format(num_joints),
    params={
        'feature_columns':feature_columns,
        'num_joints':num_joints})

