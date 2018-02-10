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

def kin_solver_fn(features,labels,mode,params):
	pass

kinematics_solver = tf.estimator.Estimator(
	model_fn=kin_solver_fn,
	model_dir="~/Documents/Yonder/kiNNematics/models",
	params={})

