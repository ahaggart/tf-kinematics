#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import random
from sys import argv


export_dir = "model_exports/relu-reduced/1518395391"
with tf.Session() as sess:
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
  goal = tf.constant([[1],[0],[0]],dtype=tf.float32)
  end_affector = tf.get_default_graph().get_tensor_by_name("dense_6/BiasAdd:0")
  thetas = tf.get_default_graph().get_tensor_by_name("concat:0")
  print(sess.run([end_affector,thetas],feed_dict={
        "goal_x:0":[[float(argv[1])]],
        "goal_y:0":[[float(argv[2])]],
        "goal_z:0":[[float(argv[3])]],
    }))
