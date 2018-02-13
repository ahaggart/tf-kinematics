#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import random
import build_kinematics as kin_builder
import kinematic_equations as kin_eqn
import time
from sys import argv


export_dir = "models/custom/testing/model.ckpt"
num_joints = 3
batch_size = 10000
num_steps = 100000
lengths = [1,1,1]
kinematics_eqn = kin_eqn.forward_kinematics_3

# build the graph
prediction,loss,train_op = kin_builder.build(num_joints,lengths,kinematics_eqn)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, export_dir)
    end_affector = tf.get_default_graph().get_tensor_by_name("forward_kinematics_3:0")
    thetas = tf.get_default_graph().get_tensor_by_name("joint_angles:0")
    start = np.array([1,0,0])
    goal = np.array([float(argv[1]),float(argv[2]),float(argv[3])])
    path = kin_eqn.move_to_position(start,goal)

    print("Moving from {} to {}....".format(start,goal))

    start_time = time.time()
    kin_out = sess.run([end_affector,thetas],feed_dict={
        "goal:0":path,
    })

    print("{} positions in {}s:".format(len(path),time.time()-start_time))
    
    float_printer = "{0}: ({1:-.2f},{2:-.2f},{3:-.2f})\t"
    for i in range(0,len(kin_out[0])):
        position    = kin_out[0][i]
        theta       = kin_out[1][i]
        print(float_printer.format("pos",position[0],position[1],position[2])),
        print(float_printer.format("theta",theta[0],theta[1],theta[2]))
