#!/usr/bin/env python

import numpy as np
import argparse
import tensorflow as tf

import kinematic_equations as kin_eqn

# fully customized neural net building

def build_kinematics_model(params):
    assert len(params["lengths"]) == params["num_joints"]
    # define feature columns
    # goal will be 3DOF cartesian coordinates only (for now)
    goal_coord = tf.feature_column.numeric_column(key="goal",shape=3)
    goal = tf.placeholder(name="goal",shape=[None,3],dtype=tf.float32)
    goal_coord_mapping = {
        "goal":goal
    }
    # build the net from the top down
    # var net represents the bottom layer of the current graph
    # replace value of net when we add a layer below it
    net = tf.feature_column.input_layer(goal_coord_mapping,goal_coord)

    for units in params["hidden_layers"]:
        net = tf.layers.dense(net,units=units,activation=tf.nn.relu)

    # end layer
    net = tf.layers.dense(net,units=params["num_joints"],activation=tf.sigmoid,name="raw_output")

    # output is constrained to (-1,1), so multiply by 2pi to give two full rotations as output
    joint_angles = tf.multiply(net,tf.constant(2*np.pi,dtype=tf.float32),name="joint_angles")

    # convert joint angles to cartesian space
    lengths = tf.constant(params["lengths"],dtype=tf.float32,shape=[1,3])
    end_affector = params["kinematics_eqn"](joint_angles,lengths)
    prediction = end_affector

    # calculate loss based on distance of end affector to goal coords
    dist = kin_eqn.euclidian_distance(end_affector,goal)
    dist_loss = tf.reduce_mean(dist,[0])

    # space to add other loss calculations
    combined_loss = dist_loss

    return prediction,combined_loss

def build(num_joints,lengths,kinematics_eqn):
    # build the computational graph representing our neural net
    prediction,loss = build_kinematics_model(params={
        "num_joints":num_joints,
        "lengths":lengths,
        "kinematics_eqn":kinematics_eqn,
        "hidden_layers":[7 for i in range(0,7)]
    })

    # set up the optimizer
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss,global_step=tf.train.get_global_step())

    return prediction,loss,train_op

def load(sess,saver,save_loc):
    try:
        saver.restore(sess,save_loc)
        # print("Loaded variables from checkpoint: {}".format(save_loc))
    except:
        print("Unable to load saved session at location: {}".format(save_loc))
        print("Creating new session...")
        sess.run(tf.global_variables_initializer())

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    save_loc = "models/custom/testing/model.ckpt"
    print_interval = 100
    num_joints = 3
    batch_size = 10000
    num_steps = 100000
    lengths = [1,1,1]
    kinematics_eqn = kin_eqn.forward_kinematics_3

    # build the graph
    prediction,loss,train_op = build(num_joints,lengths,kinematics_eqn)

    # create a saver for the defualt graph (the one we built)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # load previous values from a checkpoint, if it exists
        load(sess,saver,save_loc)
        init_out = sess.run([loss],feed_dict={
            "goal:0":[[1,0,0]],
        })

        print("Session start Loss: {}".format(init_out[0]))

        # training loop
        for i in range(0,num_steps):
            # for j in range(0,10):
            batch = kin_eqn.random_end_affector(num_joints,batch_size)
            train_out = sess.run([loss,train_op],feed_dict={
                "goal:0":batch,
            })
            if not i%print_interval:
                print("step: {}\tloss: {}".format(i,train_out[0]))

        final_out = sess.run([loss],feed_dict={
            "goal:0":[[1,0,0]],
        })

        print("Session End Loss: {}".format(final_out[0]))


        save_path = saver.save(sess, save_loc)
        print("Model saved in path: %s" % save_path)