#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import random

def kin_input_gen(kin_fn,batch_size,num_joints):
    iterations = range(0,batch_size)
    thetas = []
    lengths = []
    for joint in range(0,num_joints):
        thetas.append(tf.constant([[random.random()*2*np.pi] for i in iterations],dtype=np.float32))
        lengths.append(tf.constant([[1] for i in iterations],dtype=np.float32))
        # lengths.append(tf.constant([[random.random()] for i in iterations],dtype=np.float32))
    theta_features = dict([('theta{}'.format(i),thetas[i]) for i in range(0,num_joints)])
    length_features = dict([('len{}'.format(i),lengths[i]) for i in range(0,num_joints)])

    positions = kin_fn(tf.concat(thetas,1),tf.concat(lengths,1)) 
    perturbation_amount = 0.01 #centimeters(?)
    perturbation_t = tf.constant([[perturbation_amount]])
    perturbations = tf.concat([tf.multiply(
                        tf.constant([[random.random(),random.random(),0]]),
                        perturbation_t) 
                    for i in iterations],0)
    goal_positions_t = tf.add(positions,perturbations)
    goal_positions = {
        "goal_x":goal_positions_t[:,0],
        "goal_y":goal_positions_t[:,1],
        "goal_z":goal_positions_t[:,2]
    }
    # merge all the dictionaries
    features = {}
    features.update(theta_features)
    features.update(length_features)
    features.update(goal_positions)
    print('got some output')
    return (features,goal_positions_t)

def kinematic_input_generator(kin_fn,num_joints):
    while True:
        thetas = []
        lengths = []
        for joint in range(0,num_joints):
            thetas.append(tf.constant([[random.random()*2*np.pi]],dtype=np.float32))
            lengths.append(tf.constant([[random.random()]],dtype=np.float32))
        theta_features = dict([('theta{}'.format(i),thetas[i]) for i in range(0,num_joints)])
        length_features = dict([('len{}'.format(i),lengths[i]) for i in range(0,num_joints)])

        positions = kin_fn(tf.concat(thetas,1),tf.concat(lengths,1)) 
        perturbation_amount = 0.01 #centimeters(?)
        perturbation_t = tf.constant([[perturbation_amount]])
        perturbations = tf.multiply(
                            tf.constant([[random.random(),random.random(),0]]),
                            perturbation_t)
        goal_positions_t = tf.add(positions,perturbations)
        goal_positions = {
            "goal_x":goal_positions_t[:,0],
            "goal_y":goal_positions_t[:,1],
            "goal_z":goal_positions_t[:,2]
        }
        # merge all the dictionaries
        features = {}
        features.update(theta_features)
        features.update(length_features)
        features.update(goal_positions)
        loop = False
        print('got some output')
        yield (features,goal_positions_t)

def build_kinematic_training_dataset(kin_fn,batch_size,num_joints):
    return tf.data.Dataset.from_generator(
        lambda:kin_input_gen(kin_fn,batch_size,num_joints)
    )

def move_to_position(x,y,z):
    goal_v = np.array([x,y,z])
    goal = tf.constant(goal_v,dtype=tf.float32)
    segments = tf.constant([[1,1,1]],dtype=tf.float32)
    start_angles = tf.constant([[0,0,0]],dtype=tf.float32)
    start_pos = forward_kinematics_3(
        start_angles,
        segments
    )

    path = np.subtract(goal_v,np.array([1,0,0]))
    path_len = np.linalg.norm(path)
    step_size = 0.01
    checkpoints = [
        tf.add(start_pos,tf.multiply(tf.constant([[step]]),path))
        for step in np.arange(0,path_len,step_size)
    ]
    yield {
        "goal_x":checkpoints[1][:,0],
        "goal_y":checkpoints[1][:,1],
        "goal_z":checkpoints[1][:,2],
        "theta0":[[0.0]],
        "theta1":[[0.0]],
        "theta2":[[0.0]],
        "len0":[[1.0]],
        "len1":[[1.0]],
        "len2":[[1.0]]
    }

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

    # build a 2 dimensional tensor by concatenating length tensors
    length_list = [features['len{}'.format(i)] for i in range(0,params['num_joints'])]
    lengths = tf.concat(length_list,1)

    # build a 2 dimensional tensor by concatenating angle tensors
    theta_list = [features['theta{}'.format(i)] for i in range(0,params['num_joints'])]
    start_angles = tf.concat(length_list,1)

    # use the supplied kinematic equation to find end affector location
    # TODO: transform/normalize outputs to [0,2pi]?
    kinematics = params['kinematic_eqn'](result,lengths)

    #actions for prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {kinematics:result}
        return tf.estimator.EstimatorSpec(mode,predictions)

    # define the loss layers
    # compute the distance from the desired result
    distance_loss = tf.reduce_mean(
                        tf.sqrt(
                            tf.reduce_sum(
                                tf.square(tf.subtract(kinematics,labels)),
                                [1])),
                        [0]
                    )

    # compute the difference from the calculated state to the previous state
    # minimize joint adjustment to reach new position
    angular_diff = tf.subtract(result,start_angles)
    angular_loss = tf.reduce_sum(tf.atan2(tf.sin(angular_diff),tf.cos(angular_diff)),1)

    # TODO: how to weight the angular and distance losses?
    combined_loss = distance_loss

    #actions for eval mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode,
                loss=combined_loss,
                eval_metric_ops={"angular_loss":angular_loss,
                                 "distance_loss":distance_loss}
                )

    #actions for train mode
    assert mode == tf.estimator.ModeKeys.TRAIN 

    optimizer = tf.train.AdamOptimizer()
    # optimizer = tf.train.AdagradOptimizer(0.01)
    train_op = optimizer.minimize(combined_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=combined_loss, train_op=train_op)


# test kinematic functions
# test_theta  = tf.constant([[np.pi/2,-np.pi/2,-np.pi/2]],dtype=tf.float32)
# test_length = tf.constant([[1,1,1]],dtype=tf.float32)
# test_kin    = forward_kinematics_3(test_theta,test_length)

# with tf.Session() as sess:
#     print(sess.run(test_kin))

def main(argv):
    num_joints = 3
    kinematic_eqn = forward_kinematics_3
    batch_size=1000
    hidden_layers=25
    hidden_layer_size=25
    num_steps=int(argv[3])
    nickname=argv[2]

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

    model_dir_base = "models/{}/{}seg/{}l{}u";

    kinematics_solver = tf.estimator.Estimator(
        model_fn=kin_solver_fn,
        model_dir=model_dir_base.format(nickname,num_joints,hidden_layers,hidden_layer_size),
        params={
            'feature_columns':feature_columns,
            'num_joints':num_joints,
            'kinematic_eqn':kinematic_eqn,
            'hidden_units':[hidden_layer_size for i in range(0,hidden_layers)],
            })

    if argv[1] == '--train':
        input_fn = lambda:kin_input_gen(kinematic_eqn,batch_size,num_joints)
        kinematics_solver.train(input_fn=input_fn,steps=num_steps)
    elif argv[1] == '--solve':
        solutions = kinematics_solver.predict(
            input_fn=lambda:move_to_position(float(argv[2]),float(argv[3]),float(argv[4]))
        )
        for solution in solutions:
            print(solution)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

