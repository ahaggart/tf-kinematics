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
    # goal_positions_t = tf.add(positions,perturbations)
    goal_positions_t = positions
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
    return (goal_positions,positions)

def build_kinematic_training_dataset(goal):
    return tf.data.Dataset.from_generator(
        lambda:move_to_position(goal[0],goal[1],goal[2]),
        tf.float32
    ).make_one_shot_iterator()

def euclidian_distance_sq(x,y):
    return tf.reduce_sum(tf.square(tf.subtract(x,y)),[1]) 

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
    goal_x = []
    goal_y = []
    goal_z = []
    for step in np.arange(0,path_len,step_size):
        point = tf.add(start_pos,tf.multiply(tf.constant([[step]]),path))     
        goal_x.append([point[0]])
        goal_y.append([point[1]])
        goal_z.append([point[2]])
    features = {
        "goal_x":tf.constant(goal_x,dtype=tf.float32),
        "goal_y":tf.constant(goal_y,dtype=tf.float32),
        "goal_z":tf.constant(goal_z,dtype=tf.float32),
    }
    return features;

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


def kin_solver_model_fn(features,labels,mode,params):
	# set up the input layer using the feature columns info
	# function call sets up the net for chaining later
    net = tf.feature_column.input_layer(features,params['feature_columns'])

	#set up the hidden layers
	#sequentially add layers to the net
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # output layer
    result = tf.layers.dense(net,units=params['num_joints'],activation=None)

    # use the supplied kinematic equation to find end affector location
    # TODO: transform/normalize outputs to [0,2pi]?
    kinematics = params['kinematic_eqn'](result,tf.constant(params['lengths'],dtype=tf.float32))

    #actions for prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"kinematics":result}
        return tf.estimator.EstimatorSpec(
                    mode,
                    predictions,
                    export_outputs={
                        "joint_angles":tf.estimator.export.PredictOutput({
                            "thetas":tf.placeholder(tf.float32,name="thetas"),
                    })}
        )

    # define the loss layers
    # compute the distance from the desired result
    distance_loss = tf.reduce_mean(tf.sqrt(euclidian_distance_sq(kinematics,labels)),[0])

    # TODO: how to weight the angular and distance losses?
    combined_loss = distance_loss

    #actions for eval mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode,
                loss=combined_loss,
                eval_metric_ops=tf.metrics.mean_squared_error(kinematics,labels),
                )

    #actions for train mode
    assert mode == tf.estimator.ModeKeys.TRAIN 

    optimizer = tf.train.AdamOptimizer()
    # optimizer = tf.train.AdagradOptimizer(0.01)
    train_op = optimizer.minimize(combined_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=combined_loss, train_op=train_op)

def kin_serving_input_receiver_fn():
    feature_spec = {
        "test_x":tf.placeholder(tf.float32,shape=[0,0],name="goal_x"),
        "goal_y":tf.placeholder(tf.float32,shape=[0,0],name="goal_y"),
        "goal_z":tf.placeholder(tf.float32,shape=[0,0],name="goal_z"),
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

def get_kinematics_solver(model_dir,lengths,num_joints,kinematic_eqn,hidden_units):
    #define feature columns
    goal_x = tf.feature_column.numeric_column(key="goal_x")
    goal_y = tf.feature_column.numeric_column(key="goal_y")
    goal_z = tf.feature_column.numeric_column(key="goal_z")
    feature_columns = [goal_x,goal_y,goal_z]

    kinematics_solver = tf.estimator.Estimator(
        model_fn=kin_solver_model_fn,
        model_dir=model_dir,
        params={
            'feature_columns':feature_columns,
            'lengths':lengths,
            'num_joints':num_joints,
            'kinematic_eqn':kinematic_eqn,
            'hidden_units':hidden_units,
            })

    return kinematics_solver   

def main(argv):
    num_joints = 3
    kinematic_eqn = forward_kinematics_3
    batch_size=10000
    hidden_layers=5
    hidden_layer_size=5
    if len(argv) >= 4:
        num_steps=int(argv[3])
    nickname=argv[2]

    hidden_units = [hidden_layer_size for i in range(0,hidden_layers)]
    model_dir_base = "models/{}/{}seg/{}l{}u";
    model_dir= model_dir_base.format(nickname,num_joints,hidden_layers,hidden_layer_size)

    lengths = [[1,1,1]]


    kinematics_solver = get_kinematics_solver(
                            model_dir,lengths,
                            num_joints,
                            kinematic_eqn,
                            hidden_units)

    if argv[1] == '--train':
        input_fn = lambda:kin_input_gen(kinematic_eqn,batch_size,num_joints)
        kinematics_solver.train(input_fn=input_fn,steps=num_steps)
    elif argv[1] == '--solve':
        goal = [float(argv[3]),float(argv[4]),float(argv[5])]
        solutions = kinematics_solver.predict(
            input_fn=lambda:move_to_position(goal[0],goal[1],goal[2])
        )
        for solution in solutions:
            print("got here")
            print(solution)
    elif argv[1] == '--eval':
        input_fn = lambda:kin_input_gen(kinematic_eqn,batch_size,num_joints)
        kinematics_solver.evaluate(input_fn=input_fn)
    elif argv[1] == '--save':
        export_base = 'model_exports/{}'.format(nickname)
        kinematics_solver.export_savedmodel(
            export_base,
            kin_serving_input_receiver_fn(),
            )

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

