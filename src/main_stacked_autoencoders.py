# packages
import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# for reproducible results
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


################
#Dataset import#
################

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

##########################
#PCA with an auto-encoder#
##########################

reset_graph()

d_inputs = 28 * 28
d_hidden1 = 300
d_hidden2 = 150  # codings
d_hidden3 = d_hidden1
d_outputs = d_inputs
n_class = 10

learning_rate = 0.01
l2_reg = 0.0005

initializer = tf.contrib.layers.variance_scaling_initializer()
activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

#Phase 1 (building its own tf subgraph)

#Input data
X = tf.placeholder(tf.float32,shape=[None, d_inputs])

#Hidden layer1 (first code generating layer)
weights1_init = initializer([d_inputs, d_hidden1])
weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
biases1 = tf.Variable(tf.zeros(d_hidden1), name="biases1")
hidden1 = activation(tf.matmul(X, weights1) + biases1)

#Output layer (input reconstruction)
weights1_ = tf.transpose(weights1,name="weights1_")
biases1_ = tf.Variable(tf.zeros(d_outputs),name="biases1_")
outputs1_ = activation(tf.matmul(hidden1,weights1_) + biases1_)

#Objective function: MSE + L2 penalty
reconstruction_loss_phase1 = tf.reduce_mean(tf.square(outputs1_ - X)) 
reg_loss_phase1 = regularizer(weights1) 
J_phase1 = reconstruction_loss_phase1 + reg_loss_phase1

optimizer_phase1 = tf.train.AdamOptimizer(learning_rate)
training_op_phase1 = optimizer_phase1.minimize(J_phase1)


##Phase 2 (also building its own tf subgraph)
#
#Input data (intermediate representation)
Z = tf.placeholder(tf.float32,shape=[None, d_hidden1])

#Hidden layer2 (second code generating layer)
weights2_init = initializer([d_hidden1, d_hidden2])
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
biases2 = tf.Variable(tf.zeros(d_hidden2), name="biases2")
hidden2_ = activation(tf.matmul(Z, weights2) + biases2)

#Hidden layer3 (second intermediate representation reconstruction layer = output for this graph)
weights2_ = tf.transpose(weights2,name="weights2_")
biases2_ = tf.Variable(tf.zeros(d_hidden1),name="biases2_")
outputs2_ = activation(tf.matmul(hidden2_,weights2_) + biases2_)

#Objective function: MSE + L2 penalty
reconstruction_loss_phase2 = tf.reduce_mean(tf.square(outputs2_ - Z))
reg_loss_phase2 = regularizer(weights2)
J_phase2 = reconstruction_loss_phase2 + reg_loss_phase2

optimizer_phase2 = tf.train.AdamOptimizer(learning_rate)
training_op_phase2 = optimizer_phase2.minimize(J_phase2)

###################################################################
##Stacking the layers together to build the multilayer autoencoder#
###################################################################

#Reconnecting first coding layer with the second one
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2) 
hidden3 = activation(tf.matmul(hidden2,weights2_) + biases2_) 
#Reconnecting first decoding layer with the output one
outputs = activation(tf.matmul(hidden3,weights1_) + biases1_) 

#####################################################
##Stacking the layers together to build a classifier#
#####################################################

#Adding a final layer (softmax) for supervised training on top of coding layers
weights3_init_stack = initializer([d_hidden2, n_class])
weights3_stack = tf.Variable(weights3_init_stack, dtype=tf.float32, name="weights3_mlp")
biases3_stack = tf.Variable(tf.zeros(n_class), name="biases3_mlp")
logits = tf.matmul( hidden2 , weights3_stack) + biases3_stack 
y = tf.placeholder(tf.int32, shape=[None])
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3_stack)
loss = cross_entropy + reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
#
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

batch_size = 150
n_labeled_instances = 20000
init = tf.global_variables_initializer()
saver = tf.train.Saver() #one saver for all sessions

with tf.Session() as sess:
    n_epochs = 5
    init.run()
    #Phase 1
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op_phase1, feed_dict={X: X_batch})
    Z_train = sess.run(hidden1,feed_dict={X: mnist.train.images})
        
    #Phase 2   
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = np.random.permutation(n_labeled_instances)[:batch_size]
            Z_batch = Z_train[:batch_size]
            sess.run(training_op_phase2, feed_dict={Z: Z_batch})
    
    # Showing images
    print("")
    print("---------------------------------------------------------------------------------")
    print("couple d’image avant/après reconstruction")
    print("---------------------------------------------------------------------------------")
    X_test = mnist.test.images
    results = sess.run(outputs,feed_dict={X: X_test})
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.imshow(X_test[0].reshape(28,28))
    plt.subplot(1,2,2)
    plt.imshow(results[0].reshape(28,28))
    plt.show()
    
    #Supervised learning
    n_epochs = 20
    print("---------------------------------------------------------------------------------")
    print("Poids du réseau")
    print("---------------------------------------------------------------------------------")
    print(weights1.eval())
    print("---------------------------------------------------------------------------------")
    print("Accuracy")
    print("---------------------------------------------------------------------------------")
    for epoch in range(n_epochs):
        n_batches = n_labeled_instances // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            indices = np.random.permutation(n_labeled_instances)[:batch_size]
            X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end=" ")
        saver.save(sess, "./my_model_supervised.ckpt")
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy:", accuracy_val)    

##########################
#END--------of-------code#
##########################