##################################################################################
#                            Author: Anas ESSOUNAINI                             #
#                         File Name: main_autoencoder.py                         #
#                     Creation Date: July 17, 2020 06:41 PM                      #
#                    Last Updated: November 17, 2020 03:09 AM                    #
#                            Source Language: python                             #
#   Repository: https://github.com/AnasEss/stacked_autoencoders_tensorflow.git   #
#                                                                                #
#                            --- Code Description ---                            #
#                                  autoencoders                                  #
##################################################################################

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

learning_rate = 0.01
l2_reg = 0.0005

initializer = tf.contrib.layers.variance_scaling_initializer()
activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

#Input data
X = tf.placeholder(tf.float32,shape=[None, d_inputs])

#Hidden layer1 (code generating layer)
weights1_init = initializer([d_inputs, d_hidden1])
weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
biases1 = tf.Variable(tf.zeros(d_hidden1), name="biases1")
hidden1 = activation(tf.matmul(X, weights1) + biases1)

#Hidden layer2 (code generating layer)
weights2_init = initializer([d_hidden1, d_hidden2])
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
biases2 = tf.Variable(tf.zeros(d_hidden2), name="biases2")
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)

#Hidden layer3 (intermediate representation reconstruction layer)
weights3 = tf.transpose(weights2,name="weights3")
biases3 = tf.Variable(tf.zeros(d_hidden3),name="biase3")
hidden3 = activation(tf.matmul(hidden2,weights3) + biases3)

#Output layer (input reconstruction)
weights4 = tf.transpose(weights1,name="weights4")
biases4 = tf.Variable(tf.zeros(d_outputs),name="biases4")
outputs = activation(tf.matmul(hidden3,weights4) + biases4)

#Objective function: MSE + L2 penalty
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2)
J = reconstruction_loss + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(J)

init = tf.global_variables_initializer()

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()        
    print("---------------------------------------------------------------------------------")
    print("erreur quadratique moyenne au fil des epochs")
    print("---------------------------------------------------------------------------------")
    for epoch in range(n_epochs):
        print("")
        print("*Epoch "+str(epoch+1)+":")
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
            
        print("  "+str(sess.run(J,feed_dict={X:mnist.train.images})))
    
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

##########################
#END--------of-------code#
##########################