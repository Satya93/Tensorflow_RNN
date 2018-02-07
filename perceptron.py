# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:02:28 2018

@author: satya
"""

import tensorflow as tf
sess = tf.Session()

a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)
target = 10.

product = tf.multiply(a,x_data)
loss = tf.square(tf.subtract(product, target))

init = tf.initialize_all_variables()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

print('Optimizing Product Gate output to '+str(target))
for i in range(10):
    sess.run(train_step,feed_dict={x_data:x_val})
    a_val = sess.run(a)
    mult_op = sess.run(product, feed_dict={x_data:x_val})
    print(str(a_val)+' x '+str(x_val)+' = '+str(mult_op))