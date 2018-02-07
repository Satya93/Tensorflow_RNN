# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:02:28 2018

@author: satya
"""

import tensorflow as tf
sess = tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)
target = 1000.

two_gate = tf.add(tf.multiply(a,x_data),b)
loss = tf.square(tf.subtract(two_gate, target))

init = tf.initialize_all_variables()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

print('Optimizing Two Gate output to '+str(target))
for i in range(20):
    sess.run(train_step,feed_dict={x_data:x_val})
    a_val,b_val = (sess.run(a),sess.run(b))
    mult_op = sess.run(two_gate, feed_dict={x_data:x_val})
    print(str(a_val)+' x '+str(x_val)+' + '+str(b_val)+' = '+str(mult_op))