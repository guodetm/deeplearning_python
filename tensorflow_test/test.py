import tensorflow as tf
tf.compat.v1.disable_eager_execution()


a = tf.constant([1.0,2.0],name="a")
b = tf.constant([3.0,4.0],name="b")

result = a+b
sess = tf.compat.v1.Session()
print(sess.run(result))