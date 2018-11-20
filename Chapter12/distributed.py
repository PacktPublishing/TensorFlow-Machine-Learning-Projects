import tensorflow as tf

cluster = tf.train.ClusterSpec({
                "worker": [
                           "192.168.1.4:2222",
                           "192.168.1.5:2223"
                          ]})
# Define Servers
worker0 = tf.train.Server(cluster, job_name="worker", task_index=0)
#worker1 = tf.train.Server(cluster, job_name="worker", task_index=1)


with tf.device("/job:worker/task:1"):
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)  
    add_node = tf.add(a,b)

    
with tf.device("/job:worker/task:0"):
    mul_node =  a * b
    
    
with tf.Session("grpc://192.168.1.4:2222") as sess:
    result = sess.run([add_node, mul_node])
    print(result)