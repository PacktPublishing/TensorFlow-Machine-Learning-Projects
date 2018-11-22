import sys
import tensorflow as tf
# Add other module libraries you may need

cluster = tf.train.ClusterSpec(
          {'ps':['192.168.1.3:2222'],
           'worker': ['192.168.1.4:2222',
                      '192.168.1.5:2222',
                      '192.168.1.6:2222',
                      '192.168.1.7:2222']
 })

job = sys.argv[1]
task_idx = sys.argv[2]

server = tf.train.Server(cluster, job_name=job, task_index= int(task_idx))

if job == 'ps':  
    # Makes the parameter server wait 
    # until the Server shuts down
    server.join()
else:
    # Executes only on worker machines    
    with tf.device(tf.train.replica_device_setter(cluster=cluster, worker_device='/job:worker/task:'+task_idx)):
        #build your model here like you are working on a single machine
        print("In worker")

    with tf.Session(server.target):
        # Train the model 
        print("Training")
