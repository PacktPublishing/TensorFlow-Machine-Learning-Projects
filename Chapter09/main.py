import tensorflow as tf
import os
from utils import *
from DiscoGAN import DiscoGAN
import random



def train(model):
    # Load the data first
    # Define a function to load the next batch
    # start training

    # Define a function to get the data for the next batch
    def get_next_batch(BATCH_SIZE, type ="shoes"):
        if type == "shoes":
            next_batch_indices = random.sample(range(0, X_shoes.shape[0]), BATCH_SIZE)
            batch_data = X_shoes[next_batch_indices,:,:,:]
        elif type == "bags":
            next_batch_indices = random.sample(range(0, X_bags.shape[0]), BATCH_SIZE)
            batch_data = X_bags[next_batch_indices, :, :, :]
        return batch_data

    # Loading the dataset
    print ("Loading Dataset")
    X_shoes, X_bags = load_data(load_type='train')

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        if RESTORE_TRAINING:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state("./model")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Loaded')
            start_epoch = int(str(ckpt.model_checkpoint_path).split('-')[-1].split(".")[0])
            print ("Start EPOCH", start_epoch)
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
            if not os.path.exists("logs"):
                os.makedirs("logs")
            start_epoch = 0

        # Starting training from here
        train_writer = tf.summary.FileWriter(os.getcwd() + '/logs', graph=sess.graph)
        print ("Starting Training")
        for global_step in range(start_epoch,EPOCHS):
            shoe_batch = get_next_batch(BATCH_SIZE,"shoes")
            bag_batch = get_next_batch(BATCH_SIZE,"bags")
            feed_dict_batch = {model.X_bags: bag_batch, model.X_shoes: shoe_batch}
            op_list = [model.disc_optimizer, model.gen_optimizer, model.disc_loss, model.gen_loss, model.summary_]
            _, _, disc_loss, gen_loss, summary_ = sess.run(op_list, feed_dict=feed_dict_batch)
            shoe_batch = get_next_batch(BATCH_SIZE, "shoes")
            bag_batch = get_next_batch(BATCH_SIZE, "bags")
            feed_dict_batch = {model.X_bags: bag_batch, model.X_shoes: shoe_batch}
            _, gen_loss = sess.run([model.gen_optimizer, model.gen_loss], feed_dict=feed_dict_batch)
            if global_step%10 ==0:
                train_writer.add_summary(summary_,global_step)

            if global_step%100 == 0:
                print("EPOCH:" + str(global_step) + "\tGenerator Loss: " + str(gen_loss) + "\tDiscriminator Loss: " + str(disc_loss))


            if global_step % 1000 == 0:

                shoe_sample = get_next_batch(1, "shoes")
                bag_sample = get_next_batch(1, "bags")

                ops = [model.gen_s_fake, model.gen_b_fake, model.gen_recon_s, model.gen_recon_b]
                gen_s_fake, gen_b_fake, gen_recon_s, gen_recon_b = sess.run(ops, feed_dict={model.X_shoes: shoe_sample, model.X_bags: bag_sample})

                save_image(global_step, gen_s_fake, str("gen_s_fake_") + str(global_step))
                save_image(global_step,gen_b_fake, str("gen_b_fake_") + str(global_step))
                save_image(global_step, gen_recon_s, str("gen_recon_s_") + str(global_step))
                save_image(global_step, gen_recon_b, str("gen_recon_b_") + str(global_step))

            if global_step % 1000 == 0:
                if not os.path.exists("./model"):
                    os.makedirs("./model")
                saver.save(sess, "./model" + '/model-' + str(global_step) + '.ckpt')
                print("Saved Model")

def main():
    # Get the dataset first.

    if not os.path.exists(os.path.join(os.getcwd(), "bags")):
        print("Generating Dataset")
        generate_dataset()
    # Create the model
    print ("Defining the model")
    model = DiscoGAN()
    print ("Training")
    train(model)


if __name__ == "__main__":
    main()








