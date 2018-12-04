from CapsNet import CapsNet
from helper_functions import *
import os


def train(model):
    global fd_train
    x_train, y_train, x_valid, y_valid = load_data(load_type='train')
    print('Data set Loaded')
    num_batches = int(y_train.shape[0] / BATCH_SIZE)
    if not os.path.exists(CHECKPOINT_PATH_DIR):
        os.makedirs(CHECKPOINT_PATH_DIR)

    with tf.Session() as sess:
        if RESTORE_TRAINING:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(CHECKPOINT_PATH_DIR)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Loaded')
            start_epoch = int(str(ckpt.model_checkpoint_path).split('-')[-1])
            train_file, val_file, best_loss_val = load_existing_details()
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
            print('All variables initialized')
            train_file, val_file = write_progress('train')
            start_epoch = 0
            best_loss_val = np.infty
        print('Training Starts')
        acc_batch_all = loss_batch_all = np.array([])
        train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        for epoch in range(start_epoch, EPOCHS):
            # Shuffle the input data
            x_train, y_train = shuffle_data(x_train, y_train)
            for step in range(num_batches):
                start = step * BATCH_SIZE
                end = (step + 1) * BATCH_SIZE
                global_step = epoch * num_batches + step
                x_batch, y_batch = x_train[start:end], y_train[start:end]
                feed_dict_batch = {model.X: x_batch, model.Y: y_batch, model.mask_with_labels: True}
                if not (step % 100):
                    _, acc_batch, loss_batch, summary_ = sess.run([model.train_optimizer, model.accuracy,
                                                                     model.combined_loss, model.summary_],
                                                                    feed_dict=feed_dict_batch)
                    train_writer.add_summary(summary_, global_step)
                    acc_batch_all = np.append(acc_batch_all, acc_batch)
                    loss_batch_all = np.append(loss_batch_all, loss_batch)
                    mean_acc,mean_loss = np.mean(acc_batch_all),np.mean(loss_batch_all)
                    summary_ = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=mean_acc)])
                    train_writer.add_summary(summary_, global_step)
                    summary_ = tf.Summary(value=[tf.Summary.Value(tag='Loss/combined_loss', simple_value=mean_loss)])
                    train_writer.add_summary(summary_, global_step)

                    train_file.write(str(global_step) + ',' + str(mean_acc) + ',' + str(mean_loss) + "\n")
                    train_file.flush()
                    print("  Batch #{0}, Epoch: #{1}, Mean Training loss: {2:.4f}, Mean Training accuracy: {3:.01%}".format(
                        step, (epoch+1), mean_loss, mean_acc))
                    acc_batch_all = loss_batch_all = np.array([])
                else:
                    _, acc_batch, loss_batch = sess.run([model.train_optimizer, model.accuracy, model.combined_loss],
                                                        feed_dict=feed_dict_batch)
                    acc_batch_all = np.append(acc_batch_all, acc_batch)
                    loss_batch_all = np.append(loss_batch_all, loss_batch)

            # Validation metrics after each EPOCH
            acc_val, loss_val = eval_performance(sess, model, x_valid, y_valid)
            val_file.write(str(epoch + 1) + ',' + str(acc_val) + ',' + str(loss_val) + '\n')
            val_file.flush()
            print("\rEpoch: {}  Mean Train Accuracy: {:.4f}% ,Mean Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                epoch + 1, mean_acc * 100, acc_val * 100, loss_val,
                " (improved)" if loss_val < best_loss_val else ""))

            # Saving the improved model
            if loss_val < best_loss_val:
                saver.save(sess, CHECKPOINT_PATH_DIR + '/model.tfmodel', global_step=epoch + 1)
                best_loss_val = loss_val
        train_file.close()
        val_file.close()


def test(model):
    x_test, y_test = load_data(load_type='test')
    print('Loaded the test dataset')
    test_file = write_progress('test')
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_PATH_DIR)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model Loaded')
        acc_test, loss_test = eval_performance(sess, model, x_test, y_test)
        test_file.write(str(acc_test) + ',' + str(loss_test) + '\n')
        test_file.flush()
        print('-----------------------------------------------------------------------------')
        print("Test Set Loss: {0:.4f}, Test Set Accuracy: {1:.01%}".format(loss_test, acc_test))


def reconstruct_sample(model, n_samples=5):
    x_test, y_test = load_data(load_type='test')
    sample_images, sample_labels = x_test[:BATCH_SIZE], y_test[:BATCH_SIZE]
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_PATH_DIR)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        feed_dict_samples = {model.X: sample_images, model.Y: sample_labels}
        decoder_out, y_predicted = sess.run([model.decoder_output, model.y_predicted],
                                       feed_dict=feed_dict_samples)
    reconstruction(sample_images, sample_labels, decoder_out, y_predicted, n_samples)


def main(_):
    # Train the model and evaluate on test set
    model = CapsNet()
    print ("Step1: Train")
    train(model)
    print("Step2: Testing the performance of model on the Test Set")
    test(model)
    print ("Step3: Reconstructing some sample images")
    reconstruct_sample(model,n_samples =3)

if __name__ == "__main__":
    tf.app.run()
