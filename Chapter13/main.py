'''
This is the main file that will be used to generate TV Scripts using Simpson's Dataset
'''
from utils import *
from parameters import *
from model import Model


def train(model,int_text):

    # Creating the checkpoint directory
    if not os.path.exists(CHECKPOINT_PATH_DIR):
        os.makedirs(CHECKPOINT_PATH_DIR)

    batches = generate_batch_data(int_text)

    with tf.Session() as sess:
        if RESTORE_TRAINING:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(CHECKPOINT_PATH_DIR)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Loaded')
            start_epoch = int(str(ckpt.model_checkpoint_path).split('-')[-1])
        else:
            start_epoch = 0
            tf.global_variables_initializer().run()
            print('All variables initialized')

        for epoch in range(start_epoch, NUM_EPOCHS):
            saver = tf.train.Saver()
            state = sess.run(model.initial_state, {model.X: batches[0][0]})

            for batch, (x, y) in enumerate(batches):
                feed = {
                    model.X: x,
                    model.Y: y,
                    model.initial_state: state}
                train_loss, state, _ = sess.run([model.loss, model.final_state, model.train_op], feed)

                if (epoch * len(batches) + batch) % 200 == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch,
                        batch,
                        len(batches),
                        train_loss))
                    # Save Checkpoint for restoring if required
                    saver.save(sess, CHECKPOINT_PATH_DIR + '/model.tfmodel', global_step=epoch + 1)

        # Save Model
        saver.save(sess, SAVE_DIR)
        print('Model Trained and Saved')
        save_params((SEQ_LENGTH, SAVE_DIR))



def main():
    if os.path.exists("./processed_text.p"):
        print ("Processed File Already Present. Proceeding with that")
    else:
        print ("Preprocessing the data")
        preprocess_and_save_data()

    print ("Loading the preprocessed data")
    int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess_file()

    model = Model(int_to_vocab)
    print ("Training the model")
    train(model,int_text)

    print ("Generating the Book Script")
    predict_book_script()


if __name__ == "__main__":
    main()