
import warnings
import tensorflow_probability as tfp
from utils import *

tfd = tfp.distributions
warnings.filterwarnings("ignore")

def build_model(images):
    '''
    Defining a LeNet model for traffic sign classification
    :param images:
    :return: defined model
    '''
    with tf.name_scope("BNN", values=[images]):
        model = tf.keras.Sequential([
            tfp.layers.Convolution2DFlipout(10,
                                            kernel_size=5,
                                            padding="VALID",
                                            activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[3, 3],
                                         strides=[1, 1],
                                         padding="VALID"),
            tfp.layers.Convolution2DFlipout(15,
                                            kernel_size=3,
                                            padding="VALID",
                                            activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding="VALID"),
            tfp.layers.Convolution2DFlipout(30,
                                            kernel_size=3,
                                            padding="VALID",
                                            activation=tf.nn.relu),

            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding="VALID"),

            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(400, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(120, activation = tf.nn.relu),
            tfp.layers.DenseFlipout(84, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(43) ])

        logits = model(images)
        targets_distribution = tfd.Categorical(logits=logits)

    return model,logits, targets_distribution


def main(argv):
    del argv #unused
    print ("Extracting dataset from zip files")
    extract_dataset()
    X_train, y_train, X_test,y_test= load_preprocessed_data()
    X_train_gray = load_grayscale_images(X_train)
    X_test_gray = load_grayscale_images(X_test,data_type ='test')

    # Shape of the dataset
    print("Shape of X_train is ", X_train.shape)
    print("Shape of X_test is ", X_test.shape)
    print("Shape of y_train is ", y_train.shape)
    print("Shape of y_test is ", y_test.shape)

    print("Shape of X_train Grayscale is ", X_train_gray.shape)
    print("Shape of X_test is Grayscale", X_test_gray.shape)
    print("Preprocessing Done")

    # Plotting the input data
    #plot_input_data(X_train,y_train)

    # Data Pipeline for modeling
    (images, targets, iter_handle,
     train_iterator, test_iterator) = build_data_pipeline(X_train_gray, X_test_gray,y_train, y_test)

    #Building Model
    model, logits, targets_distribution =build_model(images)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(targets_distribution.log_prob(targets))
    kl = sum(model.losses) / X_train.shape[0]
    elbo_loss = neg_log_likelihood + kl

    # Defining metrics for evalution
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels=targets, predictions=predictions)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(elbo_loss)

    # Extract weight posterior statistics for layers with weight distributions
    # for later visualization.
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(model.layers):
        try:
            q = layer.kernel_posterior
        except AttributeError:
            continue
        names.append("Layer {}".format(i))
        qmeans.append(q.mean())
        qstds.append(q.stddev())

    # Initialize the variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        # Run the training loop.
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        for step in range(EPOCHS):
            _ = sess.run([train_op, accuracy_update_op],
                         feed_dict={iter_handle: train_handle})

            if step % 5== 0:
                loss_value, accuracy_value = sess.run(
                    [elbo_loss, accuracy], feed_dict={iter_handle: train_handle})
                print("Epoch: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                    step, loss_value, accuracy_value))

        #Sampling from the posterior and obtaining mean probability for held out dataset
        probs = np.asarray([sess.run((targets_distribution.probs),
                                     feed_dict={iter_handle: test_handle})
                            for _ in range(NUM_MONTE_CARLO)])
        mean_probs = np.mean(probs, axis=0)

        test_acc_dist = []
        for prob in probs:
            y_test_pred = np.argmax(prob, axis=1).astype(np.float32)
            accuracy = (y_test_pred == y_test).mean() * 100
            test_acc_dist.append(accuracy)

        plt.hist(test_acc_dist)
        plt.title("Histogram of prediction accuracies on test dataset")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        save_dir = os.path.join(DATA_DIR, "..", "Plots")
        plt.savefig(os.path.join(save_dir,  "Test_Dataset_Prediction_Accuracy.png"))

        # Get the average accuracy
        Y_pred = np.argmax(mean_probs, axis=1)
        print("Overall Accuracy in predicting the test data =  percent", round((Y_pred == y_test).mean() * 100,2))
        # Draw two random samples from the test data
        sample_images_idx= np.random.choice(range(X_test_gray.shape[0]), size=10)
        for i in sample_images_idx:
            sampled_image = X_test_gray[i]
            sample_label = y_test[i]
            mean_prediction = Y_pred[i]
            plot_heldout_prediction(sampled_image, probs[:,i,:],
                                    fname="Sample{:05d}_pred".format(i),
                                    title="Correct Label {:02d}, Mean Prediction {:02d}"
                                    .format(sample_label,mean_prediction))

        qm_vals, qs_vals = sess.run((qmeans, qstds))

        # Plotting Weight Means and Standard deviation
        plot_weight_posteriors(names, qm_vals, qs_vals,
                               fname="step{:05d}_weights.png".format(step))

if __name__ == "__main__":
    tf.app.run()
