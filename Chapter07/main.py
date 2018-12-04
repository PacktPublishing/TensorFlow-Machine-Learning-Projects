
from utils import *
from model import *


def main():
    # Loading and processing data
    print ("Loading the data")
    processed_data = load_and_preprocess_data()
    print ("Getting train and test dataset")
    X_train, X_test, y_test = get_train_and_test_data(processed_data)

    model_obj = MODEL(X_train,X_test,y_test)
    print ("Training the model")
    model_obj.train_model()
    print ("Loading the trained model")
    model_obj.get_trained_model()
    print ("Get Reconstruction Loss By Class")
    model_obj.plot_reconstruction_error_by_class()
    print ("Getting Precision Recall Curves by Thresholds")
    model_obj.get_precision_recall_curves()
    print ("Get confusion matrix with 80% recall on Test Dataset")
    model_obj.get_confusion_matrix(min_recall = 0.8)


if __name__==main():
    main()















