import os
import pickle
from parameters import *
import tensorflow as tf
import numpy as np

def load_data():
    """
    Loading Data
    """
    input_file = os.path.join(TEXT_SAVE_DIR)
    with open(input_file, "r") as f:
        data = f.read()

    return data

def preprocess_and_save_data():
    """
    Preprocessing the Book Scripts Dataset
    """
    text = load_data()
    token_dict = define_tokens()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_map(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('processed_text.p', 'wb'))


def load_preprocess_file():
    """
    Loading the processed Book Scripts Data
    """
    return pickle.load(open('processed_text.p', mode='rb'))


def save_params(params):
    """
    Saving parameters to file
    """
    pickle.dump(params, open('parameters.p', 'wb'))


def load_params():
    """
    Loading parameters from file
    """
    return pickle.load(open('parameters.p', mode='rb'))

def create_map(input_text):
    """
    Map words in vocab to int and vice versa for easy lookup
    :param input_text: Book Script data split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab = set(input_text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    return vocab_to_int, int_to_vocab

def define_tokens():
    """
    Generate a dict to turn punctuation into a token. Note that Sym before each text denotes Symbol
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    dict = {'.':'_Sym_Period_',
            ',':'_Sym_Comma_',
            '"':'_Sym_Quote_',
            ';':'_Sym_Semicolon_',
            '!':'_Sym_Exclamation_',
            '?':'_Sym_Question_',
            '(':'_Sym_Left_Parentheses_',
            ')':'_Sym_Right_Parentheses_',
            '--':'_Sym_Dash_',
            '\n':'_Sym_Return_',
           }
    return dict

def generate_batch_data(int_text):
    """
    Generate batch data of x (inputs) and y (targets)
    :param int_text: Text with the words replaced by their ids
    :return: Batches as a Numpy array
    """
    num_batches = len(int_text) // (BATCH_SIZE * SEQ_LENGTH)

    x = np.array(int_text[:num_batches * (BATCH_SIZE * SEQ_LENGTH)])
    y = np.array(int_text[1:num_batches * (BATCH_SIZE * SEQ_LENGTH) + 1])

    x_batches = np.split(x.reshape(BATCH_SIZE, -1), num_batches, 1)
    y_batches = np.split(y.reshape(BATCH_SIZE, -1), num_batches, 1)
    batches = np.array(list(zip(x_batches, y_batches)))
    return batches

def extract_tensors(tf_graph):
    """
    Get input, initial state, final state, and probabilities tensor from the graph
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (tensor_input,tensor_initial_state,tensor_final_state, tensor_probs)
    """
    tensor_input = tf_graph.get_tensor_by_name("Input/input:0")
    tensor_initial_state = tf_graph.get_tensor_by_name("Network/initial_state:0")
    tensor_final_state = tf_graph.get_tensor_by_name("Network/final_state:0")
    tensor_probs = tf_graph.get_tensor_by_name("Network/probs:0")
    return tensor_input, tensor_initial_state, tensor_final_state, tensor_probs

def select_next_word(probs, int_to_vocab):
    """
    Select the next work for the generated text
    :param probs: list of probabilities of all the words in vocab which can be selected as next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: predicted next word
    """
    index = np.argmax(probs)
    word = int_to_vocab[index]
    return word


def predict_book_script():
    _, vocab_to_int, int_to_vocab, token_dict = load_preprocess_file()
    seq_length, load_dir = load_params()

    script_length = 250 # Length of Book script to generate. 250 denotes 250 words

    first_word = 'postgresql' # postgresql or any other word from the book

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = extract_tensors(loaded_graph)

        # Sentences generation setup
        sentences = [first_word]
        previous_state = sess.run(initial_state, {input_text: np.array([[1]])})
        # Generate sentences
        for i in range(script_length):
            # Dynamic Input
            dynamic_input = [[vocab_to_int[word] for word in sentences[-seq_length:]]]
            dynamic_seq_length = len(dynamic_input[0])

            # Get Prediction
            probabilities, previous_state = sess.run([probs, final_state], {input_text: dynamic_input, initial_state: previous_state})
            probabilities= np.squeeze(probabilities)

            pred_word = select_next_word(probabilities[dynamic_seq_length - 1], int_to_vocab)
            sentences.append(pred_word)

        # Scraping out tokens from the words
        book_script = ' '.join(sentences)
        for key, token in token_dict.items():
            book_script = book_script.replace(' ' + token.lower(), key)
        book_script = book_script.replace('\n ', '\n')
        book_script = book_script.replace('( ', '(')

        # Write the generated script to a file
        with open("book_script", "w") as text_file:
            text_file.write(book_script)

        print(book_script)
