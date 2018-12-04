import os
from time import time
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from parameters import *


def create_model_dir():
    current_time = time()
    model_dir = LOGGING_DIR + "/model_files_{}".format(current_time)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def create_freeze_graph_dir(model_dir):
    freeze_graph_dir = os.path.join(model_dir, "freeze")
    if not os.path.exists(freeze_graph_dir):
        os.makedirs(freeze_graph_dir)
    return freeze_graph_dir

def create_optimized_graph_dir(model_dir):
    optimized_graph_dir = os.path.join(model_dir, "optimized")
    if not os.path.exists(optimized_graph_dir):
        os.makedirs(optimized_graph_dir)
    return optimized_graph_dir

def create_frozen_graph(sess,output_name,freeze_graph_dir):
    frozen_graph = freeze_session(sess,
                                  output_names=output_name)
    tf.train.write_graph(frozen_graph, freeze_graph_dir, FREEZE_FILE_NAME , as_text=False)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Converts the existing graph into a new graph where variable nodes are replaced by
    constants. New graph trims the existing graph of any operations which are not required
    to compute the requested output.

    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def pb_to_tensorboard(input_graph_dir,graph_type ="freeze"):
    '''
    Converts the graph ".pb" file to Tensorboard readable format
    :param input_graph_dir: Directory where the graph file is stored
    :param graph_type: "freeze" or "optimize" depending on the operation.
    :return: Saves the file in the folder which can be opened through Tensorboard
    '''
    file_name = ""
    if graph_type == "freeze":
        file_name = FREEZE_FILE_NAME
    elif graph_type == "optimize":
        file_name = OPTIMIZE_FILE_NAME

    with tf.Session() as sess:
        model_filename = input_graph_dir + "/" + file_name
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    train_writer = tf.summary.FileWriter(input_graph_dir)
    train_writer.add_graph(sess.graph)

def strip(input_graph, drop_scope, input_before, output_after, pl_name):
    '''
    This function strips the drop_scope node from the graph.
    :param input_graph: Input graph
    :param drop_scope: Scope like "Dropout" which needs to be removed from the graph
    :param input_before: Input before the drop_scope
    :param output_after:  Output after the drop_scope
    :param pl_name: Name of pl
    :return: stripped output graph
    '''
    input_nodes = input_graph.node
    nodes_after_strip = []
    for node in input_nodes:
        if node.name.startswith(drop_scope + '/'):
            continue

        if node.name == pl_name:
            continue

        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        if new_node.name == output_after:
            new_input = []
            for node_name in new_node.input:
                if node_name == drop_scope + '/cond/Merge':
                    new_input.append(input_before)
                else:
                    new_input.append(node_name)
            del new_node.input[:]
            new_node.input.extend(new_input)
        else:
            new_input= []
            for node_name in new_node.input:
                if node_name == drop_scope + '/cond/Merge':
                    new_input.append(input_before)
                else:
                    new_input.append(node_name)
            del new_node.input[:]
            new_node.input.extend(new_input)

        nodes_after_strip.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_strip)
    return output_graph


def optimize_graph(input_dir, output_dir):
    '''
    This is used to optimize the frozen graph by removing any unnecessary ops
    :param input_dir: directory where input graph is stored.
    :param output_dir: directory where final graph should be stored.
    :return: None
    '''
    input_graph = os.path.join(input_dir, FREEZE_FILE_NAME)
    output_graph = os.path.join(output_dir, OPTIMIZE_FILE_NAME)

    input_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(input_graph, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = strip(input_graph_def, u'dropout_1', u'conv2d_2/bias', u'dense_1/kernel', u'training')
    output_graph_def = strip(output_graph_def, u'dropout_3', u'max_pooling2d_2/MaxPool', u'flatten_2/Shape',
                             u'training')
    output_graph_def = strip(output_graph_def, u'dropout_4', u'dense_3/Relu', u'dense_4/kernel', u'training')
    output_graph_def = strip(output_graph_def, u'Adadelta_1', u'softmax_tensor_1/Softmax',
                             u'training/Adadelta/Variable', u'training')
    output_graph_def = strip(output_graph_def, u'training', u'softmax_tensor_1/Softmax',
                             u'_', u'training')

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
