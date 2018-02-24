import tensorflow as tf
import numpy as np
from pathlib import Path
import os

from pydl_audio_encoders.library.utility.audio_utils import compute_melgram
from pydl_audio_encoders.library.utility.cifar10_loader import download_cifar10_model_if_not_found
from pydl_audio_encoders.library.utility.gtzan_loader import gtzan_labels


class Cifar10AudioEncoder(object):

    model_name = 'audio-encoders'

    def __init__(self, working_directory=None):
        if working_directory is None:
            working_directory = os.path.join(str(Path.home()), Cifar10AudioEncoder.model_name)

        self.working_directory = working_directory

        if not os.path.exists(self.working_directory):
            os.mkdir(self.working_directory)

        self.model_file_path = os.path.join(self.working_directory, 'cifar10.pb')

        download_cifar10_model_if_not_found(self.model_file_path)

        with tf.gfile.FastGFile(self.model_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            self.graph_def = graph_def

    def print_graph_nodes(self):
        for n in self.graph_def.node:
            print(n.name)

    def encode(self, audio_path, high_dimension=True):
        with tf.Session() as sess:
            output_name = 'dense_2/BiasAdd:0'
            if high_dimension:
                output_name = 'dense_1/BiasAdd:0'
            predict_op = sess.graph.get_tensor_by_name(output_name)
            mg = compute_melgram(audio_path)
            mg = np.expand_dims(mg, axis=0)

            predicted = sess.run(predict_op, feed_dict={"conv2d_1_input:0": mg})

            return predicted[0]

    def predict(self, audio_path):
        with tf.Session() as sess:
            predict_op = sess.graph.get_tensor_by_name('output_node0:0')
            mg = compute_melgram(audio_path)
            mg = np.expand_dims(mg, axis=0)

            predicted = sess.run(predict_op, feed_dict={"conv2d_1_input:0": mg})

            return predicted[0]

    def predict_class(self, audio_path):
        predicted = self.predict(audio_path)
        return np.argmax(predicted)

    def predict_class_label(self, audio_path):
        predicted_label_idx = self.predict_class(audio_path)
        return gtzan_labels[predicted_label_idx]



