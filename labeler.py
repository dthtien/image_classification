from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Labeler(object):
  def __init__(self, image, type='path'):
    self.image = image
    self.input_height = 299
    self.input_width = 299
    self.input_mean = 0
    self.input_std = 255
    self.input_layer = 'Placeholder'
    self.output_layer = 'final_result'
    self.file_name = 'test_container.jpg'
    self.labels_file = 'training_model/labels_v4.txt'
    self.model_file = 'training_model/graph_v4.pb'
    self.type = type

  def read_tensor_from_image_file(self):
    file_name = self.image if self.type == 'path' else self.image.filename
    input_height = self.input_height
    input_width = self.input_width
    input_mean = self.input_mean
    input_std = self.input_std
    input_name = "file_reader"
    output_name = "normalized"

    file_reader = tf.read_file(file_name, input_name) if self.type == 'path' else self.image.read()

    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
      image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result

  def load_graph(self):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open(self.model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

  def load_labels(self):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(self.labels_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

  def execute(self):
    graph = self.load_graph()
    t = self.read_tensor_from_image_file()

    input_name = "import/" + self.input_layer
    output_name = "import/" + self.output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
      })

    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = self.load_labels()

    response = {}
    for i in top_k:
      response[labels[i]] = results[i]

    return response
