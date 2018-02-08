"""doc me."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import tensorflow as tf

import wide_deep_tfrecords as wnd

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_file",
    "E:/codespace/data/census_data/adult.test",
    "Path to the training data.")
flags.DEFINE_boolean(
    "skip_header",
    False,
    "Skip first line.")

def create_csv_iter(path, skip_header=False):
    """
    Returns an iterator over a CSV file. Skips the header.
    """
    with tf.gfile.Open(path) as csv_file:
        reader = csv.reader(csv_file)
        if skip_header: # Skip the header
            next(reader)
        for row in reader:
            yield row

def create_example(row):
    """
    Returns a tensorflow.Example Protocol Buffer object.
    """
    example = tf.train.Example()

    for i in range(len(wnd._CSV_COLUMNS)):
        colname = wnd._CSV_COLUMNS[i]
        colvalue = row[i]
        if colname in wnd.CATEGORICAL_COLUMNS:
            example.features.feature[colname].bytes_list.value.extend([colvalue.encode('utf-8')])
        elif colname in wnd.CONTINUOUS_COLUMNS:
            example.features.feature[colname].int64_list.value.extend([int(colvalue)])

        if colname == "income_bracket":
            example.features.feature[wnd.LABEL_COLUMN].int64_list.value.extend([int(">50K" in colvalue)])

    return example

def create_tfrecords_file(input_file, output_file, example_fn, skip_header):
    """
    Creates a TFRecords file for the given input data and
    example transofmration function
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    print("Creating TFRecords file at", output_file, "...")
    for i, row in enumerate(create_csv_iter(input_file, skip_header)):
        if len(row) == 0:
            continue
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print("Wrote to", output_file)

def save_tfrecord(data_file, skip_header):
    filename, ext = os.path.splitext(data_file)
    output_file = filename + "_" + ext[1:] + ".tfrecords"
    create_tfrecords_file(data_file, output_file, create_example, skip_header)

def main(_):
    if not FLAGS.data_file:
        raise Exception("Missing argument: data_file")
    print("TFRecord converter start...")
    data_file = FLAGS.data_file
    skip_header = FLAGS.skip_header
    print("Data file:", data_file)
    print("Skip header:", skip_header)

    save_tfrecord(data_file, skip_header)

    print("TFRecord converter done!")

if __name__ == "__main__":
    tf.app.run()
