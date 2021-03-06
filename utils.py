import numpy as np
import pandas as pd
#import tensorflow as tf
import matplotlib.pyplot as plt

"""
def pre_process_image(image):
    # flip some images
    image = tf.image.random_flip_left_right(image)
    
    # randomly adjust hue, contrast and saturation
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    # limit pixel between [0, 1] in case of overflow
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)

    return image
"""

def generate_batches(data, labels, batch_size):
    p = np.random.permutation(labels.shape[0])
    new_labels = labels[p]
    new_data = data[p]
    start = 0
    end = batch_size
    results = list()
    while end < labels.shape[0]:
        results.append((new_data[start:end], new_labels[start:end]))
        start = end
        end += batch_size
    results.append((new_data[end:], new_labels[end:]))
    return results


def read_from_csv(one_hot, filename, header):
    features = np.genfromtxt(filename, delimiter=',', skip_header = header, usecols=range(1, 785), dtype=np.float32)
    features = np.divide(features, 255.0)

    labels_original = np.genfromtxt(filename, delimiter=',', skip_header = header, usecols=0, dtype=np.int)
    if one_hot:
        labels = np.zeros([len(labels_original), 10])
        labels[np.arange(len(labels_original)), labels_original] = 1
        labels = labels.astype(np.float32)
        return features, labels
    else:
        return features, labels_original
