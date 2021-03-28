from functools import partial

import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

from utils import scalar_to_onehot, normalize_img


def _random_rotate(img_arr, max_angle):
    image_arr = ndimage.rotate(img_arr, np.random.uniform(-max_angle, max_angle), reshape=False)
    return np.array(image_arr)


def _gaussain_noise(img_arr, var=0.1):
    sigma = var ** 0.5
    mean = 0
    gaussian = np.random.normal(mean, sigma, img_arr.shape)
    img_noisy = img_arr + gaussian
    return img_noisy


def load_doodle_data():
    target_folder = "doodle_dataset"
    train_size = 0.7
    npy_names = os.listdir(target_folder)

    full_images = np.zeros((0, 28, 28))
    full_labels = np.zeros((0))

    for category_idx, npy_name in enumerate(npy_names):
        npy_path_for_a_category = os.path.join(target_folder, npy_name)
        category_name = npy_name.split(".")[0]

        data = np.load(npy_path_for_a_category)
        number_of_data = data.shape[0]

        images = data.reshape(-1, 28, 28)
        labels = np.full((number_of_data), category_idx)

        full_images = np.concatenate((full_images, images), 0)
        full_labels = np.concatenate((full_labels, labels), 0)

    full_labels = full_labels.astype(int)
    return train_test_split(full_images, full_labels, train_size=train_size, shuffle=True, random_state=2021)


def create_dataset(config):
    x_train, x_test, y_train, y_test = load_doodle_data()
    x_train = normalize_img(x_train)
    x_test = normalize_img(x_test)
    y_train_oh = scalar_to_onehot(y_train, 3)
    y_test_oh = scalar_to_onehot(y_test, 3)
    image_train = tf.data.Dataset.from_tensor_slices(x_train[..., np.newaxis])
    label_train = tf.data.Dataset.from_tensor_slices(y_train_oh)
    image_test = tf.data.Dataset.from_tensor_slices(x_test[..., np.newaxis])
    label_test = tf.data.Dataset.from_tensor_slices(y_test_oh)

    def tf_random_rotate(img_tensor):
        im_shape = img_tensor.shape
        random_rotate = partial(_random_rotate, max_angle=config.data.test.rotation.max_angle)
        [image_tensor, ] = tf.py_function(random_rotate, [img_tensor], [tf.float32])
        image_tensor.set_shape(im_shape)
        return image_tensor

    if config.data.augmentation:
        gaussian_noise_aug = partial(_gaussain_noise, var=0.5)
        image_train = image_train.map(tf_random_rotate)
        image_train = image_train.map(gaussian_noise_aug)

    if config.data.test.rotation.apply:
        image_test = image_test.map(tf_random_rotate)

    elif config.data.test.noise.apply:
        gaussian_noise = partial(_gaussain_noise, var=config.data.test.noise.variance)
        image_test = image_test.map(gaussian_noise)

    train_dataset = tf.data.Dataset.zip((image_train, label_train))
    test_dataset = tf.data.Dataset.zip((image_test, label_test))

    train_dataset = train_dataset.batch(config.trainer.batch_size)
    test_dataset = test_dataset.batch(config.trainer.batch_size)

    return train_dataset, test_dataset
