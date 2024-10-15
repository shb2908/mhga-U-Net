import os
import tensorflow as tf

def create_train(tissue_train, mask_train, batch, train_path):
    def parse_images(tissue_train, mask_train):
        tissue_file_str = tf.strings.join([train_path, "TissueImages/", tissue_train])
        mask_file_str = tf.strings.join([train_path, "GroundTruth/", mask_train])

        tissue_image = tf.io.read_file(tissue_file_str)
        mask_image = tf.io.read_file(mask_file_str)

        tissue_image = tf.image.decode_png(tissue_image, channels=3)
        mask_image = tf.image.decode_png(mask_image, channels=1)

        tissue_image = tf.image.resize(tissue_image, [512, 512])
        mask_image = tf.image.resize(mask_image, [512, 512])

        tissue_image = tf.cast(tissue_image, tf.float32) / 255.0
        mask_image = tf.cast(mask_image, tf.float32) / 255.0

        return tissue_image, mask_image

    dataset = tf.data.Dataset.from_tensor_slices((tissue_train, mask_train))
    dataset = dataset.map(parse_images)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()

    return dataset

def create_test(tissue_test, mask_test, batch, test_path):
    def parse_images(tissue_test, mask_test):
        tissue_file_str = tf.strings.join([test_path, "TissueImages/", tissue_test])
        mask_file_str = tf.strings.join([test_path, "GroundTruth/", mask_test])

        tissue_image = tf.io.read_file(tissue_file_str)
        mask_image = tf.io.read_file(mask_file_str)

        tissue_image = tf.image.decode_png(tissue_image, channels=3)
        mask_image = tf.image.decode_png(mask_image, channels=1)

        tissue_image = tf.image.resize(tissue_image, [512, 512])
        mask_image = tf.image.resize(mask_image, [512, 512])

        tissue_image = tf.cast(tissue_image, tf.float32) / 255.0
        mask_image = tf.cast(mask_image, tf.float32) / 255.0

        return tissue_image, mask_image

    dataset = tf.data.Dataset.from_tensor_slices((tissue_test, mask_test))
    dataset = dataset.map(parse_images)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()

    return dataset
