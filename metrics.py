import tensorflow as tf
import tensorflow.keras.backend as K

class DiceCoeff:
    def __init__(self):
        pass

    def dice_coef(self, y_true, y_pred):
        y_true_flat = tf.keras.layers.Flatten()(y_true)
        y_pred_flat = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        smooth = 1e-3
        return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)

    def __call__(self, y_true, y_pred):
        return self.dice_coef(y_true, y_pred)

class Precision(tf.keras.metrics.Metric):
    def __init__(self, name='precision', **kwargs):
        super(Precision, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bool = tf.cast(y_true, tf.bool)
        y_pred_bool = tf.cast(y_pred, tf.bool)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true_bool, y_pred_bool), tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(y_pred_bool, tf.float32))
        self.true_positives.assign_add(true_positives)
        self.predicted_positives.assign_add(predicted_positives)

    def result(self):
        return self.true_positives / (self.predicted_positives + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.predicted_positives.assign(0)

class Recall(tf.keras.metrics.Metric):
    def __init__(self, name='recall', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.possible_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bool = tf.cast(y_true, tf.bool)
        y_pred_bool = tf.cast(y_pred, tf.bool)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true_bool, y_pred_bool), tf.float32))
        possible_positives = tf.reduce_sum(tf.cast(y_true_bool, tf.float32))
        self.true_positives.assign_add(true_positives)
        self.possible_positives.assign_add(possible_positives)

    def result(self):
        return self.true_positives / (self.possible_positives + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.possible_positives.assign(0)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
