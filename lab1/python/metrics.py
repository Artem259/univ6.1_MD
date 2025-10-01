import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import Metric


def _cast_y(y_true, y_pred):
    if len(y_true.shape) > 1:
        y_true = tf.squeeze(y_true, axis=-1)
    if len(y_pred.shape) > 1:
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)
    return y_true, y_pred


class PrecisionMacro(Metric):
    def __init__(self, classes_num, name='precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = classes_num
        self.confusion_matrix = self.add_weight(
            shape=(classes_num, classes_num),
            initializer='zeros',
            dtype=tf.int64,
            name='confusion_matrix'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _cast_y(y_true, y_pred)
        batch_confusion_matrix = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.int64
        )
        self.confusion_matrix.assign_add(batch_confusion_matrix)

    def result(self):
        true_positives = tf.cast(tf.linalg.diag_part(self.confusion_matrix), tf.float32)
        predicted_positives = tf.cast(tf.reduce_sum(self.confusion_matrix, axis=0), tf.float32)
        precision_per_class = true_positives / (predicted_positives + K.epsilon())
        return tf.reduce_mean(precision_per_class)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


class RecallMacro(Metric):
    def __init__(self, classes_num, name='recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = classes_num
        self.confusion_matrix = self.add_weight(
            shape=(classes_num, classes_num),
            initializer='zeros',
            dtype=tf.int64,
            name='confusion_matrix'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _cast_y(y_true, y_pred)
        batch_confusion_matrix = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.int64
        )
        self.confusion_matrix.assign_add(batch_confusion_matrix)

    def result(self):
        true_positives = tf.cast(tf.linalg.diag_part(self.confusion_matrix), tf.float32)
        actual_positives = tf.cast(tf.reduce_sum(self.confusion_matrix, axis=1), tf.float32)
        recall_per_class = true_positives / (actual_positives + K.epsilon())
        return tf.reduce_mean(recall_per_class)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


class F1Macro(Metric):
    def __init__(self, classes_num, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = classes_num
        self.confusion_matrix = self.add_weight(
            shape=(classes_num, classes_num),
            initializer='zeros',
            dtype=tf.int64,
            name='confusion_matrix'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _cast_y(y_true, y_pred)
        batch_confusion_matrix = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.int64
        )
        self.confusion_matrix.assign_add(batch_confusion_matrix)

    def result(self):
        true_positives = tf.cast(tf.linalg.diag_part(self.confusion_matrix), tf.float32)
        predicted_positives = tf.cast(tf.reduce_sum(self.confusion_matrix, axis=0), tf.float32)
        actual_positives = tf.cast(tf.reduce_sum(self.confusion_matrix, axis=1), tf.float32)

        precision_per_class = true_positives / (predicted_positives + K.epsilon())
        recall_per_class = true_positives / (actual_positives + K.epsilon())

        f1_per_class = 2 * (precision_per_class * recall_per_class) / (
                precision_per_class + recall_per_class + K.epsilon()
        )

        return tf.reduce_mean(f1_per_class)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


def model_predict(model, data):
    y_true = []
    y_pred = []
    for x_batch, y_batch in data:
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(y_pred_batch, axis=1))
    return np.array(y_true), np.array(y_pred)


def evaluate_metrics(model, test_data):
    ev_metrics = model.evaluate(test_data, verbose=0)
    ev_dict = dict(zip(model.metrics_names, ev_metrics))
    return {
        'f1': ev_dict['f1'],
        'precision': ev_dict['precision'],
        'recall': ev_dict['recall'],
    }


def evaluate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm


def load_metrics(metrics_file):
    if metrics_file.is_file():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}


def save_metrics(fold_metrics, metrics_file, cv_fold):
    metrics = load_metrics(metrics_file)
    metrics.pop('summary', None)
    metrics[f"cv{cv_fold}"] = fold_metrics
    metrics = dict(sorted(metrics.items(), key=lambda x: x[0]))

    num_folds = len(metrics)
    if num_folds > 0:
        summary_metrics = {
            'f1_mean': np.mean([m['f1'] for m in metrics.values()]),
            'f1_max': np.max([m['f1'] for m in metrics.values()]),
            'f1_min': np.min([m['f1'] for m in metrics.values()]),
            'precision_mean': np.mean([m['precision'] for m in metrics.values()]),
            'precision_max': np.max([m['precision'] for m in metrics.values()]),
            'precision_min': np.min([m['precision'] for m in metrics.values()]),
            'recall_mean': np.mean([m['recall'] for m in metrics.values()]),
            'recall_max': np.max([m['recall'] for m in metrics.values()]),
            'recall_min': np.min([m['recall'] for m in metrics.values()]),
        }
        metrics['summary'] = summary_metrics

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
