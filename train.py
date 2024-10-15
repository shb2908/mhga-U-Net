import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from model import FunctionalModel
from data_utils import create_train, create_test
from losses import HybridLoss, DiceLoss, HausdorffLoss
from metrics import DiceCoeff, F1Score, Recall, Precision

# Prepare data
# Replace with actual paths
train_path = '/path/to/train'
test_path = '/path/to/test'

tissue_train = sorted(os.listdir(os.path.join(train_path, "TissueImages")))
mask_train = sorted(os.listdir(os.path.join(train_path, "GroundTruth")))
tissue_test = sorted([file for file in os.listdir(os.path.join(test_path, "TissueImages")) if file.lower().endswith('.png')])
mask_test = sorted(os.listdir(os.path.join(test_path, "GroundTruth")))

TOTAL_TRAIN_SAMPLES = len(tissue_train)
TOTAL_TEST_SAMPLES = len(tissue_test)
BATCH_SIZE = 2

train_dataset = create_train(tissue_train, mask_train, BATCH_SIZE, train_path)
test_dataset = create_test(tissue_test, mask_test, BATCH_SIZE, test_path)

t_steps_per_epoch = TOTAL_TRAIN_SAMPLES // BATCH_SIZE
v_steps_per_epoch = TOTAL_TEST_SAMPLES // BATCH_SIZE

class CustomModelWrapper:
    def __init__(self, model):
        self.model = model
        self.prev_model = model
        self.t_steps_per_epoch = t_steps_per_epoch
        self.v_steps_per_epoch = v_steps_per_epoch
        self.cur_epoch = 0
        self.best_model = {'score': 0, 'model': self.model, 'result': None, 'metric_vals': None}
        self.history = {}

    def compile(self, loss_objs: dict, optimizer_obj, metrics: dict):
        self.loss_objs, self.optimizer_obj, self.metrics = loss_objs, optimizer_obj, metrics

    @tf.function
    def train_single_batch(self, x, y):
        with tf.GradientTape() as tape:
            preds, s3, s4, map11, map12, map13, map21, map22, map23, map31, map32, map33 = self.model(x, training=True)
            custom_loss_value = self.loss_objs['custom_loss'](y, preds)
            dice_loss_value = self.loss_objs['dice_loss'](y, preds)
            hausdorff_loss_value = self.loss_objs['hausdorff_loss'](preds, y) / 30
            loss_value = custom_loss_value + hausdorff_loss_value

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer_obj.apply_gradients(zip(grads, self.model.trainable_weights))
        return preds, loss_value, dice_loss_value, hausdorff_loss_value, custom_loss_value

    def train_single_epoch(self, data):
        losses = []
        dice_scores = []
        hausdorff_losses = []
        custom_losses = []

        for step in range(1, self.t_steps_per_epoch + 1):
            x, y = next(data)
            preds, loss_value, dice_loss_value, hausdorff_loss_value, custom_loss_value = self.train_single_batch(x, y)

            losses.append(loss_value)
            hausdorff_losses.append(hausdorff_loss_value)
            custom_losses.append(custom_loss_value)
            dice_score = self.metrics['train_dice'](y, preds)
            if tf.reduce_max(y * preds) > 0:
                dice_scores.append(dice_score)

        return losses, dice_scores, hausdorff_losses, custom_losses

    def validate_single_epoch(self, data):
        losses = []
        val_dice_scores = []

        for step in range(self.v_steps_per_epoch):
            x, y = next(data)
            preds, s4, s3, map11, map12, map13, map21, map22, map23, map31, map32, map33 = self.model(x)

            loss_value = self.loss_objs['custom_loss'](y, preds)
            losses.append(loss_value)

            val_dice_score = self.metrics['val_dice'](y, preds)
            if tf.reduce_max(y * preds) > 0:
                val_dice_scores.append(val_dice_score)

        return losses, val_dice_scores

    def fit(self, train_data, val_data, epochs):
        train_data_iter = iter(train_data)
        val_data_iter = iter(val_data)

        history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'hausdorff_loss': [], 'custom_loss': []}
        for epoch in range(epochs):
            train_losses, train_dice_scores, hausdorff_losses, custom_losses = self.train_single_epoch(train_data_iter)
            train_dice_result = np.mean(train_dice_scores)
            train_hausdorff_loss = np.mean(hausdorff_losses)
            train_custom_loss = np.mean(custom_losses)
            history['train_loss'].append(np.mean(train_losses))
            history['train_dice'].append(train_dice_result)
            history['hausdorff_loss'].append(train_hausdorff_loss)
            history['custom_loss'].append(train_custom_loss)

            val_losses, val_dice_scores = self.validate_single_epoch(val_data_iter)
            val_dice_result = np.mean(val_dice_scores)
            history['val_loss'].append(np.mean(val_losses))
            history['val_dice'].append(val_dice_result)

            print(f'\nEpoch {epoch + 1}: Train loss: {np.mean(train_losses):.4f}, '
                  f'Validation Loss: {np.mean(val_losses):.4f}, '
                  f'Train Dice: {train_dice_result:.4f}, '
                  f'Validation Dice: {val_dice_result:.4f}, '
                  f'Train Custom Loss: {train_custom_loss:.4f}, '
                  f'Train Hausdorff Loss: {train_hausdorff_loss:.4f}')

            self.cur_epoch += 1
            self.prev_model = self.model

            if self.best_model['score'] < val_dice_result:
                print("Storing new best model...")
                self.best_model['score'] = val_dice_result
                self.best_model['model'] = self.model
                self.best_model['metric_vals'] = {'val_dice': val_dice_result}

        history['model'] = self.model
        self.history = history

        return history

# Initialize model
model_wrapper = CustomModelWrapper(FunctionalModel().get())

# Compile the model
model_wrapper.compile(
    loss_objs={
        'custom_loss': HybridLoss(),
        'dice_loss': DiceLoss(),
        'hausdorff_loss': HausdorffLoss()
    },
    optimizer_obj=Nadam(learning_rate=0.0003),
    metrics={
        'train_dice': DiceCoeff(),
        'val_dice': DiceCoeff(),
        'train_f1': F1Score(),
        'val_f1': F1Score(),
        'train_recall': Recall(),
        'val_recall': Recall(),
        'train_precision': Precision(),
        'val_precision': Precision()
    }
)

# Fit the model
history = model_wrapper.fit(
    train_dataset,
    test_dataset,
    epochs=50
)
