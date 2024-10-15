import tensorflow as tf
from keras_unet_collection import losses as unet_losses

smooth = 1e-3

class DiceCoeff:
    def __init__(self):
        pass

    def dice_coef(self, y_true, y_pred):
        y_true_flat = tf.keras.layers.Flatten()(y_true)
        y_pred_flat = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)

    def __call__(self, y_true, y_pred):
        return self.dice_coef(y_true, y_pred)

class DiceLoss:
    def __init__(self):
        pass

    def dice_loss(self, y_true, y_pred):
        return 1.0 - DiceCoeff()(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        return self.dice_loss(y_true, y_pred)

class HybridLoss:
    def __init__(self):
        pass

    def hybrid_loss(self, y_true, y_pred):
        loss_focal = unet_losses.focal_tversky(y_true, y_pred, alpha=0.3, gamma=4 / 3)
        loss_dice = DiceLoss()(y_true, y_pred)
        return loss_focal + loss_dice

    def __call__(self, y_true, y_pred):
        return self.hybrid_loss(y_true, y_pred)

class HausdorffLoss:
    def __init__(self):
        pass

    def euclidean_distance(self, x, y):
        return tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(x, 1) - tf.expand_dims(y, 0)), axis=-1))

    def hausdorff_distance(self, x, y):
        distances_x_to_y = self.euclidean_distance(x, y)
        distances_y_to_x = self.euclidean_distance(y, x)
        hausdorff_distance = tf.reduce_max(tf.reduce_min(distances_x_to_y, axis=1))
        hausdorff_distance = tf.maximum(hausdorff_distance, tf.reduce_max(tf.reduce_min(distances_y_to_x, axis=1)))
        return hausdorff_distance

    def hausdorff_loss(self, pmask, gtmask):
        pmask1 = tf.squeeze(pmask[0])
        pmask2 = tf.squeeze(pmask[1])
        gtmask1 = tf.squeeze(gtmask[0])
        gtmask2 = tf.squeeze(gtmask[1])

        loss1 = self.hausdorff_distance(pmask1, gtmask1)
        loss2 = self.hausdorff_distance(pmask2, gtmask2)

        return (loss1 + loss2) / 2

    def __call__(self, pmask, gtmask):
        return self.hausdorff_loss(pmask, gtmask)
