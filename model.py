import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Activation
from tensorflow.keras.models import Model
from layers import SqueezeExcitation, GraphAttentionLayer, GroupNormalization, CombinedUpsampleLayer

class FunctionalModel:
    def __init__(self):
        self.pre_trained_backbone = tf.keras.applications.DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(512, 512, 3)
        )

    def conv_block(self, inputs, num_filters):
        x1 = Conv2D(num_filters // 2, 5, padding="same")(inputs)
        x1 = GroupNormalization()(x1)
        x1 = Activation("relu")(x1)

        x2 = Conv2D(num_filters // 2, 3, padding="same")(inputs)
        x2 = GroupNormalization()(x2)
        x2 = Activation("relu")(x2)

        x2 = Concatenate()([x1, x2])
        x2 = Conv2D(num_filters, 1, padding="same")(x2)

        x3 = Conv2D(num_filters, 1, padding="same")(inputs)
        x3 = GroupNormalization()(x3)
        x3 = Activation("relu")(x3)

        x3 = Concatenate()([x2, x3])

        x = Conv2D(num_filters, 3, padding="same")(x3)
        x = GroupNormalization()(x)
        x = Activation("relu")(x)

        return x

    def decoder_block(self, inputs, skip_features, num_filters):
        x = CombinedUpsampleLayer(inputs)
        skip_features = Activation("relu")(skip_features)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def get(self):
        inputs = self.pre_trained_backbone.input

        s1 = inputs
        s2 = self.pre_trained_backbone.get_layer('conv1_relu').output  # 256
        s3 = self.pre_trained_backbone.get_layer('pool2_relu').output  # 128
        s4 = self.pre_trained_backbone.get_layer('pool3_relu').output  # 64
        b = self.pre_trained_backbone.get_layer("pool4_relu").output   # 32

        x1 = SqueezeExcitation()(s4)
        G11, map11 = GraphAttentionLayer(n=32, name="Graph11_64")(x1, b)
        G12, map12 = GraphAttentionLayer(n=32, name="Graph12_64")(x1, b)
        G13, map13 = GraphAttentionLayer(n=32, name="Graph13_64")(x1, b)

        G1 = Concatenate(axis=-1)([G11, G12, G13])

        d1 = self.decoder_block(b, G1, 512)  # 64

        x2 = SqueezeExcitation()(s3)
        G21, map21 = GraphAttentionLayer(n=64, name="Graph21_64")(x2, d1)
        G22, map22 = GraphAttentionLayer(n=64, name="Graph22_64")(x2, d1)
        G23, map23 = GraphAttentionLayer(n=64, name="Graph23_64")(x2, d1)

        G2 = Concatenate(axis=-1)([G21, G22, G23])

        d2 = self.decoder_block(d1, G2, 256)  # 128

        x3 = SqueezeExcitation()(s2)
        G31, map31 = GraphAttentionLayer(n=128, name="Graph31_256")(x3, d2)
        G32, map32 = GraphAttentionLayer(n=128, name="Graph32_256")(x3, d2)
        G33, map33 = GraphAttentionLayer(n=128, name="Graph33_256")(x3, d2)

        G3 = Concatenate(axis=-1)([G31, G32, G33])

        d3 = self.decoder_block(d2, G3, 128)  # 256
        d4 = self.decoder_block(d3, s1, 64)   # 512

        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        model = Model(inputs=inputs, outputs=[outputs, s3, s4,
                                              map11, map12, map13,
                                              map21, map22, map23,
                                              map31, map32, map33])
        return model
