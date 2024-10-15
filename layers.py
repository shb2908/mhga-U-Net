import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, UpSampling2D, Concatenate, Multiply, Activation, BatchNormalization
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="MyLayers")
class SqueezeExcitation(Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        c = input_shape[-1]
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.squeeze_conv = Conv2D(
            filters=c // self.reduction_ratio,
            kernel_size=(1, 1),
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False
        )
        self.excitation_conv = Conv2D(
            filters=c,
            kernel_size=(1, 1),
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(SqueezeExcitation, self).build(input_shape)

    def call(self, inputs):
        x = self.global_pooling(inputs)
        x = tf.reshape(x, (-1, 1, 1, inputs.shape[-1]))
        x = self.squeeze_conv(x)
        x = self.excitation_conv(x)
        return inputs * x

    def get_config(self):
        config = super(SqueezeExcitation, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight_layer = Conv2D(
            filters=input_shape[-1],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same"
        )
        super(WeightLayer, self).build(input_shape)

    def call(self, inputs):
        w = self.weight_layer(inputs)
        weighted_inp = tf.multiply(inputs, w)
        return weighted_inp

class NewGraphLayer:
    def __init__(self, n):
        self.n = n
        self.mask_matrix = tf.tile(tf.eye(n), [1, n])
        self.unmask_matrix = tf.constant([[int(j // n == i) for j in range(n ** 2)] for i in range(n)], dtype=tf.float32)
        self.unflatten_mat = tf.transpose(tf.reshape(tf.eye(n) * tf.expand_dims(tf.expand_dims(tf.eye(n), axis=-1), axis=-1), (n, n*n, n)), perm=[2, 1, 0])

    def custom_flatten(self, A):
        n = A.shape[1]
        B = tf.transpose(A, perm=[0, 3, 1, 2])
        B = (B @ self.mask_matrix) * self.unmask_matrix
        B = tf.reduce_sum(B, axis=-2)
        C = tf.transpose(B, perm=[0, 2, 1])
        return C

    def custom_unflatten(self, A):
        C = tf.transpose((tf.matmul(A, self.unflatten_mat)), perm=[1, 2, 0])
        return C

    def sum_channels(self, flattened_nodes):
        x = tf.reduce_sum(flattened_nodes, axis=1)
        x = tf.expand_dims(x, axis=1)
        return x

    def compute_dot_products(self):
        dot_products = tf.reduce_sum(tf.multiply(self.node_features, self.summed_vectors), axis=-1)
        return dot_products

    def prune_channel_by_variance(self, feature_map):
        variance_per_channel_vector = tf.math.reduce_variance(feature_map, axis=(1, 2), keepdims=True)
        mean_variance_per_sample = tf.reduce_mean(variance_per_channel_vector, axis=(1, 2, 3), keepdims=True)
        pruning_mask = tf.where(variance_per_channel_vector > mean_variance_per_sample, 1.0, 0.0)
        pruned_feature_map = feature_map * pruning_mask
        return pruned_feature_map

    def create_graph_map(self, dot_products):
        map = self.custom_unflatten(dot_products)
        max_values = tf.reduce_max(map, axis=(1, 2), keepdims=True)
        map = tf.cast(map, tf.float32) / (max_values + 1e-8)
        return 3.0 * map

    def __call__(self, input_data):
        self.input_data = input_data
        self.pruned_data = self.prune_channel_by_variance(input_data)
        self.node_features = self.custom_flatten(self.pruned_data)
        self.summed_vectors = self.sum_channels(self.node_features)
        dot_products = self.compute_dot_products()
        map = self.create_graph_map(dot_products)
        return map

@register_keras_serializable(package="MyLayers")
class GraphAttentionLayer(Layer):
    def __init__(self, n, **kwargs):
        self.n = n
        super(GraphAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.phi_g = Conv2D(
            filters=input_shape[-1],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same"
        )
        self.theta_x = Conv2D(
            filters=input_shape[-1],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same"
        )
        self.wl = WeightLayer()
        self.concatenate = Concatenate(axis=-1)
        self.result = Conv2D(
            filters=input_shape[-1],
            kernel_size=(1, 1),
            padding="same"
        )
        self.multiply = Multiply()
        self.bn = BatchNormalization()
        self.up = UpSampling2D(size=(2, 2))
        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, input_x, input_g):
        _, h, w, c = input_x.shape
        x = tf.reshape(input_x, (tf.shape(input_x)[0], h // 2, w // 2, c * 4))
        x = self.theta_x(x)
        g = self.phi_g(input_g)

        concat_inputs = self.concatenate([x, g])
        concat_inputs_wl = self.wl(concat_inputs)
        concat_inputs_activated = Activation('relu')(concat_inputs_wl)

        a = NewGraphLayer(self.n)
        map = a(concat_inputs_activated)
        map_expanded = tf.expand_dims(map, axis=-1)
        map_expanded_ = Activation('sigmoid')(map_expanded)
        map_upsampled = self.up(map_expanded_)

        y = self.multiply([input_x, map_upsampled])
        y_res = self.result(y)
        y_bn = self.bn(y_res)
        return y_bn, map_upsampled

@register_keras_serializable(package="MyLayers")
class GroupNormalization(Layer):
    def __init__(self, groups=1, epsilon=1e-5, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        assert input_shape[-1] % self.groups == 0
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.shift = self.add_weight(
            name='shift',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        grouped_inputs = tf.reshape(inputs, [batch_size, height, width, self.groups, channels // self.groups])
        mean = tf.reduce_mean(grouped_inputs, axis=[1, 2, 4], keepdims=True)
        variance = tf.reduce_mean(tf.square(grouped_inputs - mean), axis=[1, 2, 4], keepdims=True)
        normalized = (grouped_inputs - mean) / tf.sqrt(variance + self.epsilon)
        normalized = tf.reshape(normalized, [batch_size, height, width, channels])
        output = self.scale * normalized + self.shift
        return output

    def get_config(self):
        config = super(GroupNormalization, self).get_config()
        config.update({
            'groups': self.groups,
            'epsilon': self.epsilon,
        })
        return config

def CombinedUpsampleLayer(inputs):
    _, H, W, C = inputs.shape
    combined = UpSampling2D(size=(2, 2), interpolation="lanczos5")(inputs)
    combined_attn = Conv2D(C, 1, padding="same")(combined)
    upsampled = Conv2DTranspose(C, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([combined_attn, upsampled])
    return x
