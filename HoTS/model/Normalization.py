from keras.layers import *#Layer, Wrapper
import tensorflow as tf
from keras.initializers import Ones, Zeros
import keras.backend as K

_DEFAULT_WEIGHT_NAME = 'kernel'


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class MeanOnlyBatchNormalization(Layer):
    def __init__(self,
                 momentum=0.999,
                 moving_mean_initializer='zeros',
                 axis=-1,
                 **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.moving_mean_initializer = moving_mean_initializer
        self.axis = axis

    def build(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)

        super().build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]

        # inference
        def normalize_inference():
            return inputs - self.moving_mean

        if training in {0, False}:
            return normalize_inference()

        mean = K.mean(inputs, axis=reduction_axes)
        normed_training = inputs - mean

        self.add_update(K.moving_average_update(self.moving_mean, mean,
                                                self.momentum), inputs)

        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'moving_mean_initializer':
                self.moving_mean_initializer,
        }
        base_config = super(MeanOnlyBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightNormalization(Wrapper):
    """ Applies weight normalization to a layer. Weight normalization is a reparameterization that decouples the
    magnitude of a weight tensor from its direction. This speeds up convergence by improving the conditioning of the
    optimization problem.
    Reference: https://arxiv.org/abs/1602.07868
    Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    Tim Salimans, Diederik P. Kingma (2016)
    Arguments:
      layer: The `Layer` instance to be wrapped.
      weight_name: name of weights `Tensor` in wrapped layer to be normalized
    """

    def __init__(self, layer, weight_name=_DEFAULT_WEIGHT_NAME, **kwargs):
        self.weight_name = weight_name

        if not hasattr(layer, '_weight_norm'):
            layer._weight_norm = {weight_name}
        elif weight_name not in layer._weight_norm:
            layer._weight_norm.add(weight_name)
        else:
            raise ValueError(
                'Weight normalization already applied to parameter {} in layer {}'.format(weight_name, layer))

        self.g_name = '{}_g'.format(weight_name)
        self.v_name = '{}_v'.format(weight_name)

        super(WeightNormalization, self).__init__(layer, **kwargs)
        self.supports_masking = True
        #self._track_checkpointable(layer, name='layer')

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=tf.TensorShape(input_shape).as_list())

        if not self.layer.built:
            self.layer.build(input_shape)

        if hasattr(self.layer, self.g_name) or hasattr(self.layer, self.v_name):
            raise ValueError(
                'Weight normalization already applied to {} in layer {}'.format(self.weight_name, self.layer))

        if not hasattr(self.layer, self.weight_name):
            raise ValueError('Parameter {} not found in layer {}'.format(self.weight_name, self.layer))

        v = getattr(self.layer, self.weight_name)
        v = K.cast(v, K.floatx())
        weight_depth = v.shape[-1]
        norm_axes = list(range(v.shape.ndims - 1))

        def g_init(v_init):
            c = K.int_shape(v_init)
            v_init = K.reshape(v_init, [-1, c[-1]])
            v_init = K.sqrt(K.sum(v_init**2, axis=0, keepdims=False))
            return v_init

        self.layer.built = False
        g = self.layer.add_weight(
            name=self.g_name,
            shape=(weight_depth,),
            initializer=lambda *args, **kwargs: g_init(v),
            dtype=v.dtype,
            trainable=True,
            #aggregation=tf.VariableAggregation.MEAN
        )
        self.layer.built = True
        orig_shape = [1]*(v.shape.ndims - 1) + [weight_depth]
        w = K.l2_normalize(v, axis=norm_axes) * K.reshape(g, orig_shape)

        setattr(self.layer, self.g_name, g)
        setattr(self.layer, self.v_name, v)
        setattr(self.layer, self.weight_name, w)

        super(WeightNormalization, self).build()

    def call(self, inputs, **kwargs):
        return self.layer.call(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super(WeightNormalization, self).get_config()

        if _DEFAULT_WEIGHT_NAME != self.weight_name:
            config['weight_name'] = self.weight_name

        return config