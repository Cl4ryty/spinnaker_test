import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


class Model1(tf.keras.Model):
    def __init__(self, ns=10):
        super(Model1, self).__init__()
        # initialize
        self.layer_list = [tf.keras.layers.Dense(units=ns, activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.random_normal_initializer(),
                                                 bias_initializer=tf.random_normal_initializer()),
                           tf.keras.layers.Dense(units=ns, activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.random_normal_initializer(),
                                                 bias_initializer=tf.random_normal_initializer()),
                           tf.keras.layers.Dense(units=1, activation=tf.nn.relu,
                                                 kernel_initializer=tf.random_normal_initializer(),
                                                 bias_initializer=tf.random_normal_initializer())

                           ]

    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x



def training_step(model, input, loss_function, optimizer):
    """
    Performs a training step of the model using the given imput and target,
    calculating the loss with the given function and then using the optimizer to optimize the model.

    :param tf.keras.Model model: the model to be trained
    :param tf.Tensor input: the input
    :param tf.Tensor target: the target
    :param tf.keras.losses.Loss loss_function: the loss function
    :param tf.keras.optimizers.Optimizer optimizer: the optimizer
    :return: loss - the loss for this training step
    :rtype: tf.Tensor
    """
    with tf.GradientTape() as tape:
        loss = loss_function(model, input)
        # print("calculated loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        # print("calculated gradients")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # print("applied gradients")
    return loss

def training_step2(model, input, loss_function, optimizer):
    """
    Performs a training step of the model using the given imput and target,
    calculating the loss with the given function and then using the optimizer to optimize the model.

    :param tf.keras.Model model: the model to be trained
    :param tf.Tensor input: the input
    :param tf.Tensor target: the target
    :param tf.keras.losses.Loss loss_function: the loss function
    :param tf.keras.optimizers.Optimizer optimizer: the optimizer
    :return: loss - the loss for this training step
    :rtype: tf.Tensor
    """
    with tf.GradientTape() as tape:
        f = model(input)
        loss = loss_function(f, input)
        # print("calculated loss")
        gradients = tape.gradient(loss, model.trainable_variables)
        # print("calculated gradients")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # print("applied gradients")
    return loss


def loss_func1(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        f = model(x)
        df_dx = tape.gradient(f, x)
        eq = df_dx + 2. * x * f
        ic = model(tf.constant(0.0, shape=(1, 1))) - 1.
    return tf.math.reduce_mean(tf.square(eq)) + tf.square(ic)




def loss_func2(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        f = model(x)
        df_dx = tape.gradient(f, x)
        eq = df_dx - 1. * x * (1 - x)
        ic = model(tf.constant(0.0, shape=(1, 1))) - 1.
    return tf.math.reduce_mean(tf.square(eq)) + tf.square(ic)



