import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from ann import Model, training_step


# class for simple DE implementation
class DE:
    def __init__(self, name, input_min, input_max, eq, order, ic_x, ic_y, solution):
        """
        Creates an object containing all necessary information for solving a DE with a NN and evaluating the solution.

        :param str name: The name of the equation.
        :param float input_min:
        :param float input_max:
        :param eq: lambda function specifying the equation with parameters df_dx, f, x – "lambda df_dx, f, x: …" – with df_dx
        :param int order: the order of the equation (i.e. the highest order of derivative it contains), e.g. 2 for a second order DE. Currently supports only equations of order 1 - 4
        :param list[float] ic_x: list of x values of the initial conditions, e.g. for the initial conditions f(x=0)=1 and f(2)=3 this is [0., 2.]
        :param list[float] ic_y: list of y values of the initial conditions, e.g. for the initial conditions f(x=0)=1 and f(2)=3 this is [1., 3.]
        :param solution: lambda function of the solution, containing one parameter x, i.e. "lambda x: …"
        """
        # raise error if order is out of currently supported range
        if order < 0 or order > 4:
            raise ValueError("Only equations of order 1 - 4 are supported")

        self.name = name
        self.input_min = input_min
        self.input_max = input_max
        self.eq = eq
        self.order = order
        self.ic_x = ic_x
        self.ic_y = ic_y
        self.solution = solution

    def get_inputs(self, number_points):
        inputs = tf.linspace(self.input_min, self.input_max, num=number_points)
        inputs = tf.expand_dims(inputs, -1)
        return inputs

    def analytical_solution(self, x):
        return self.solution(x)

    def get_loss_function(self):
        return self.__make_ann_loss_func()

    def __make_ann_loss_func(self):

        def ann_loss_function(y_pred, y_true):
            x = y_true
            with tf.GradientTape() as tape4:
                with tf.GradientTape() as tape3:
                    with tf.GradientTape() as tape2:
                        with tf.GradientTape() as tape1:
                            tape1.watch(x)
                            tape2.watch(x)
                            tape3.watch(x)
                            tape4.watch(x)
                            f = model(x)
                            df_dx = tape1.gradient(f, x)
                            if self.order == 1:
                                eq = self.eq(df_dx, f, x)
                            if self.order > 1:
                                df_dxx = tape2.gradient(df_dx, x)
                                if self.order == 2:
                                    eq = self.eq(df_dx, df_dxx, f, x)
                            if self.order > 2:
                                df_dxxx = tape3.gradient(df_dxx, x)
                                if self.order == 3:
                                    eq = self.eq(df_dx, df_dxx, df_dxxx, f, x)
                            if self.order > 3:
                                df_dxxxx = tape4.gradient(df_dxxx, x)
                                if self.order == 4:
                                    eq = self.eq(df_dx, df_dxx, df_dxxx, df_dxxxx, f, x)

                ic = 0.0
                for ic_x, ic_y in zip(self.ic_x, self.ic_y):
                    ic += tf.square(model(tf.constant(ic_x, shape=(1, 1))) - ic_y)

            return tf.math.reduce_mean(tf.square(eq)) + ic

        return ann_loss_function


# initializing the DEs and storing them in a list
equations = []

test_de = DE(name="test_de", input_min=-2., input_max=2., eq=lambda df_dx, f, x: df_dx + 2. * x * f, order=1, ic_x=[0], ic_y=[1], solution=lambda x: tf.exp(-inputs**2))
equations.append(test_de)

# Solution missing for this test
R = 1.
test_de1 = DE(name="test_de1", input_min=0., input_max=1., eq=lambda df_dx, f, x: df_dx - R * x * (1 - x), order=1, ic_x=[0], ic_y=[1], solution=lambda x: None)
equations.append(test_de1)

logistic_equation = DE(name="logistic_equation", input_min=-2., input_max=2., eq=lambda df_dx, f, x: df_dx - f * (1-f), order=1, ic_x=[0], ic_y=[0.5], solution=lambda x: tf.sigmoid(x))
equations.append(logistic_equation)


# linear second order
# Simple Harmonic Motion (of spring) / Newton's Second Law
m = 10.
k = 1.
newtons_second = DE(name="newtons_second", input_min=-2., input_max=2., eq=lambda df_dx, df_dxx, f, x: m*df_dxx + k*f, order=2, ic_x=[0], ic_y=[1], solution=lambda x: None)
equations.append(newtons_second)

# set the hyperparameters
epochs = 1000
learning_rate = 0.01
loss_threshold = 0.00001

# lists for storing results
final_losses = []
final_errors = []
de_names = []

first_epoch_under_threshold = []
time_to_threshold = []
total_training_time = []


rmse = tf.keras.metrics.RootMeanSquaredError()

# for all equations
for de in equations:
    # initialize the model and optimizer
    model = Model(10)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # get the loss function and inputs from the DE object
    loss_function = de.get_loss_function()
    inputs = de.get_inputs(401)

    # Initialize lists for later visualization.
    train_losses = []
    train_errors = []

    under_threshold = False

    start_time = time.process_time()

    # We train for epochs.
    for epoch in range(epochs):
        # print the current loss every 100 epochs to check if training is working
        if epoch % 100 == 0:
            f = model(inputs)
            print(f'Epoch: {str(epoch)} starting with loss {loss_function(f, inputs)}')

        # run training and store the loss
        train_loss = training_step(model, inputs, loss_function, optimizer)
        train_losses.append(tf.squeeze(train_loss))

        # calculate error and store it
        # only possible if solution is not None
        try:
            approx = tf.squeeze(model(inputs))
            solution = tf.squeeze(de.analytical_solution(tf.squeeze(inputs)))
            error = rmse(approx, solution).numpy()
            train_errors.append(error)
        except ValueError:
            pass

        # check if loss is under threshold
        if tf.squeeze(train_loss) < loss_threshold and not under_threshold:
            time_to_threshold.append(time.process_time() - start_time)
            first_epoch_under_threshold.append(epoch)
            under_threshold = True

    # store total runtime and None for time to threshold if it was not reached
    total_training_time.append(time.process_time() - start_time)
    if not under_threshold:
        time_to_threshold.append(None)
        first_epoch_under_threshold.append(None)

    # save final loss + error
    if train_errors:
        final_errors.append(train_errors[-1])
    else:
        final_errors.append(None)
    final_losses.append(train_losses[-1])
    de_names.append(de.name)

    # save the model
    model.save("models/"+de.name+"_model")

    plt.figure()
    plt.plot(train_losses, label="loss")
    if train_errors:
        plt.plot(train_errors, label="RMSE")
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Error")
    plt.legend()

    figname = "plots/" + de.name + "__loss_error.png"
    plt.savefig(figname)
    plt.show()

    # plot the model's approximation and the actual solution
    approx = model(inputs)
    plt.plot(tf.squeeze(inputs), tf.squeeze(approx), label="model's solution")
    try:
        plt.plot(tf.squeeze(inputs), tf.squeeze(de.analytical_solution(tf.squeeze(inputs))), label="true solution")
    except ValueError:
        pass
    plt.legend()

    figname = "plots/" + de.name + "__solution.png"
    plt.savefig(figname)
    plt.show()


# create an array with all final losses + errors and de names
final_losses = np.array(final_losses)
final_errors = np.array(final_errors)
de_names = np.array(de_names)
time_to_threshold = np.array(time_to_threshold)
total_training_time = np.array(total_training_time)
first_epoch_under_threshold = np.array(first_epoch_under_threshold)

metrics = np.array([de_names, final_losses, final_errors, first_epoch_under_threshold, time_to_threshold, total_training_time]).T

# save as file
np.savetxt("metrics.txt", metrics, fmt="%s", delimiter=",")

