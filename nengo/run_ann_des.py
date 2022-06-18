
import tensorflow as tf
from matplotlib import pyplot as plt

from ann import Model1, loss_func2, training_step, loss_func1, training_step2

# set hyperparameters
epochs = 1000
learning_rate = 0.01

# initialize model, loss and optimizer
model = Model1(10)


class DE:
    def __init__(self, input_min, input_max, eq, order, ic_x, ic_y, solution):
        """
        Creates an object containing all necessary information for solving a DE with a NN and evaluating the solution.

        :param float input_min:
        :param float input_max:
        :param eq: lambda function specifying the equation with parameters df_dx, f, x – "lambda df_dx, f, x: …" – with df_dx
        :param int order: the order of the equation (i.e. the highest order of derivative it contains), e.g. 2 for a second order DE. Currently supports only equations of order 1 - 4
        :param float ic_x: x value of the initial condition, e.g. for the initial condition f(x=0)=1 this is 0
        :param float ic_y: y value of the initial condition, e.g. for the initial condition f(x=0)=1 this is 1
        :param solution: lambda function of the solution, containing one parameter x, i.e. "lambda x: …"
        """
        # eq = lambda dfdx, f, x: df_dx + 2. * x * f
        if order < 0 or order > 4:
            raise ValueError("Only equations of order 1 - 4 are supported")
        self.input_min = input_min
        self.input_max = input_max
        self.eq = eq
        self.order = order
        self.ic_x = ic_x
        self.ic_y = ic_y
        self.solution = solution

    def get_inputs(self, number_points): # TODO: adapt nb of expand_dims depending on network type
        inputs = tf.linspace(self.input_min, self.input_max, num=number_points)
        inputs = tf.expand_dims(inputs, -1)
        return inputs

    def analytical_soulution(self, x):
        return self.solution(x)

    def get_loss_function(self, network_type="ann"):
        if network_type == "ann":
            return self.__make_ann_loss_func()

        elif network_type == "snn":
            pass
        else:
            raise ValueError("network_type should be one of 'ann' or 'snn'")

    def __make_ann_loss_func(self):

        def ann_loss_function(y_pred, y_true):
            x=y_true
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

                ic = model(tf.constant(self.ic_x, shape=(1, 1))) - self.ic_y
            return tf.math.reduce_mean(tf.square(eq)) + tf.square(ic)

        return ann_loss_function


equations = []

test_de = DE(input_min=-2., input_max=2., eq=lambda df_dx, f, x: df_dx + 2. * x * f, order=1, ic_x=0, ic_y=1, solution=lambda x: tf.exp(-inputs**2))
equations.append(test_de)

# Solution missing for this test
R = 1.
test_de1 = DE(input_min=0., input_max=1., eq=lambda df_dx, f, x: df_dx - R * x * (1 - x), order=1, ic_x=0, ic_y=1, solution=lambda x: None)
equations.append(test_de1)

logistic_equation = DE(input_min=-2., input_max=2., eq=lambda df_dx, f, x: df_dx - f * (1-f), order=1, ic_x=0, ic_y=0.5, solution=lambda x: tf.sigmoid(x))
equations.append(logistic_equation)


# linear second order
# Simple Harmonic Motion (of spring) / Newton's Second Law
m = 10.
k = 1.
newtons_second = DE(input_min=-2., input_max=2., eq=lambda df_dx, df_dxx, f, x: m*df_dxx + k*f, order=2, ic_x=0, ic_y=1, solution=lambda x: None)
equations.append(newtons_second)




for de in equations:
    model = Model1(10)
    loss_function = de.get_loss_function("ann")
    print(de.get_loss_function("ann"))
    inputs = de.get_inputs(401)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    # We train for epochs.
    for epoch in range(epochs):
        # training (and checking in with training)
        epoch_loss_agg = []

        if epoch % 100 == 0:
            f = model(inputs)
            print(f'Epoch: {str(epoch)} starting with loss {loss_function(f, inputs)}')
        train_loss = training_step2(model, inputs, loss_function, optimizer)
        train_losses.append(tf.squeeze(train_loss))

    print(train_losses[0])
    # Visualize accuracy and loss for training and test data.
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.show()

    approx = model(inputs)
    plt.plot(tf.squeeze(inputs), tf.squeeze(approx), label="model's solution")
    try:
        plt.plot(tf.squeeze(inputs), tf.squeeze(de.analytical_soulution(tf.squeeze(inputs))), label="true solution")
    except ValueError:
        pass
    plt.legend()
    plt.show()

