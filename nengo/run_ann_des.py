
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

test_de = DE(name="test_de", input_min=-2., input_max=2., eq=lambda df_dx, f, x: df_dx + 2. * x * f, order=1, ic_x=[0], ic_y=[1], solution=lambda x: tf.exp(-inputs**2))
equations.append(test_de)

# # Solution missing for this test
# R = 1.
# test_de1 = DE(input_min=0., input_max=1., eq=lambda df_dx, f, x: df_dx - R * x * (1 - x), order=1, ic_x=0, ic_y=1, solution=lambda x: None)
# equations.append(test_de1)

# non-linear bernouulli
logistic_equation = DE(input_min=-2., input_max=2., eq=lambda df_dx, f, x: df_dx - f * (1-f), order=1, ic_x=[0], ic_y=[0.5], solution=lambda x: tf.sigmoid(x))
equations.append(logistic_equation)


# # linear second order
# # Simple Harmonic Motion (of spring) / Newton's Second Law
# m = 10.
# k = 1.
# newtons_second = DE(input_min=-2., input_max=2., eq=lambda df_dx, df_dxx, f, x: m*df_dxx + k*f, order=2, ic_x=0, ic_y=1, solution=lambda x: None)
# equations.append(newtons_second)

# # Kirchhoff’s law
# # dürfte keinen Sinn ergeben, da zwei verschränkte Gleichungen verwendet werden: E(t) & I(t)
# # pürfen, ob Ergebnis plausibel
# L = 4
# R = 12
# E_t = 60
# kirchhoff = DE(input_min=-2., input_max=2., eq=lambda dI_dt, I, t: L * dI_dt + R * I - E_t, order=1, ic_x=1, ic_y=4.75, solution=lambda x: tf.exp(-inputs**2))
# equations.append(kirchhoff)

# # Newtons first Law of cooling
# # auch hier sind zu viele Bedingungen zu erfüllen
# # richtige Lösung wird nicht angezeigt
# k = 0.092
# M = 25
# C = 4.36
# t = 6
# newtons_first = DE(input_min=-2., input_max=2., eq=lambda dT, k, M, T: dT - k * M + k * T, order=1, ic_x=0, ic_y=1, solution=lambda dT, k, M, T: M - (tf.exp(-C) * tf.exp(-k * t)))
# equations.append(newtons_first)

# x^2y′′+3xy′+4y=0
# defintionylücke bei y(0)
c_1 = 5
c_2 = 3
second_order = DE(name="second_order", input_min=-2., input_max=2., eq=lambda dy_dx, dy_dxx, y, x: x * tf.math.pow(x, 2 * dy_dxx) + 3 * x * dy_dx + 4 * y, order=2, ic_x=[1, 2.476632271], ic_y=[5, 0.4037741136], solution=lambda x: c_1 * (1 / x) * tf.math.cos(tf.sqrt(3)*tf.math.log(x))+ c_2 * (1/x) * tf.math.sin(tf.sqrt(3)*tf.math.log(x)) - y)
equations.append(second_order)

#prüfen, ob funktioniert wegen t
# t undefiniert, könnte zu Probemen führen. sind aber atsächlich  gleicvhverteielte t-Werte
second_2 = DE(name="second_2", input_min=-2., input_max=2., eq=lambda df_dxx, df_dx, x, t: df_dxx + x, order=2, ic_x=[0, 0.6366197724], ic_y=[1, 1], solution=lambda x: tf.cos(t) + sin(t) - x)
equations.append(second_2)


# third_order, y''' - 9y'' + 15y' + 25y = 0
# x undefiniert, könnte zu Probemen führen. sind aber atsächlich  gleicvhverteielte x-Werte
third_order = DE(name="third_order", input_min=0., input_max=1., eq=lambda dy_dt, dy_dtt, dy_dttt, y : dy_dttt - 9* dy_dtt + 15 * dy_dt + 25 * y, order=3, ic_x=[0, 1, -1], ic_y=[3, 297.1941976, 2.718281828], solution=lambda dy_dt, dy_dtt, dy_dttt, y, x:  tf.math.exp(-x) + tf.math.exp(5 * x) + x *  tf.math.exp(5 * x) - y)
equations.append(third_order)

# third_order_2, y'''+y''-2y=e^x(14+34x+15x^2)
# x undefiniert, könnte zu Probemen führen. sind aber atsächlich  gleicvhverteielte x-Werte
third_order_2 = DE(name="third_order_2", input_min=0., input_max=1., eq=lambda dy_dt, dy_dtt, dy_dttt, y, x : dy_dttt + dy_dtt - 2 * y - tf.math.exp(14 + 34 * x + 15 * tf.math.pow(x, 2)), order=3, ic_x=[0, 1.570796327, 1], ic_y=[2, 35.53210822, 8.529089278], solution=lambda dy_dt, dy_dtt, dy_dttt, y, x:  tf.math.exp(x) + math.exp((-x)) * (tf.math.cos(c) + tf.math.sind(x)) + math.exp(x) * (tf.math.pow(x, 2) + tf.math.pow(x, 3))  - y)
equations.append(third_order_2)


# nonline y' = x(y^3) where y(0)=2
nonline = DE(name="nonline", input_min=-2., input_max=2., eq=lambda df_dx, y, x: x * tf.math.pow(y, 3) - y, order=1, ic_x=[0], ic_y=[2], solution=lambda df_dx, y, x: tf.math.pow((1/4 - tf.math.pow(x, 2)), -0.5) - y)
equations.append(nonline)


# third_order_nonlin, y′′′+(y′)^2−yy′′=0 
# x undefiniert, könnte zu Probemen führen. sind aber atsächlich  gleicvhverteielte x-Werte
third_order_nonlin = DE(name="third_order_nonlin", input_min=0., input_max=1., eq=lambda dy_dt, dy_dtt, dy_dttt, y, x : dy_dttt + tf.math.pow(dy_dt, 2) - y * dy_dtt, order=3, ic_x=[0, 1, 2], ic_y=[1, 2.08616127, 6.524391382], solution=lambda dy_dt, dy_dtt, dy_dttt, y, x:  tf.math.exp(x) + math.exp(-x) -1  - y)
equations.append(third_order_nonlin)


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

