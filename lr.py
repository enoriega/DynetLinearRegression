import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

import dynet as dy

def generate_data(samples = 5000):
    x = 20*random.rand(samples)
    e = 7*random.randn(samples)

    return x, 12*x+5+e


def mse_loss(predictions, target):
    diff = predictions - target
    square = dy.square(diff)
    mean = dy.mean_elems(square)

    return mean


m = dy.ParameterCollection()

W = m.add_parameters((1, 1))
b = m.add_parameters((1, ))

dy.renew_cg()

optimizer = dy.RMSPropTrainer(m)

BATCH_SIZE = 250
EPOCHS = 20000
TARGET_UPDATE = 1

x, y = generate_data()

# Training loop
losses = list()
for epoch in range(EPOCHS):
    # Sample a minibatch
    indices = random.choice(5000, BATCH_SIZE, False)
    mb_x, mb_y = x[indices], y[indices]

    mb_x = x
    mb_y = y

    # Renew the computational graph


    input = dy.inputTensor(mb_x.reshape((1, -1)))
    output = dy.inputTensor(mb_y.reshape(1, -1))

    y_pred = W*input+b

    loss = mse_loss(y_pred, output)

    losses.append(loss)


    avg_loss = dy.average(losses)
    avg_loss.forward()
    avg_loss.backward()
    optimizer.update()
    if epoch % 1000 == 0:
        print("Epoch %i avg loss: %f" % (epoch, avg_loss.value()))
    dy.renew_cg()
    losses = list()

domain = np.linspace(0, 20, 1000)

print(W.value(), b.value())

exit(1)

# dy.renew_cg()
#
# response = W*dy.inputTensor(domain.reshape((1, -1)))+b

# plt.figure()
# plt.scatter(x, y)
# plt.plot(domain, response.npvalue().reshape((-1,)), color='red')
# plt.show()
