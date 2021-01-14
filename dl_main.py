import numpy as np
import dl_functions as dlf

# x, t = dlf.get_data()
# network = dlf.init_network_mnist()

(x_train, t_train), (x_test, t_test) = \
    dlf.load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch.size)
print(x_batch.shape)

print(dlf.numerical_gradient(dlf.function_2, np.array([3.0, 4.0])))
print(dlf.numerical_gradient(dlf.function_2, np.array([0.0, 2.0])))
print(dlf.numerical_gradient(dlf.function_2, np.array([3.0, 0.0])))
