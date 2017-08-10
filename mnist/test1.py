
import mnist_loader
import network

# Cargamos los datos
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# Creamos la red con una capa oculta de veinte neuronas
net = network.Network([784, 20, 10])

# Entrenamos la red utilizando descenso del gradiente
# epochs = 20
# mini_batch_size = 10
# eta = 3.0
net.SGD(training_data, 20, 10, 3.0, test_data=test_data)

