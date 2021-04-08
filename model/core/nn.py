import tensorflow as tf

class NeuralNetWork:
    def __init__(self,feature_number, rows, columns, layers,dtype=tf.float32):


        self.input_tensor = tf.placeholder(dtype, shape=[None, rows, columns, feature_number],name='feature_input')

        self.predicted_w = tf.placeholder(dtype, shape=[None, rows],name='predicted_actions')
        self._rows = rows
        self._columns = columns
        self.dtype = dtype
        self.layers_dict = {}
        self.layer_count = 0

        self.output = self._build_network(layers)

    def _build_network(self, layers):
        pass

class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self,feature_number, rows, columns, layers, dtype):
        NeuralNetWork.__init__(self, feature_number, rows, columns, layers, dtype)

    def add_layer_to_dict(self, layer_type, tensor, weights=True):

        self.layers_dict[layer_type + '_' + str(self.layer_count) + '_activation'] = tensor
        self.layer_count += 1
        self.training = True

    # grenrate the operation, the forward computaion
    def _build_network(self, layers):
        # [batch, assets, window, features]
        # data format channel last
        # network = network / network[:, :, -1, 0, None, None]
        network = self.input_tensor
        for layer_number, layer in enumerate(layers):
            if layer["type"] == "DenseLayer":
                network = tf.keras.layers.Dense(units = int(layer["neuron_number"]),
                                          activation = (lambda i: i or None)(layer["activation_function"]),
                                          kernel_regularizer=layer["regularizer"],
                                          dtype=self.dtype)(network)
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "DropOut":
                network = tf.layers.Dropout(rate = layer["keep_probability"])(network)
            elif layer["type"] == "EIIE_Dense":
                width = network.get_shape()[2]
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                           kernel_size = [1, width],
                                           strides = [1, 1],
                                            padding = "valid",
                                            activation = layer["activation_function"],
                                            kernel_regularizer=layer["regularizer"],
                                            dtype=self.dtype)(network)
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "ConvLayer":
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = allint(layer["filter_shape"]),
                                            strides = allint(layer["strides"]),
                                            padding = layer["padding"],
                                            activation = layer["activation_function"],
                                            kernel_regularizer=layer["regularizer"],
                                            dtype=self.dtype)(network)
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "MaxPooling":
                network = tf.keras.layers.MaxPooling2D(pool_size = [2,2],
                                                 strides = layer["strides"],
                                                 padding = 'valid')(network)
            elif layer["type"] == "AveragePooling":
                network = tf.keras.layers.AveragePooling2D(pool_size = [2,2],
                                                     strides = layers['strides'],
                                                     padding = 'valid')(network)
            elif layer["type"] == "BatchNormalization":
                network = tf.keras.layers.BatchNormalization()(network, training=self.training)

            elif layer["type"] == "Iutput_WithW":
                network = tf.keras.layers.Flatten()(network)
                network = tf.concat([network,self.predicted_w], axis=1)
                network = tf.keras.layers.Dense(units = layer["neuron_number"],
                                          activation = layer["activation_function"],
                                          kernel_regularizer = layer["regularizer"],
                                          dtype=self.dtype)(network)

            elif layer["type"] == "LSTM":
                network = network[:,0,:,:]
                network = tf.keras.layers.LSTM(units = int(layer["neuron_number"]),
                                               return_state = str2bool(layer["return_state"]),
                                               return_sequences = str2bool(layer["return_sequences"]),
                                               dtype=self.dtype)(network)
                self.add_layer_to_dict(layer["type"], network)
            # this layer should be added at the end of layers
            elif layer["type"] == 'MultiTaskLayer':
                Layers = []
                for i in range(self._rows):
                    Layers.append(tf.keras.layers.Dense(units = int(layer["neuron_number"]),
                                                        activation = layer["activation_function"],
                                                        kernel_regularizer=layer["regularizer"],
                                                        dtype=self.dtype)(network))
                network = Layers
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]

def str2bool(str):
    return str == "True"
