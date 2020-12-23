import tensorflow as tf

class NeuralNetWork:
    def __init__(self,feature_number, rows, columns, layers):


        self.input_num = tf.placeholder(tf.int32, shape=[])
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, rows, columns, feature_number])
        self.previous_w = tf.placeholder(tf.float32, shape=[None, rows])
        self.predicted_w = tf.placeholder(tf.float32, shape=[None, rows])
        self._rows = rows
        self._columns = columns

        self.layers_dict = {}
        self.layer_count = 0

        self.output = self._build_network(layers)

    def _build_network(self, layers):
        pass

class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self,feature_number, rows, columns, layers):
        NeuralNetWork.__init__(self, feature_number, rows, columns, layers)

    def add_layer_to_dict(self, layer_type, tensor, weights=True):

        self.layers_dict[layer_type + '_' + str(self.layer_count) + '_activation'] = tensor
        self.layer_count += 1

    # grenrate the operation, the forward computaion
    def _build_network(self, layers):
        # [batch, assets, window, features]
        # data format channel last
        # network = network / network[:, :, -1, 0, None, None]
        network = self.input_tensor
        for layer_number, layer in enumerate(layers):
            if layer["type"] == "DenseLayer":
                network = tf.keras.layers.Dense(units = int(layer["neuron_number"]),
                                          activation = layer["activation_function"],
                                          kernel_regularizer=layer["regularizer"])(network)
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
                                            kernel_regularizer=layer["regularizer"])(network)
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "ConvLayer":
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = allint(layer["filter_shape"]),
                                            strides = allint(layer["strides"]),
                                            padding = layer["padding"],
                                            activation = layer["activation_function"],
                                            kernel_regularizer=layer["regularizer"])(network)
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
                network = tf.keras.layers.BatchNormalization(training = layer['training'])(network)
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape()[2]
                network = tf.keras.layers.Conv2D(filters = 1,
                                           kernel_size = [1, width],
                                           padding="valid",
                                           kernel_regularizer=layer["regularizer"])(network)
                self.add_layer_to_dict(layer["type"], network)
                network = network[:, :, 0, 0]
                btc_bias = tf.ones((self.input_num, 1))
                self.add_layer_to_dict(layer["type"], network)
                network = tf.concat([btc_bias, network], 1)
                network = tf.nn.softmax(network)
                self.add_layer_to_dict(layer["type"], network, weights=False)
            elif layer["type"] == "Iutput_WithW":
                network = tf.keras.layers.Flatten()(network)
                network = tf.concat([network,self.predicted_w], axis=1)
                network = tf.keras.layers.Dense(units = layer["neuron_number"],
                                          activation = layer["activation_function"],
                                          kernel_regularizer = layer["regularizer"])(network)
            elif layer["type"] == "EIIE_Output_WithW":
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                network = tf.concat([network, w], axis=3)
                network = tf.layers.Conv2D(filters = 1,
                                           kernel_size = [1, 1],
                                           padding="valid",
                                           kernel_regularizer=layer["regularizer"])(network)
                self.add_layer_to_dict(layer["type"], network)
                network = network[:, :, 0, 0]
                #btc_bias = tf.zeros((self.input_num, 1))
                btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32,
                                       initializer=tf.zeros_initializer)
                # self.add_layer_to_dict(layer["type"], network, weights=False)
                btc_bias = tf.tile(btc_bias, [self.input_num, 1])
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                self.add_layer_to_dict('voting', network, weights=False)
                network = tf.nn.softmax(network)
                self.add_layer_to_dict('softmax_layer', network, weights=False)

            elif layer["type"] == "EIIE_LSTM" or\
                            layer["type"] == "EIIE_RNN":
                network = tf.transpose(network, [0, 2, 3, 1])
                resultlist = []
                reuse = False
                for i in range(self._rows):
                    if i > 0:
                        reuse = True
                    if layer["type"] == "EIIE_LSTM":
                        with tf.variable_scope("lstm"+str(layer_number), reuse=reuse):
                            result = tf.keras.layers.LSTM(units = int(layer["neuron_number"]),
                                                         dropout=layer["dropouts"],
                                                         return_sequences=layer["return_sequences"],
                                                         return_state = layer['return_state'])(network[:, :, :, i])
                    else:
                        with tf.variable_scope("lstm"+str(layer_number), reuse=reuse):
                            result = tf.keras.layers.SimpleRNN(units = int(layer["neuron_number"]),
                                                           dropout=layer["dropouts"],
                                                           return_sequences=layer["return_sequences"],
                                                           return_state = layer['return_state'])(network[:, :, :, i])
                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, self._rows, 1, int(layer["neuron_number"])])
            elif layer["type"] == "LSTM":
                network = network[:,0,:,:]
                network = tf.keras.layers.LSTM(units = int(layer["neuron_number"]),
                                               return_state = str2bool(layer["return_state"]),
                                               return_sequences = str2bool(layer["return_sequences"]))(network)
                self.add_layer_to_dict(layer["type"], network)
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]

def str2bool(str):
    return str == "True"
