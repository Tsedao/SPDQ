import tensorflow as tf
import numpy as np

def cal_layer_num(length,kernel_size,dilation):
    """
    Calculate the minimum number of res-layers to capture the whole time-series information
    Source: https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4
    """
    if dilation == 1:
        return int(np.ceil((length-1)/((kernel_size-1)*2)))
    else:
        return int(np.ceil(np.log(((length-1)*(dilation-1))/((kernel_size-1)*2)+1) / np.log(dilation)))

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class NeuralNetWork:
    def __init__(self,feature_number, rows, columns, layers,dtype=tf.float32):

        # we concat the original stock price together with its portfolio weigths
        self.input_tensor = tf.placeholder(dtype, shape=[None, rows, columns, feature_number+1],name='feature_input')
        self.predicted_w = tf.placeholder(dtype, shape=[None, rows],name='predicted_actions')
        self._rows = rows
        self._columns = columns
        self.dtype = dtype
        self.layers_dict = {}
        self.layer_count = 0
        self.training = True
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

            elif layer["type"] == "ConvLayer":
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = allint(layer["filter_shape"]),
                                            strides = allint(layer["strides"]),
                                            padding = layer["padding"],
                                            activation = layer["activation_function"],
                                            kernel_regularizer=layer["regularizer"],
                                            dtype=self.dtype)(network)
                self.add_layer_to_dict(layer["type"], network)
            elif layer['type'] == "EIIE_Dense":
                width = network.get_shape().as_list()[2]
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                                 kernel_size = [1, width],
                                                 strides = [1, 1],
                                                 padding = "valid",
                                                 activation = layer["activation_function"],
                                                 kernel_regularizer=layer["regularizer"])(network)
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape().as_list()[2]
                batch_num = network.get_shape().as_list()[0]
                network = tf.keras.layers.Conv2D(filters = 1,
                                                 kernel_size = [1, width],
                                                 padding="valid",
                                                 kernel_regularizer=layer["regularizer"])(network)
                network = network[:, :, 0, 0]
                network = tf.keras.activations.softmax(network)
            elif layer["type"] == "EIIE_Output_WithW":
                width = network.get_shape().as_list()[2]
                height = network.get_shape().as_list()[1]
                features = network.get_shape().as_list()[3]

                network = tf.reshape(network, [-1, int(height), 1, int(width*features)])
                w = tf.reshape(self.predicted_w, [-1, int(height), 1, 1])
                network = tf.concat([network, w], axis=3)
                network = tf.keras.layers.Conv2D(filters = 1,
                                                 kernel_size = [1, 1],
                                                 padding="valid",
                                                 kernel_regularizer=layer["regularizer"])(network)
                network = network[:, :, 0, 0]

            elif layer["type"] == "ResLayer":
                identity = network
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = allint(layer["filter_shape"]),
                                            strides = [1,1],
                                            padding = "same",
                                            dtype=self.dtype)(network)
                network = tf.keras.layers.BatchNormalization()(network, training=self.training)
                network = tf.keras.activations.relu(network)
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = allint(layer["filter_shape"]),
                                            strides = [1,1],
                                            padding = "same",
                                            dtype=self.dtype)(network)

                network = tf.keras.layers.BatchNormalization()(network, training=self.training)
                network = network + identity
                network = tf.keras.activations.relu(network)
            elif layer["type"] == "Bottleneck":
                identity = network

                downsample = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = allint(layer["filter_shape"]),
                                            strides = [2,1],
                                            padding = "valid",
                                            dtype=self.dtype)(identity)
                downsample = tf.keras.layers.BatchNormalization()(downsample, training=self.training)

                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = [1,1],
                                            strides = [1,1],
                                            padding = "same",
                                            dtype=self.dtype)(network)
                network = tf.keras.layers.BatchNormalization()(network, training=self.training)
                network = tf.keras.activations.relu(network)
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = allint(layer["filter_shape"]),
                                            strides = [2,1],
                                            padding = "valid",
                                            dtype=self.dtype)(network)
                network = tf.keras.layers.BatchNormalization()(network, training=self.training)
                network = tf.keras.activations.relu(network)
                network = tf.keras.layers.Conv2D(filters = int(layer["filter_number"]),
                                            kernel_size = [1,1],
                                            strides = [1,1],
                                            padding = "same",
                                            dtype=self.dtype)(network)
                network = tf.keras.layers.BatchNormalization()(network, training=self.training)
                network = network + downsample
                network = tf.keras.activations.relu(network)

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

            elif layer["type"] == 'TemporalConvNets':
                layer_num = cal_layer_num(self._columns,layer["kernel_size"],layer["dilation_rate"])
                init_filter_num, dilation = layer["filters"], layer["dilation_rate"]
                dropout = layer["dropout_rate"]
                kernel_size = layer["kernel_size"]
                outs = []
                for i in range(self._rows):
                    in_asset = network[:,i,...]

                    for k in range(layer_num):
                        out_tmp = tf.keras.layers.SeparableConv1D(filters=init_filter_num*(k+1),
                                                                    kernel_size=kernel_size,strides=1,padding='causal',
                                                                  dilation_rate=dilation,data_format='channels_last')(in_asset)
                        out_tmp = tf.keras.activations.relu(out_tmp)
                        out_tmp = tf.keras.layers.Dropout(rate=dropout)(out_tmp,training=self.training)


                        out_tmp = tf.keras.layers.SeparableConv1D(filters=init_filter_num*(k+1),
                                                                   kernel_size=kernel_size,strides=1,padding='causal',
                                                                  dilation_rate=dilation,data_format='channels_last')(out_tmp)
                        ## TO DO add weight-norm or regularization

                        out_tmp = tf.keras.activations.relu(out_tmp)
                        out_tmp = tf.keras.layers.Dropout(rate=dropout)(out_tmp,training=self.training)

                        upsampling = tf.keras.layers.Conv1D(filters=init_filter_num*(k+1),kernel_size=1,strides=1,padding='causal')(in_asset)

                        out_tmp = out_tmp + upsampling

                        in_asset = out_tmp

                    out_tmp = tf.keras.layers.Dense(units=1)(out_tmp)
                    out_tmp = tf.expand_dims(tf.squeeze(out_tmp,axis=-1),axis=1)
                    outs.append(out_tmp)
                tcn_out = tf.concat(outs,axis=1)
                network = tcn_out
            elif layer["type"] == "TCN_ConcatW":
                portfolio_vector = tf.expand_dims(self.predicted_w,axis=-1)
                network = tf.concat([network,portfolio_vector],axis=-1)
            elif layer["type"] == "MultiHeadAttention":
                out = network
                mha_layer_num, n_heads = layer["mha_layer_num"], layer["heads_num"]
                dropout = layer["dropout_rate"]
                d_k = out.get_shape().as_list()[2]  # get d_model = d_k = window_size
                for j in range(mha_layer_num):
                    ## Cross Asset Attention
                    query = tf.keras.layers.Dense(units=d_k)(out)
                    key = tf.keras.layers.Dense(units=d_k)(out)
                    value = tf.keras.layers.Dense(units=d_k)(out)

                    ## scalar dot product
                    assert d_k % n_heads == 0
                    query = tf.reshape(query,shape=[-1,n_heads,self._rows,d_k // n_heads])
                    key = tf.reshape(key,shape=[-1,n_heads,self._rows,d_k // n_heads])
                    value = tf.reshape(value,shape=[-1,n_heads,self._rows,d_k // n_heads])

                    attention_weights = tf.keras.activations.softmax(tf.matmul(
                                     query,tf.transpose(key,perm=[0,1,3,2])/tf.math.sqrt(tf.constant(d_k,dtype=self.dtype))),axis=-1)
                    q = tf.matmul(attention_weights,value)                      #[None, n_heads, asset_num, window_size]
                    q = tf.transpose(q,perm=[0,2,1,3])                          #[None, asset_num,  n_heads,  window_size]
                    q = tf.reshape(q,shape=[-1,self._rows,d_k])
                    att_out = tf.keras.layers.Dropout(rate=dropout)(
                                                      q,training=self.training)
                    mha_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                                                                  out + att_out) #[None, asset_num, window_size]

                    ## feedforward network
                    ffn_out = point_wise_feed_forward_network(d_model=d_k,dff=60)(mha_out)
                    ffn_out = tf.keras.layers.Dropout(rate=dropout)(ffn_out,training=self.training)
                    out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(mha_out + ffn_out)
                network = out
            elif layer["type"] == "TCN_MHA_OutA":
                out = network
                weights_out = tf.keras.layers.Dense(1)(out)                      #[None, asset_num, 1]
                weights_out = tf.squeeze(weights_out,axis=-1)
                weights_out = tf.keras.layers.BatchNormalization()(weights_out,training=self.training)
                weights_out = tf.keras.activations.tanh(weights_out)
                # weights_out = tf.keras.activations.sigmoid(weights_out)
                # weights_out /= (tf.math.reduce_sum(weights_out,keepdims=True) + 1e-8)
            elif layer["type"] == "TCN_MHA_OutW":
                out = network
                weights_out = tf.keras.layers.Dense(1)(out)                      #[None, asset_num, 1]
                weights_out = tf.squeeze(weights_out,axis=-1)
                # weights_out = tf.keras.layers.BatchNormalization()(weights_out,training=self.training)
                weights_out = tf.keras.activations.softmax(weights_out)
                # weights_out = tf.keras.activations.sigmoid(weights_out)
                # weights_out /= (tf.math.reduce_sum(weights_out,keepdims=True) + 1e-8)
                network = weights_out
            elif layer["type"] == "TCN_MHA_OutQ":
                out = network
                out = tf.keras.layers.Dense(1)(out)                              #[None, asset_num, 1]
                out = tf.squeeze(out,axis=-1)
                out = tf.keras.layers.BatchNormalization()(out, training=self.training)
                out = tf.keras.activations.relu(out)
                out = tf.keras.layers.Dense(1)(out)                              #[None, 1]
                network = out
            elif layer["type"] == "TCN_MHA_Out":
                out = network
                out = tf.keras.layers.Dense(1)(out)                              #[None, asset_num, 1]
                out = tf.squeeze(out,axis=-1)                                    #[None, asset_num]
                out = tf.keras.layers.BatchNormalization()(out, training=self.training)
                out = tf.keras.activations.relu(out)
                network = out
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]

def str2bool(str):
    return str == "True"
