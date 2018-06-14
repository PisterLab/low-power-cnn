"""
Adapted from Frederik Kratzert's AlexNet code.
Modified by Lydia Lee 2018-06-13
"""

class NN_2D(object):
    def __init__(self, inputs, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        """
        Author: Frederik Kratzert
        Args:
            inputs: tf.placeholder, for the input images
            keep_prob: tf.placeholder, the dropout rate
            num_classes: int, number of classes of the new dataset
            skip_layer: list of strings, names of the layers to reinitialize
            weights_path: path string, path to the pretrained weights
        """
        self.INPUTS = inputs
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.IS_TRAINING = is_training
        
        if weights_path=='DEFAULT':
            self.WEIGHTS_PATH = '.'
        else:
            self.WEIGHTS_PATH = weights_path
        self.create()
        
    def create(self):
        pass
    
    def load_initial_weights(self):
        """
        Used to assign the pretrained weights to the created variables.
        """
        pass
    
    def _conv_layer(self, inputs, filter_shape, num_filters, stride_shape, name, padding='SAME', groups=1):
        """
        Adapted from: https://github.com/ethereon/caffe-tensorflow
        Args:
            inputs: tf.Tensor, input tensor
            filter_shape: [int height, int width], filter dimensions
            num_filters: int, number of filters for the layer
            stride_shape: [int height, int width], dimensions of stride
            name: string, unique name of the layer
            padding: string 'SAME' or 'VALID', see TF documentation for more detail
            groups: int, number of groups to split input and weights into
        Returns:
            Convolutional layer operation
        """
        # Get number of input channels
        input_channels = int(inputs.get_shape()[-1])
  
        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                             strides = [1, stride_y, stride_x, 1],
                                             padding = padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape = [num_filters])  

            if groups == 1:
                conv = convolve(inputs, weights)
            # In the cases of multiple groups, split inputs & weights and
            else:
                # Split input and weights and convolve them separately
                input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=inputs)
                weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
                # Concat the convolved output together again
                conv = tf.concat(axis = 3, values = output_groups)

            # Add biases 
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
            # Apply relu function
            relu = tf.nn.relu(bias, name = scope.name)
            return relu
        
    def _fc_layer(self, inputs, num_in, num_out, name, relu = True):
        """
        Author: Frederik Kratzert
        Args:
            inputs: tf.Tensor, input tensor
            num_in: int, number of columns of weights tensor
            num_out: int, number of rows of weights tensor
            name: string, unique name of the layer
            relu: boolean, True if returning a ReLU
        Returns:
            Dense layer operation
        """
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

            if relu == True:
            # Apply ReLu nonlinearity
                relu = tf.nn.relu(act)      
                return relu
            else:
                return act

    def _maxpool_layer(self, inputs, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        """
        Author: Frederik Kratzert
        Args:
            inputs: tf.Tensor, input tensor
            filter_height: int, pool height
            filter_width: int, pool width
            stride_y: int, stride step in y-direction
            stride_x: int, stride step in x-direction
            name: string, unique name of the layer
            padding:
        Returns:
            Max pool layer operation
        """
        return tf.nn.max_pool(inputs, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)
    
    def _avgpool_layer(self, inputs, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        """
        Args:
            inputs: tf.Tensor, input tensor
            filter_height: int, pool height
            filter_width: int, pool width
            stride_y: int, stride step in y-direction
            stride_x: int, stride step in x-direction
            name: string, unique name of the layer
            padding:
        Returns:
            Average pool layer operation
        """
        return tf.nn.avg_pool(inputs, ksize=[1, filter_height, filter_width, 1],
            strides=[1, stride_y, stride_x, 1],
            padding=padding, name=name)
    
    def _lrn_layer(self, inputs, radius, alpha, beta, name, bias=1.0):
        """
        Author: Frederik Kratzert
        Args:
            inputs: tf.Tensor, input tensor
            radius:
            alpha:
            beta:
            name: string, unique name of the layer
            bias:
        Returns:
            Local response normalization layer operation
        """
        return tf.nn.local_response_normalization(inputs, depth_radius = radius, alpha = alpha,
                                                  beta = beta, bias = bias, name = name)

    def _dropout_layer(self, inputs, keep_prob, name):
        """
        Author: Frederik Kratzert
        Args:
            inputs: tf.Tensor, input tensor
            keep_prob: scalar tensor of the same type as inputs, the probability
                that each element is kept
            name: string, unique name of the layer
        Returns:
            Dropout layer operation
        """
        return tf.nn.dropout(inputs, keep_prob, name=name)


class SqueezeNet(NN_2D):
    def __init__(self, inputs, keep_prob, num_classes, skip_layer, weights_path='DEFAULT'):
        """
        Args:
            inputs: tf.placeholder, for the input images
            keep_prob: tf.placeholder, the dropout rate
            num_classes: int, number of classes of the new dataset
            skip_layer: list of strings, names of the layers to reinitialize
            weights_path: path string, path to the pretrained weights
        """
        NN_2D.__init__(self, inputs, keep_prob, num_classes, skip_layer, weights_path)
    
    def create(self):
        """
        NN implementation.
        """
        conv1 = self._conv_layer(inputs=inputs, filter_shape=[7, 7],
                                 num_filters=96, stride_shape=[2, 2],
                                 name='conv1', padding='SAME', groups=1)
        maxpool1 = self._maxpool_layer(inputs=conv1, filter_height=3, filter_width=3,
                                       stride_y=2, stride_x=2, name='maxpool1',
                                       padding='SAME')
        
        fire2 = self._fire_layer(inputs=maxpool1,
                                 s1x1=16, e1x1=64, e3x3=64,
                                 name='fire2')
        fire3 = self._fire_layer(inputs=fire2,
                                 s1x1=16, e1x1=64, e3x3=64,
                                 name='fire3')
        fire4 = self._fire_layer(inputs=fire3,
                                 s1x1=32, e1x1=128, e3x3=128,
                                 name='fire4')
        maxpool4 = self._maxpool_layer(inputs=fire4, filter_height=3, filter_width=3,
                                       stride_y=2, stride_x=2, name='maxpool4',
                                       padding='SAME')
        
        fire5 = self._fire_layer(inputs=maxpool4,
                                 s1x1=32, e1x1=128, e3x3=128,
                                 name='fire5')
        fire6 = self._fire_layer(inputs=fire5,
                                 s1x1=48, e1x1=192, e3x3=192,
                                 name='fire6')
        fire7 = self._fire_layer(inputs=fire6,
                                 s1x1=48, e1x1=192, e3x3=192,
                                 name='fire7')
        fire8 = self._fire_layer(inputs=fire7,
                                 s1x1=64, e1x1=256, e3x3=256,
                                 name='fire8')
        maxpool8 = self._maxpool_layer(inputs=fire8, filter_height=3, filter_width=3,
                                       stride_y=2, stride_x=2, name='maxpool8',
                                       padding='SAME')
        
        fire9 = self._fire_layer(inputs=maxpool8,
                                 s1x1=64, e1x1=256, e3x3=256,
                                 name='fire9')
        dropout9 = self._dropout_layer(inputs=fire9, keep_prob=self.KEEP_PROB, name='dropout9')
        conv10 = self._conv_layer(inputs=dropout9, filter_shape=[1, 1],
                                  num_filters=self.NUM_CLASSES, stride_shape=[1, 1],
                                  name='conv10', padding='SAME', groups=1)
        avgpool10 = self._avgpool_layer(inputs=conv10, filter_height=13, filter_width=13,
                                        stride_y=1, stride_x=1, name='avgpool10',
                                        padding='SAME')
        return avgpool10
        
    def load_initial_weights(self, session):
        """
        Author: Alex Kratzert
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse = True):
                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable = False)
                            session.run(var.assign(data))
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable = False)
                            session.run(var.assign(data))
    
    def _fire_layer(self, inputs, s1x1, e1x1, e3x3, name):
        """
        Args:
            inputs: tf.Tensor, input tensor
            s1x1: int, number of 1x1 filters in squeeze layer
            e1x1: int, number of 1x1 filters in expand layer
            e3x3: int, number of 3x3 filters in expand layer
            name: string, unique name of the layer
        Returns:
            Fire layer operation
        Raises:
            Assertion error if the number of squeeze layer filters 
            exceeds the sum of the number of expand layer filters
            (see paper on SqueezeNet for explanation)
        """
        assert s1x1 <= e1x1+e3x3, "Too many squeeze layer filters \
                                    in {}".format(name)
        
        squeeze1x1 = self._conv_layer(inputs=inputs, filter_shape=[1, 1],
                                      num_filters=s1x1, stride_shape=[1,1],
                                      name=name+'/s1x1')
        expand1x1 = self._conv_layer(inputs=squeeze1x1, filter_shape=[1, 1],
                                    num_filters=e1x1, stride_shape=[1,1],
                                    name=name+'/e1x1')
        expand3x3 = self._conv_layer(inputs=squeeze1x1, filter_shape=[3, 3],
                                    num_filters=e3x3, stride_shape=[1,1],
                                    name=name+'/e3x3')
        return tf.concat([expand1x1, expand3x3], 3, name=name+'/concat')