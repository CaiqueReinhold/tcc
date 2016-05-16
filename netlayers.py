import theano
import theano.tensor as tt
import theano.tensor.nnet.conv as conv
from theano.tensor.signal.pool import pool_2d


class DenseLayer(object):

    def __init__(self, input, n_in, n_out, W_values, b_values,
                 activation=tt.tanh):
        self.input = input

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        
        lin_output = tt.dot(self.input, self.W) + self.b
        self.output = (lin_output if activation == None
                       else activation(lin_output))
        self.params = [self.W, self.b]


class ConvPoolLayer(object):

    def __init__(self, input, filter_shape, image_shape,
                 W_values, b_values, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = tt.tanh(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        )
        self.params = [self.W, self.b]
