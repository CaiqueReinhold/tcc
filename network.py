import gzip
import time
import cPickle

import numpy as np
import theano
import theano.tensor as tt

from netlayers import DenseLayer, ConvPoolLayer


class Network(object):

    def __init__(self, params, input=tt.tensor4('x')):
        self.rng = np.random.RandomState(1234)
        self.input = input

        self.layer0 = ConvPoolLayer(
            input=self.input,
            filter_shape=params['layer0']['filter_shape'],
            image_shape=params['layer0']['image_shape'],
            W_values=self._init_W(params['layer0']['filter_shape']),
            b_values=np.zeros((params['layer0']['filter_shape'][0],),
                                 dtype=theano.config.floatX)
        )

        self.layer1 = DenseLayer(
            input=self.layer0.output.flatten(2),
            n_in=params['layer1']['n_in'],
            n_out=params['layer1']['n_out'],
            W_values=self._init_W((params['layer1']['n_in'],
                                   params['layer1']['n_out'])),
            b_values=np.zeros((params['layer1']['n_out'],),
                              dtype=theano.config.floatX)
        )

        self.out_layer = DenseLayer(
            input=self.layer1.output,
            n_in=params['out']['n_in'],
            n_out=params['out']['n_out'],
            W_values=self._init_W((params['out']['n_in'],
                                   params['out']['n_out'])),
            b_values=np.zeros((params['out']['n_out'],),
                              dtype=theano.config.floatX),
            activation=None
        )

        self.p_y_given_x = tt.nnet.softmax(self.out_layer.output)
        self.y_pred = tt.argmax(self.p_y_given_x, axis=1)
        self.params = (self.layer0.params + self.layer1.params +
                       self.out_layer.params)

    def _init_W(self, shape):
        if len(shape) == 4:
            fan_in = np.prod(shape[1:])
            fan_out = (shape[0] * np.prod(shape[2:]) / 4)
        else:
            fan_in = shape[0]
            fan_out = shape[1]

        W_bound = np.sqrt(6. / (fan_in + fan_out))
        return np.asarray(
            self.rng.uniform(low=-W_bound, high=W_bound, size=shape),
            dtype=theano.config.floatX
        )

    def negative_log_likehood(self, y):
        return -tt.mean(tt.log(self.p_y_given_x)[tt.arange(y.shape[0]), y])

    def errors(self, y):
        return tt.mean(tt.neq(self.y_pred, y))

    def train(self, dataset, learning_rate=0.01, n_epochs=1000):
        print '-##### BUILDING MODEL #####-'
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = dataset[2]

        index = tt.iscalar()
        y = tt.ivector('y')
        cost = self.negative_log_likehood(y)

        test_model = theano.function(
            inputs=[index],
            outputs=self.errors(y),
            givens={
                self.input: test_set_x[index:index + 1],
                y: test_set_y[index:index + 1]
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=self.errors(y),
            givens={
                self.input: valid_set_x[index:index + 1],
                y: valid_set_y[index:index + 1]
            }
        )

        gparams = [tt.grad(cost, param) for param in self.params]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.input: train_set_x[index:index + 1],
                y: train_set_y[index:index + 1]
            }
        )

        print '-##### TRAINING MODEL #####-'

        train_set_size = train_set_x.get_value(borrow=True).shape[0]
        valid_set_size = valid_set_x.get_value(borrow=True).shape[0]
        test_set_size = test_set_x.get_value(borrow=True).shape[0]

        patience = 10000
        patience_increase = 4
        improvement_threshold = 0.995
        validation_frequency = min(train_set_size, patience / 2)

        best_validation_loss = np.inf
        test_score = 0.
        best_test_score = np.inf
        best_params = None
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for index in xrange(train_set_size):
                train_model(index)

            iter = (epoch - 1) * train_set_size + index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i
                                     in xrange(valid_set_size)]
                this_validation_loss = np.mean(validation_losses)
                print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        index + 1,
                        train_set_size,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    test_losses = [test_model(i) for i in xrange(test_set_size)]
                    test_score = np.mean(test_losses)
                    if test_score < best_test_score:
                        best_test_score = test_score
                        best_params = [p.get_value() for p in self.params]

            if patience <= iter:
                done_looping = True
                break

        end_time = time.clock()

        print(
            'Optimization complete. Best validation score of %f %% ' %
            (best_validation_loss * 100.,)
        )
        print 'Test performance of %f %%' % (best_test_score * 100.)
        print 'Time elapsed: %f m' % ((end_time - start_time) / 60.,)
        print '%fs per epoch' % (((end_time - start_time)) / epoch)
        with open('params_cnn.pickle', 'wb') as param_file:
            cPickle.dump(best_params, param_file)


def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(
            np.asarray(data_x,
                          dtype=theano.config.floatX).
                  reshape((len(data_x), 1, 28, 28)),
            name='data_x',
            borrow=borrow
        )
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 name='data_y',
                                 borrow=borrow)
        return (shared_x, tt.cast(shared_y, 'int32'))


def load_dataset(file_name='/home/caique/mnist.pkl.gz'):
    f = gzip.open(file_name, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return [shared_dataset(train_set), shared_dataset(valid_set),
            shared_dataset(test_set)]


def main():
    h, w = 28, 28
    network = Network({
        'layer0': {
            'filter_shape': (10, 1, 5, 5),
            'image_shape': (1, 1, h, w)
        },
        'layer1': {
            'n_in': 12 * 12 * 10,
            'n_out': 400
        },
        'out': {
            'n_in': 400,
            'n_out': 10
        }
    })
    dataset = load_dataset()
    network.train(dataset)


if __name__ == '__main__':
    main()
