import gc
import cPickle
import gzip
from datetime import datetime

import numpy as np
import theano
import theano.tensor as tt

from netlayers import ConvPoolLayer, LSTMLayer, CTCLayer, DenseLayer
from prepare_data import get_data, CLASSES, stringify


class Network(object):

    def __init__(self, input=tt.matrix('x'), y=tt.ivector('y'), params=None):
        self.input = input
        self.y = y

        h, w = 100, 20
        fn, fh, fw = 5, 9, 3

        self.layer0 = ConvPoolLayer(
            filter_shape=(fn, 1, fh, fw),
            image_shape=(1, 1, h, w),
            params=params[0] if params else None
        )

        def conv(index, X):
            slice_x = X[:, index * w:(index + 1) * w]
            # return self.layer0.output(
            #     slice_x.dimshuffle('x', 'x', 0, 1)
            # ).flatten(1)
            return slice_x.flatten(1)

        conv_out, _ = theano.scan(
            fn=conv, sequences=[tt.arange(self.input.shape[1] / w)],
            non_sequences=self.input, outputs_info=None
        )

        self.layer1 = LSTMLayer(
            input=conv_out,
            in_size=((h - fh + 1) / 2) * ((w - fw + 1) / 2) * fn,
            out_size=512,
            params=params[1] if params else None
        )

        self.layer2 = DenseLayer(
            input=self.layer1.output,
            in_size=512,
            out_size=256,
            params=params[2] if params else None
        )

        self.layer3 = LSTMLayer(
            input=self.layer2.output,
            in_size=256,
            out_size=256,
            params=params[3] if params else None
        )

        self.layer4 = DenseLayer(
            input=self.layer3.output,
            in_size=256,
            out_size=len(CLASSES) + 1,
            params=params[4] if params else None
        )

        self.ctc = CTCLayer(
            input=self.layer4.output,
            y=self.y
        )

        self.params = (
            self.layer1.params +
            self.layer2.params +
            self.layer3.params +
            self.layer4.params
        )

    def save_params(self, filename='net1.pkl.gzip'):
        def unwrap(shared):
            return [p.get_value() for p in shared]

        params = [
            unwrap(self.layer0.params),
            unwrap(self.layer1.params),
            unwrap(self.layer2.params),
            unwrap(self.layer3.params),
            unwrap(self.layer4.params)
        ]

        f = gzip.open(filename, 'wb')
        cPickle.dump(params, f)
        f.close()

    def train(self, learning_rate=0.001, n_epochs=50):
        print '----BUILDING MODEL----'

        cost = self.ctc.log_ctc()
        grads = tt.grad(cost, self.params)
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]

        train_model = theano.function(
            inputs=[self.input, self.y],
            outputs=cost,
            updates=updates
        )

        test = theano.function(
            inputs=[self.input],
            outputs=self.ctc.output,
        )

        print '----TRAINING MODEL----'

        best_cost = np.inf
        start_time = datetime.now()

        for epoch in range(n_epochs):
            for i in range(3):
                x, y = get_data(i)
                costs = [train_model(x[j], y[j]) for j in range(len(x))]
                mean_cost = np.mean(costs)

                if mean_cost < best_cost:
                    best_cost = mean_cost
                    self.save_params()

                print 'epoch %d/%d mean cost %f' % (epoch, i, mean_cost)
                print 'elapsed time: %d mins' % ((datetime.now() - start_time).seconds / 60)
                print 'target: ', stringify(y[1])
                print 'pred: ', stringify(test(x[1]))

                del x, y
                gc.collect()


def main():
    nnet = Network()
    nnet.train()

if __name__ == '__main__':
    main()
