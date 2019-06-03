import pytry
import os
import random
import nengo
import nengo_extras
import numpy as np
import nengo_dl
import tensorflow as tf
import sys
sys.path.append('.')
import davis_track

class TrackingTrial(pytry.PlotTrial):
    def params(self):
        self.param('number of data sets to use', n_data=-1)
        self.param('data directory', dataset_dir=r'dvs_data')
        self.param('dt', dt=0.01)
        self.param('dt_test', dt_test=0.001)
        self.param('decay time (input synapse)', decay_time=0.01)
        self.param('test set (odd|one)', test_set='one')
        self.param('augment training set with flips', augment=True)
        self.param('output filter', output_filter=0.01)
        self.param('miniback size', minibatch_size=200)
        self.param('learning rate', learning_rate=1e-3)
        self.param('number of epochs', n_epochs=5)
        self.param('saturation', saturation=5)
        self.param('separate positive and negative channels', separate_channels=True)
        
    def evaluate(self, p, plt):
        files = []
        sets = []
        for f in os.listdir(p.dataset_dir):
            if f.endswith('events'):
                files.append(os.path.join(p.dataset_dir, f))

        if p.test_set == 'one':
            test_file = random.sample(files, 1)[0]
            files.remove(test_file)
        
        if p.n_data != -1:
            files = random.sample(files, p.n_data)
            
        inputs = []
        targets = []
        for f in files:
            times, imgs, targs = davis_track.load_data(f, dt=p.dt, decay_time=p.decay_time,
                                               separate_channels=p.separate_channels, 
                                               saturation=p.saturation)
            inputs.append(imgs)
            targets.append(targs[:,:2])
                                
        inputs_all = np.vstack(inputs)
        targets_all = np.vstack(targets)
        
        if p.test_set == 'odd':
            inputs_train = inputs_all[::2]
            inputs_test = inputs_all[1::2]
            targets_train = targets_all[::2]
            targets_test = targets_all[1::2]
        elif p.test_set == 'one':
            times, imgs, targs = davis_track.load_data(test_file, dt=p.dt_test, decay_time=p.decay_time,
                                               separate_channels=p.separate_channels, 
                                               saturation=p.saturation)
            inputs_test = imgs
            targets_test = targs[:, :2]
            inputs_train = inputs_all
            targets_train = targets_all
            
        if p.augment:
            inputs_train, targets_train = davis_track.augment(inputs_train, targets_train,
                                                              separate_channels=p.separate_channels)                
                      
        if p.separate_channels:
            shape = (2, 180, 240)
        else:
            shape = (1, 180, 240)
        
        dimensions = shape[0]*shape[1]*shape[2]
        eval_points_train = inputs_train.reshape(-1, dimensions)
        eval_points_test = inputs_test.reshape(-1, dimensions)

        max_rate = 100
        amp = 1 / max_rate

        model = nengo.Network()
        with model:
            model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear(amplitude=amp)
            model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
            model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            model.config[nengo.Connection].synapse = None

            inp = nengo.Node(
                nengo.processes.PresentInput(inputs_test.reshape(-1, dimensions), p.dt_test),
                size_out=dimensions,
                )

            out = nengo.Node(None, size_in=2)

            conv1 = nengo.Convolution(6, shape, channels_last=False, strides=(2,2))
            layer1 = nengo.Ensemble(conv1.output_shape.size, dimensions=1)
            nengo.Connection(inp, layer1.neurons, transform=conv1)

            conv2 = nengo.Convolution(24, conv1.output_shape, channels_last=False, strides=(2,2))
            layer2 = nengo.Ensemble(conv2.output_shape.size, dimensions=1)
            nengo.Connection(layer1.neurons, layer2.neurons, transform=conv2)

            nengo.Connection(layer2.neurons, out, transform=nengo_dl.dists.Glorot())

            p_out = nengo.Probe(out)

        N = len(inputs_train)
        n_steps = int(np.ceil(N/p.minibatch_size))
        dl_train_data = {inp: np.resize(inputs_train, (p.minibatch_size, n_steps, dimensions)),
                         p_out: np.resize(targets_train, (p.minibatch_size, n_steps, 2))}
        #dl_train_data = {inp: inputs_train.reshape(-1, inp.size_out)[:, None, :],
        #                 p_out: targets_train[:, None, :]}
        N = len(inputs_test)
        n_steps = int(np.ceil(N/p.minibatch_size))
        dl_test_data = {inp: np.resize(inputs_test, (p.minibatch_size, n_steps, dimensions)),
                        p_out: np.resize(targets_test, (p.minibatch_size, n_steps, 2))}
        with nengo_dl.Simulator(model, minibatch_size=p.minibatch_size) as sim:
            #loss_pre = sim.loss(dl_test_data)

            if p.n_epochs > 0:
                sim.train(dl_train_data, tf.train.RMSPropOptimizer(learning_rate=p.learning_rate),
                      n_epochs=p.n_epochs)

            #loss_post = sim.loss(dl_test_data)

            sim.run_steps(n_steps, data=dl_test_data)

        data = sim.data[p_out].reshape(-1,2)[:len(targets_test)]
        rmse_test = np.sqrt(np.mean((targets_test-data)**2, axis=0))            
        if plt:
            plt.plot(data)
            plt.plot(targets_test, ls='--')


        return dict(
            #loss_pre=loss_pre,
            #loss_post=loss_post
            rmse_test = rmse_test
            )
        
        
            
            
        
        
