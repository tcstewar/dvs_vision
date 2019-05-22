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
        self.param('keep invalid frames (ones with no ball)', keep_invalid=False)
        self.param('task (valid|location)', task='location')
        self.param('dt', dt=0.01)
        self.param('dt_test', dt_test=0.001)
        self.param('decay time (input synapse)', decay_time=0.01)
        self.param('number of neurons', n_neurons=100)
        self.param('test set (odd|one)', test_set='one')
        self.param('enhance training set with flips', enhance_training=True)
        self.param('output filter', output_filter=0.01)
        self.param('miniback size', minibatch_size=200)
        self.param('learning rate', learning_rate=1e-3)
        self.param('number of epochs', n_epochs=5)
        
    def evaluate(self, p, plt):
        files = []
        sets = []
        for f in os.listdir(p.dataset_dir):
            if f.endswith('events'):
                files.append(f)
                
        n_data = p.n_data if p.n_data != -1 else len(files)
        for f in random.sample(files, n_data):
            times, targets = davis_track.extract_targets(os.path.join(p.dataset_dir, f),
                                     dt=p.dt,
                                     )
            sets.append([f, times, targets])
            
        if p.keep_invalid:
            for s in sets:
                times, images = davis_track.extract_images(os.path.join(p.dataset_dir, s[0]),
                                     dt=p.dt, decay_time=p.decay_time,
                                     )
                extra_targets = np.tile([[-1, -1, -1, 0]], (len(times)-len(s[1]),1))
                s[2] = np.vstack([s[2], extra_targets])
                s.append(images)
        else:
            for s in sets:
                index = 0
                while s[2][index][3] == 0:
                    index += 1
                    
                times, images = davis_track.extract_images(os.path.join(p.dataset_dir, s[0]),
                                     dt=p.dt, decay_time=p.decay_time,
                                     t_start=s[1][index]-2*p.dt, t_end=s[1][-1]-p.dt,
                                     )
                s[1] = s[1][index:]
                s[2] = s[2][index:]
                if len(images) > len(s[2]):
                    assert len(images) == len(s[2]) + 1
                    images = images[:len(s[2])]
                s.append(images)

                assert len(s[1])==len(s[2])
                assert len(s[2])==len(s[3])
                
                
        
        inputs = []
        targets = []
        if p.task == 'valid':
            for f, times, targ, images in sets:
                inputs.append(images)
                targets.append(targ[:,3:])
        elif p.task == 'location':
            for f, times, targ, images in sets:
                inputs.append(images)
                targets.append(targ[:,:2])
                
                
        inputs_all = np.vstack(inputs)
        targets_all = np.vstack(targets)
        if p.test_set == 'odd':
            inputs_train = inputs_all[::2]
            inputs_test = inputs_all[1::2]
            targets_train = targets_all[::2]
            targets_test = targets_all[1::2]
        elif p.test_set == 'one':
            test_index = random.randint(0, len(inputs)-1)
            if p.dt == p.dt_test:
                inputs_test = inputs[test_index]
                targets_test = targets[test_index]
            else:
                s = sets[test_index]
                times1, targets1 = davis_track.extract_targets(
                        os.path.join(p.dataset_dir, s[0]),
                        dt=p.dt_test)
                index = 0
                while targets1[index][3] == 0:
                    index += 1
                times2, images1 = davis_track.extract_images(
                        os.path.join(p.dataset_dir, s[0]),
                        dt=p.dt_test, decay_time=p.decay_time,
                        t_start=times1[index]-2*p.dt_test,
                        t_end=times1[-1]-p.dt_test,
                        )
                times1 = times1[index:]
                targets1 = targets1[index:]
                if len(images1) > len(targets1):
                    assert len(images1) == len(targets1) + 1
                    images1 = images1[:len(targets1)]

                assert len(times1)==len(targets1)
                assert len(targets1)==len(images1)
                inputs_test = images1
                targets_test = targets1[:,:2]
                
            inputs_train = np.vstack(inputs[:test_index]+inputs[test_index+1:])
            targets_train = np.vstack(targets[:test_index]+targets[test_index+1:])
            
        if p.enhance_training:
            inputs_flip_lr = inputs_train[:,::-1,:]
            targets_flip_lr = np.array(targets_train)
            targets_flip_lr[:,1] = 180 - targets_flip_lr[:,1]
            
            inputs_flip_ud = inputs_train[:,:,::-1]
            targets_flip_ud = np.array(targets_train)
            targets_flip_ud[:,0] = 240 - targets_flip_ud[:,0]
        
            inputs_flip_both = inputs_train[:,::-1,:]
            inputs_flip_both = inputs_flip_both[:,:,::-1]
            targets_flip_both = np.array(targets_train)
            targets_flip_both[:,1] = 180 - targets_flip_both[:,1]
            targets_flip_both[:,0] = 240 - targets_flip_both[:,0]
            inputs_train = np.vstack([inputs_train, inputs_flip_lr, inputs_flip_ud, inputs_flip_both])
            targets_train = np.vstack([targets_train, targets_flip_lr, targets_flip_ud, targets_flip_both])
            
                      
        dimensions = 240*180
        eval_points_train = inputs_train.reshape(-1, dimensions)
        eval_points_test = inputs_test.reshape(-1, dimensions)


        max_rate = 100
        amp = 1 / max_rate
        input_shape = (1, 240, 180)

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

            conv1 = nengo.Convolution(6, input_shape, channels_last=False, strides=(2,2))
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
            loss_pre = sim.loss(dl_test_data)

            if p.n_epochs > 0:
                sim.train(dl_train_data, tf.train.RMSPropOptimizer(learning_rate=p.learning_rate),
                      n_epochs=p.n_epochs)

            loss_post = sim.loss(dl_test_data)

            sim.run_steps(n_steps, data=dl_test_data)

        if plt:
            data = sim.data[p_out].reshape(-1,2)[:len(targets_test)]
            plt.plot(data)
            plt.plot(targets_test, ls='--')
            

        return dict(
            loss_pre=loss_pre,
            loss_post=loss_post
            )
        
        
            
            
        
        
