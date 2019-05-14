import pytry
import os
import random
import nengo
import nengo_extras
import numpy as np
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
        self.param('decay time (input synapse)', decay_time=0.01)
        self.param('number of neurons', n_neurons=100)
        self.param('gabor size', gabor_size=11)
        self.param('solver regularization', reg=0.03)
        self.param('test set (odd|one)', test_set='one')
        self.param('enhance training set with flips', enhance_training=True)
        
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
            inputs_test = inputs[test_index]
            targets_test = targets[test_index]
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

        model = nengo.Network()
        with model:
            from nengo_extras.vision import Gabor, Mask
            encoders = Gabor().generate(p.n_neurons, (p.gabor_size, p.gabor_size))
            encoders = Mask((240, 180)).populate(encoders, flatten=True)

            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=dimensions,
                                 encoders=encoders,
                                 neuron_type=nengo.RectifiedLinear(),
                                 intercepts=nengo.dists.CosineSimilarity(p.gabor_size**2+2)
                                 )

            result = nengo.Node(None, size_in=targets_all.shape[1])

            c = nengo.Connection(ens, result, 
                                 eval_points=eval_points_train,
                                 function=targets_train,
                                 solver=nengo.solvers.LstsqL2(reg=p.reg),
                                 )
        sim = nengo.Simulator(model)
        
        error_train = sim.data[c].solver_info['rmses']

        _, a_train = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=eval_points_train)    
        outputs_train = np.dot(a_train, sim.data[c].weights.T)       
        rmse_train = np.sqrt(np.mean((targets_train-outputs_train)**2))
        _, a_test = nengo.utils.ensemble.tuning_curves(ens, sim, inputs=eval_points_test)    
        outputs_test = np.dot(a_test, sim.data[c].weights.T)       
        rmse_test = np.sqrt(np.mean((targets_test-outputs_test)**2))
        
        
        if plt:
            plt.subplot(2, 1, 1)
            plt.plot(targets_train, ls='--')
            plt.plot(outputs_train)
            plt.title('train\nrmse=%1.4f' % rmse_train)
            
            plt.subplot(2, 1, 2)
            plt.plot(targets_test, ls='--')
            plt.plot(outputs_test)
            plt.title('test\nrmse=%1.4f' % rmse_test)
            
        
        
        return dict(
            rmse_train=rmse_train,
            rmse_test=rmse_test,
        )
        
        
        
            
            
        
        
