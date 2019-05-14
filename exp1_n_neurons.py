import track_trial

for seed in range(20):
    for n_neurons in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        track_trial.TrackingTrial().run(
                seed=seed,
                dataset_dir='../davis_io/data',
                n_neurons=n_neurons,
                dt=0.1,
                data_dir='exp1_n_neurons',
                )
