import train_mlp_pytorch as train
import os
import numpy as np
import pickle

fname_accuracies = 'output/experiment_acc.txt'
with open(fname_accuracies,'w') as f:
    f.write("nlayers,lr,acc\n")
fname_best = 'output/experiment_best.p'


best_acc = 0
best_data = None
i=0
while True:
    i+=1
    # Select params
    n_layers = np.random.randint(1,7)
    layers = ("100,"*n_layers)[:-1]
    lr = np.exp(-np.random.rand()*10)
    
    # Run experiment
    data_raw = train.experiment(layers, lr)
    test_acc = data_raw['metrics']['test_acc'][-1][1]

    # Update best
    if test_acc > best_acc:
        best_acc = test_acc
        best_data = data_raw
        # Write raw data
        pickle.dump(data_raw, open(fname_best, "wb"))
        print("HIGHSCORE!")

    # Write to file
    with open(fname_accuracies,'a') as f:
        f.write("{},{},{}\n".format(n_layers,lr,test_acc))
