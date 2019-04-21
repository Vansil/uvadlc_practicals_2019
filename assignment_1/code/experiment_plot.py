import csv
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

# Plot all accuracies
fname_accuracies = 'output/experiment_acc.txt'

nlayers = []
lrs = []
accuracies = []

line_count = 0
with open(fname_accuracies) as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        line_count += 1
        if line_count != 1:
            nlayers.append(float(row[0]))
            lrs.append(float(row[1]))
            accuracies.append(float(row[2]))
# make figure
acc = np.array(accuracies)
maxind = np.argmax(acc)
print("Best accuracy: {}".format(max(acc)))
acc = (acc - min(acc)) / np.var(acc) * 50 + 10
fig, ax = plt.subplots()
ax.scatter(nlayers[maxind],lrs[maxind],s=acc[maxind],facecolors='b',edgecolors='b',label='Best result')
ax.scatter(nlayers,lrs,s=acc,facecolors='none',edgecolors='r',label='Other result')
ax.set(xlabel='Number of layers', ylabel='Learning rate',
    title='Test accuracy', yscale='log')
plt.ylim(min(lrs),max(lrs))
plt.legend()
ax.grid()
fig.savefig(os.path.join('output', "experiment_acc.png"))

# Plots of best example
fname_best = 'output/experiment_best.p'
data_raw = pickle.load(open(fname_best, "rb"))
net = data_raw['net']
metrics = data_raw['metrics']
train_loss = metrics['train_loss']
gradient_norms = metrics['gradient_norms']
train_acc = metrics['train_acc']
test_acc = metrics['test_acc']

# Save plots
out_dir = 'output'
# Loss
fig, ax = plt.subplots()
iter = [i for (i,q) in train_loss]
loss = [q for (i,q) in train_loss]
ax.plot(iter, loss)
ax.set(xlabel='Iteration', ylabel='Loss (log)',
    title='Batch training loss')
ax.set_yscale('log')
ax.grid()
fig.savefig(os.path.join(out_dir, "exp_best_loss.png"))
# gradient norm
fig, ax = plt.subplots()
iter = [i for (i,q) in gradient_norms]
norm = [q for (i,q) in gradient_norms]
ax.plot(iter, norm)
ax.set(xlabel='Iteration', ylabel='Norm',
    title='Gradient norm')
ax.grid()
fig.savefig(os.path.join(out_dir, "exp_best_gradient_norm.png"))
# accuracies
fig, ax = plt.subplots()
iter = [i for (i,q) in train_acc]
accu = [q for (i,q) in train_acc]
ax.plot(iter, accu, label='Train')
iter = [i for (i,q) in test_acc]
accu = [q for (i,q) in test_acc]
ax.plot(iter, accu, label='Test')
ax.set(xlabel='Iteration', ylabel='Accuracy',
    title='Train and test accuracy')
ax.legend()
ax.grid()
fig.savefig(os.path.join(out_dir, "exp_best_accuracy.png"))

print("hack successful")
