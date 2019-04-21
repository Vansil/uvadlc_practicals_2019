import csv
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

# Plot all accuracies
fname_accuracies = 'output/experiment_acc.txt'
fname_best = 'output/experiment_best.p'

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
acc = (acc - min(acc)) / np.var(acc) * 50 + 10
print(acc,nlayers,lrs)
fig, ax = plt.subplots()
ax.scatter(nlayers,lrs,s=acc,facecolors='none',edgecolors='r')
ax.set(xlabel='Number of layers', ylabel='Learning rate',
    title='Test accuracy')
ax.legend()
ax.grid()
fig.savefig(os.path.join('output', "experiment_acc.png"))