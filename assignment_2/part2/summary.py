import os
import torch
from matplotlib import pyplot as plt


class Writer(object):
    '''
    Writes things to different output files
    '''
    def __init__(self, path_dir):
        self.dir = path_dir
        self.dir_check = os.path.join(path_dir, 'checkpoints')
        os.makedirs(self.dir_check, exist_ok=True)
        

    def write(self, file_name, text):
        '''
        Write string to line in text file
        '''
        with open(os.path.join(self.dir, file_name+".txt"), 'a') as f:
            f.write(text+"\n")


    def save_model(self, model, iter):
        '''
        Save model to pickle file
        '''
        torch.save(model, os.path.join(self.dir_check, '{:09d}.pt'.format(iter)))
        

    def log(self, text):
        '''
        Print and write to log file
        '''
        print(text)
        self.write('log.txt', text)

class Plotter(object):
    '''
    Makes plot from things written by Writer
    '''

    def plot_metrics(self, metrics_file, output_dir):

        # read
        iters = []
        accuracies = []
        losses = []
        lrs = []

        with open(metrics_file, 'r') as f:
            for line in f:
                it, acc, loss, lr = line.split(',')
                iters.append(int(it))
                accuracies.append(float(acc))
                losses.append(float(loss))
                lrs.append(float(lr))

        # make plots
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(iters, accuracies)
        ax.set_title("Character prediction batch accuracy")
        ax.xaxis.set_label_text("Iteration")
        ax.yaxis.set_label_text("Accuracy")
        fig.savefig(os.path.join(output_dir, 'accuracy.png'))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(iters, losses)
        ax.set_title("Character prediction batch loss")
        ax.xaxis.set_label_text("Iteration")
        ax.yaxis.set_label_text("Loss")
        fig.savefig(os.path.join(output_dir, 'loss.png'))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(iters, lrs)
        ax.set_title("Learning rate")
        ax.xaxis.set_label_text("Iteration")
        ax.yaxis.set_label_text("Learning rate")
        fig.savefig(os.path.join(output_dir, 'learning_rate.png'))
        