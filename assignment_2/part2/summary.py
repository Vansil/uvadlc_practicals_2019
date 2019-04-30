import os
import torch


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
