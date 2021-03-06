import os
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import csv
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


class Writer(object):
    '''
    Writes things to different output files
    '''
    def __init__(self, path_dir):
        i=1
        path = path_dir
        while os.path.exists(path):
            path = path_dir + str(i)
            i+=1
        self.dir = path
        self.dir_check = os.path.join(path, 'checkpoints')
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
        self.write('log', text)


class GanWriter(Writer):
    
    def __init__(self, path_dir):
        super(GanWriter, self).__init__(path_dir)
        self.dir_imgs = os.path.join(self.dir, 'images')
        os.makedirs(self.dir_imgs, exist_ok=True)

    def save_images(self, images, iteration):
        '''
        Saves image of first 25 images
        '''
        save_image(images.view(-1,1,28,28)[:25],
            os.path.join(self.dir_imgs,'{}.png'.format(iteration)),
            nrow=5, normalize=True)
            
    def save_state_dict(self, model, filename):
        torch.save(model.state_dict(), os.path.join(self.dir_check, filename))

    def save_stats(self, gen_loss, dis_loss):
        self.write('stats', '{},{}'.format(gen_loss, dis_loss))

    def make_stats_plot(self):
        losses_dis = []
        losses_gen = []

        with open(os.path.join(self.dir, 'stats.txt')) as f:
            reader = csv.reader(f)
            for row in reader:
                losses_gen.append(float(row[0]))
                losses_dis.append(float(row[1]))
        
        iterations = [i for i in range(len(losses_gen))]

        fig = plt.figure(figsize=(8,5))
        plt.plot(iterations, losses_dis, label='Discriminator')
        plt.plot(iterations, losses_gen, label='Generator')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Generator and Discriminator training loss')
        plt.savefig(os.path.join(self.dir,'losses.png'))

class VaeWriter(Writer):

    def __init__(self, path_dir):
        super(VaeWriter, self).__init__(path_dir)
        self.dir_imgs = os.path.join(self.dir, 'images')
        os.makedirs(self.dir_imgs, exist_ok=True)

    def save_images(self, images, epoch):
        '''
        Saves image of first 25 images
        '''
        save_image(images.view(-1,1,28,28)[:25],
            os.path.join(self.dir_imgs,'{}.png'.format(epoch)),
            nrow=5, normalize=True)
            
    def save_state_dict(self, model, filename):
        torch.save(model.state_dict(), os.path.join(self.dir_check, filename))

    def save_stats(self, train_elbo, val_elbo):
        self.write('stats', '{},{}'.format(train_elbo, val_elbo))

    def save_manifold(self, images, iteration):
        '''
        Saves image of 200 manifolds
        '''
        save_image(images.view(-1,1,28,28)[:400],
            os.path.join(self.dir_imgs,'{}_manifold.png'.format(iteration)),
            nrow=20, normalize=True)

    def save_elbo_plot(self):
        train_curve = []
        val_curve = []

        with open(os.path.join(self.dir, 'stats.txt')) as f:
            reader = csv.reader(f)
            for row in reader:
                train_curve.append(float(row[0]))
                val_curve.append(float(row[1]))

        plt.figure(figsize=(12, 6))
        plt.plot(train_curve, label='train elbo')
        plt.plot(val_curve, label='validation elbo')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('ELBO')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir,'elbos.png'))


class NfWriter(Writer):
    def __init__(self, path_dir):
        super(NfWriter, self).__init__(path_dir)
        self.dir_imgs = os.path.join(self.dir, 'images')
        os.makedirs(self.dir_imgs, exist_ok=True)


    def save_stats(self, train_bpd, val_bpd):
        self.write('stats', '{},{}'.format(train_bpd, val_bpd))

    def save_bpd_plot(self):
        train_curve = []
        val_curve = []

        with open(os.path.join(self.dir, 'stats.txt')) as f:
            reader = csv.reader(f)
            for row in reader:
                train_curve.append(float(row[0]))
                val_curve.append(float(row[1]))

        plt.figure(figsize=(12, 6))
        plt.plot(train_curve, label='train bpd')
        plt.plot(val_curve, label='validation bpd')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Bits per dimension')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir,'bpd.png'))

    def save_images(self, images, epoch):
        '''
        Saves image of first 25 images
        '''
        save_image(images.view(-1,1,28,28)[:25],
            os.path.join(self.dir_imgs,'{}.png'.format(epoch)),
            nrow=5, normalize=True)