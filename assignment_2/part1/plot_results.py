import matplotlib.pyplot as plt
import numpy as np


for model in ['RNN', 'LSTM']:
    print("Model: {}".format(model))
    # init figures
    fig_acc = plt.figure(figsize=(6,7))
    fig_loss = plt.figure(figsize=(6,7))

    fig_acc.subplotpars.update(bottom=.2, top=.95)
    fig_loss.subplotpars.update(bottom=.2, top=.95)

    ax_acc = fig_acc.add_subplot(111)
    ax_loss = fig_loss.add_subplot(111)
    legend = []

    # make figures
    with open('output/experiment_results_{}.txt'.format(model)) as f: 
       count = 0
       for line in f: 
              count += 1
              if (model == 'RNN' and count > 12) or (model =='LSTM' and count > 8):
                  break
              data = line.split(';')
              seq_len = int(data[0])
              losses = [float(x) for x in data[1].split(',')]
              accuracies = [float(x) for x in data[2].split(',')]
              print(seq_len, losses[-1], accuracies[-1])
              # Plot
              x = np.arange(len(losses))*10+10
              ax_loss.plot(x, losses)
              ax_acc.plot(x, accuracies)
              legend.append("length {}".format(seq_len))

    # Save figures
    fig_acc.legend(legend, ncol=4, loc='lower center', 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    fig_loss.legend(legend, ncol=4, loc='lower center', 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)
    
    ax_acc.set_title("{} accuracy".format(model))
    ax_acc.xaxis.set_label_text("Iteration")
    ax_acc.yaxis.set_label_text("Accuracy")

    ax_loss.set_title("{} loss".format(model))
    ax_loss.xaxis.set_label_text("Iteration")
    ax_loss.yaxis.set_label_text("Loss")

    fig_acc.savefig('output/experiment_results_{}_accuracy_first.png'.format(model))
    fig_loss.savefig('output/experiment_results_{}_loss_first.png'.format(model))
    