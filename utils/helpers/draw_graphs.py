import matplotlib.pyplot as plt
import pickle

fname = 'checkpoints/new_epoch_20.pkl'

with open(fname, 'rb') as f:
    data = pickle.load(f)

plt.figure(1)
plt.title('Loss History')
plt.ylabel('loss')
plt.xticks([])
plt.plot(data['loss_history'])
plt.savefig('LossHistory.png')

plt.figure(2)
plt.title('Accuracy History')
plt.ylabel('accuracy')
plt.xticks([])
plt.plot(data['train_acc_history'])
plt.plot(data['val_acc_history'])
plt.legend(['train', 'validation'],loc='lower right')
plt.savefig('AccuracyHistory.png')
