import matplotlib.pyplot as plt

epoch_list = list(range(10, 201, 10))
loss_list = [0.6889, 0.7199, 0.7393, 0.7099, 0.6823,
             0.7597, 0.6849, 0.7130, 0.6961, 0.7983,
             0.6978, 0.7841, 0.6930, 0.8033, 0.6910,
             0.6768, 0.6883, 0.6848, 0.6963, 0.7162]
plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.grid(True)
plt.ylim(0, 1)
plt.show()