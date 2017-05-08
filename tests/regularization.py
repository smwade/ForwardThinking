from forwardThinking.models import PassForwardThinking, save_model
from forwardThinking.datasets import load_mnist

# Load data
x_train, y_train, x_test, y_test = load_mnist()

def plot_acc_loss(acc, loss, val_acc=None, val_loss=None):
    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.title('Accuracy')
    plt.plot(acc)
    if val_acc != None:
        plt.plot(val_acc)
    plt.subplot(122)
    plt.title('Loss')
    plt.plot(loss)
    if val_loss != None:
        plt.plot(val_loss)
    plt.tight_layout()
    plt.show()

layers = [784, 100, 50, 10]

reg_type = 'l1'
reg = .1
model = PassForwardThinking(layers)
model.fit(x_train, y_train, x_test, y_test, epochs=2, reg_type=reg_type, reg=reg)

#plot_acc_loss(dnn.summary['accuracy'], dnn.summary['loss'],
#             dnn.summary['val_accuracy'], dnn.summary['val_loss'])
print max(model.summary['accuracy'])
