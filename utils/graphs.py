import matplotlib.pyplot as plt

def error_plot(history:dict)->None:

    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='o')

    plt.title('Redução do Erro')
    plt.xlabel('Época')
    plt.ylabel('Erro')
    plt.legend()

    plt.show()