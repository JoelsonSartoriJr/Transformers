import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def error_plot(history:dict)->None:

    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='o')

    plt.title('Redução do Erro')
    plt.xlabel('Época')
    plt.ylabel('Erro')
    plt.legend()

    plt.show()
    
def display_attention(sentence:list, translation:list, attention:list, n_heads:int = 8, n_rows:int = 4, n_cols:int = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+[t.lower() for t in sentence], rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()