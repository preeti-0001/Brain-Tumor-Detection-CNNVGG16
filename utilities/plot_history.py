from matplotlib import pyplot as plt
from utilities.constants import TIMESTAMP

def plot_history(history, title, color='#3498db'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train', color=color, lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val',   color=color, lw=2, ls='--')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', color='#e74c3c', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val',   color='#e74c3c', lw=2, ls='--')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'output/{TIME_STAMP}/{title.replace(" ","_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
