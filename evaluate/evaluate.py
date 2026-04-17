
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from utilities.constants import TIMESTAMP

def evaluate_model(model, generator, model_name):
    print(f'\n{"="*50}')
    print(f'  📊 Evaluating: {model_name}')
    print(f'{"="*50}')

    generator.reset()
    y_pred = (model.predict(generator, verbose=0) > 0.5).astype(int).flatten()
    y_true = generator.classes
    names  = list(generator.class_indices.keys())

    acc = accuracy_score(y_true, y_pred)
    print(f'\n  Test Accuracy : {acc*100:.2f}%')
    print(f'\n  Classification Report:')
    print(classification_report(y_true, y_pred, target_names=names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, linewidths=1)
    plt.title(f'Confusion Matrix — {model_name}', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'cm_{model_name.replace(" ","_")}.png', dpi=150)
    plt.show()
    return acc



# ── Model Comparison Bar Chart ──
def plot_model_comparison(cnn_acc, vgg_acc):
    plt.figure(figsize=(7, 5))
    bars = plt.bar(['Custom CNN', 'VGG16 TL'],
               [cnn_acc*100, vgg_acc*100],
               color=['#3498db', '#e67e22'],
               edgecolor='black', width=0.4)
    plt.ylim([0, 115])
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Model Comparison: Custom CNN vs VGG16', fontsize=13, fontweight='bold')
    for bar, acc in zip(bars, [cnn_acc*100, vgg_acc*100]):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.5,
                 f'{acc:.1f}%', ha='center', fontsize=13, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'output/{TIME_STAMP}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    winner = 'VGG16 TL' if vgg_acc > cnn_acc else 'Custom CNN'
    print(f'\n🏆 Best Model: {winner} with {max(cnn_acc, vgg_acc)*100:.2f}% accuracy')
