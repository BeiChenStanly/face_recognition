import matplotlib.pyplot as plt
import os
from config.settings import RESULTS_DIR, TIMESTAMP

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    if val_accuracies:
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(epochs, learning_rates, 'g-')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True)
    ax3.set_yscale('log')
    
    summary_text = f"""
    Training Summary:
    - Final Training Loss: {train_losses[-1]:.4f}
    - Final Training Accuracy: {train_accuracies[-1]:.2f}%
    """
    
    if val_losses:
        summary_text += f"""
    - Final Validation Loss: {val_losses[-1]:.4f}
    - Final Validation Accuracy: {val_accuracies[-1]:.2f}%
    - Best Validation Loss: {min(val_losses):.4f}
    - Best Validation Accuracy: {max(val_accuracies):.2f}%
        """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    ax4.axis('off')
    
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, f'training_curves_{TIMESTAMP}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path