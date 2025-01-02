import os
import torch
import matplotlib.pyplot as plt

def save_loss_plot(G_losses, D_losses, plot_path):
    """Save the loss plot to the specified path."""
    plt.figure()
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss Over Epochs')
    plt.savefig(os.path.join(plot_path, 'loss_plot.png'))
    plt.close()

def save_models(generator, discriminator, model_path):
    """Save the generator and discriminator models to the specified path."""
    torch.save(generator.state_dict(), os.path.join(model_path, 'generator_final.pt'))
    torch.save(discriminator.state_dict(), os.path.join(model_path, 'discriminator_final.pt'))