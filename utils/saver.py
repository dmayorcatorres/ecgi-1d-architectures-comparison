import os
import torch
import matplotlib.pyplot as plt

def save_model(generator, discriminator, config):
    """
    Guarda los modelos finales del Generador y Discriminador.

    Args:
        generator (torch.nn.Module): Modelo del Generador.
        discriminator (torch.nn.Module): Modelo del Discriminador.
        config (dict): Configuración actual.
    """
    generator_path = os.path.join(config['MODEL_PATH'], "generator_final.pt")
    discriminator_path = os.path.join(config['MODEL_PATH'], "discriminator_final.pt")
    
    os.makedirs(config['MODEL_PATH'], exist_ok=True)
    
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    
    print(f"✅ Modelo final del Generador guardado en: {generator_path}")
    print(f"✅ Modelo final del Discriminador guardado en: {discriminator_path}")


def save_loss_plot(loss_d, loss_g, config):
    """
    Guarda la gráfica de las pérdidas (Loss_D y Loss_G) a lo largo del entrenamiento.

    Args:
        loss_d (list): Lista de pérdidas del Discriminador.
        loss_g (list): Lista de pérdidas del Generador.
        config (dict): Configuración actual.
    """
    plot_path = os.path.join(config['PLOT_PATH'], "loss_plot.png")
    os.makedirs(config['PLOT_PATH'], exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_d, label='Loss_D')
    plt.plot(loss_g, label='Loss_G')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"✅ Gráfica de pérdidas guardada en: {plot_path}")
