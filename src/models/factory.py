from models.simple_gan.generator import SimpleGenerator
from models.simple_gan.discriminator import SimpleDiscriminator

def create_gan_model(model_type, latent_dim, input_dim, device):
    """
    Crea el Generador y Discriminador basado en el tipo de modelo especificado.

    Args:
        model_type (str): Tipo de GAN (simple, dcgan, wgan, etc.).
        latent_dim (int): Dimensión del espacio latente.
        input_dim (int): Dimensión de entrada para el discriminador.
        device (torch.device): Dispositivo (CPU o CUDA).
    
    Returns:
        Tuple[nn.Module, nn.Module]: Instancias del Generador y Discriminador.
    """
    if model_type == 'simple':
        generator = SimpleGenerator(latent_dim, input_dim).to(device)
        discriminator = SimpleDiscriminator(input_dim, latent_dim).to(device)
    else:
        raise ValueError(f"❌ Tipo de GAN no soportado: {model_type}")
    
    return generator, discriminator
