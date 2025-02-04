from .trainGAN import trainGAN
from .trainCGAN import trainCGAN

from .trainGAN import trainGAN
from .trainCGAN import trainCGAN

from src.models.simple_cgan import Generator as CGANGenerator, Discriminator as CGANDiscriminator
from src.models.simple_gan import Generator as GANGenerator, Discriminator as GANDiscriminator


class TrainerManager:
    def __init__(self, config, exp_dir, device, logger):
        self.config = config
        self.exp_dir = exp_dir
        self.device = device
        self.logger = logger

    def train_model(self, trial, model_type, dataloader, input_dim):
        """
        Selecciona y entrena el modelo basado en `model_type`.
        """

        latent_dim = trial.suggest_categorical('LATENT_DIM', self.config['ARCHITECTURE']['LATENT_DIM'])
        dropout = trial.suggest_categorical('DROPOUT', self.config['ARCHITECTURE']['DROPOUT'])

        if model_type == "GAN":
            generator = GANGenerator(latent_dim, input_dim).to(self.device)
            discriminator = GANDiscriminator(input_dim).to(self.device)
            return generator, discriminator, trainGAN(generator, discriminator, dataloader, self.device, self.config, trial)

        elif model_type == "CGAN":
            generator = CGANGenerator(latent_dim, 8, 1, input_dim, dropout).to(self.device)
            discriminator = CGANDiscriminator(input_dim, 8, 1, dropout).to(self.device)
            return generator, discriminator, trainCGAN(generator, discriminator, dataloader, self.device, self.config, trial)

        else:
            raise ValueError(f"Modelo no reconocido: {model_type}")