import torch
import os
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from utils.early_stopping import EarlyStopping
from utils.saver import save_loss_plot, save_models  # Import save_loss_plot and save_models
from models.factory import create_gan_model

def initialize_models(config, num_cols, device, logger):
    """Inicializa el Generador y Discriminador usando el factory."""
    logger.info(f"🛠️ Inicializando modelos GAN")
    generator, discriminator = create_gan_model(
        config['MODEL_TYPE'],
        config['LATENT_DIM'],
        num_cols,
        device
    )
    logger.info("✅ Modelos inicializados correctamente.")
    return generator, discriminator

def train(generator, discriminator, dataset, config, device, logger, experiment_tracker, early_stopping):
    """
    Entrena el Generador y Discriminador usando el dataset proporcionado.

    Args:
        generator (nn.Module): Modelo Generador.
        discriminator (nn.Module): Modelo Discriminador.
        dataset (torch.utils.data.Dataset): Dataset de entrenamiento.
        config (dict): Configuración de entrenamiento.
        device (torch.device): Dispositivo de entrenamiento (CPU o GPU).
        logger (logging.Logger): Logger para seguimiento del entrenamiento.
        experiment_tracker (ExperimentTracker): Objeto para el seguimiento de experimentos.
        early_stopping (EarlyStopping): Objeto para aplicar Early Stopping.
    """
    # Inicializar DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Inicializar optimizadores y funciones de pérdida
    optimizer_G = optim.Adam(generator.parameters(), lr=config['LR'], betas=(config['BETA1'], config['BETA2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['LR'], betas=(config['BETA1'], config['BETA2']))
    criterion = nn.BCEWithLogitsLoss()
    scaler_G = GradScaler()
    scaler_D = GradScaler()
    
    real_label = 1.0
    fake_label = 0.0
    G_losses = []
    D_losses = []

    logger.info("🔄 Iniciando bucle de entrenamiento con Early Stopping GAN.")
    
    for epoch in range(config['EPOCHS']):
        epoch_start = time.time()
        total_loss_D = 0.0
        total_loss_G = 0.0

        for data in dataloader:
            real_samples = data[0].to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            with autocast('cuda'):
                labels_real = torch.full((real_samples.size(0),), real_label, dtype=torch.float, device=device)
                output_real = discriminator(real_samples).view(-1)
                loss_D_real = criterion(output_real, labels_real)

                noise = torch.randn(real_samples.size(0), config['LATENT_DIM'], device=device)
                fake_samples = generator(noise)
                labels_fake = torch.full((real_samples.size(0),), fake_label, dtype=torch.float, device=device)
                output_fake = discriminator(fake_samples.detach()).view(-1)
                loss_D_fake = criterion(output_fake, labels_fake)

                loss_D = loss_D_real + loss_D_fake

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            total_loss_D += loss_D.item()

            # Train Generator
            optimizer_G.zero_grad()
            with autocast('cuda'):
                labels_g = torch.full((real_samples.size(0),), real_label, dtype=torch.float, device=device)
                output_g = discriminator(fake_samples).view(-1)
                loss_G = criterion(output_g, labels_g)

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            total_loss_G += loss_G.item()

        # Calculate average losses
        avg_loss_D = total_loss_D / len(dataloader)
        avg_loss_G = total_loss_G / len(dataloader)
        G_losses.append(avg_loss_G)
        D_losses.append(avg_loss_D)

        epoch_time = time.time() - epoch_start
        logger.info(f"📊 Epoch {epoch} | Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f} | Tiempo: {epoch_time:.2f}s")

        # Log metrics in the ExperimentTracker
        experiment_tracker.log_metric("Loss_D", avg_loss_D, step=epoch)
        experiment_tracker.log_metric("Loss_G", avg_loss_G, step=epoch)

        # Check Early Stopping
        if early_stopping(avg_loss_G):
            logger.info(f"🛑 Early Stopping activado en la época {epoch}")
            break

    # Save the loss plot
    save_loss_plot(G_losses, D_losses, config['PLOT_PATH'])

    return G_losses, D_losses