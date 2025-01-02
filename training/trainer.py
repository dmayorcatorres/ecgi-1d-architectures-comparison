import torch
import os
import time
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from utils.early_stopping import EarlyStopping
from utils.saver import save_model, save_loss_plot

def initialize_dataloader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

def initialize_optimizers(generator, discriminator, config):
    optimizer_G = optim.Adam(generator.parameters(), lr=config['LEARNING_RATE_GENERATOR'])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['LEARNING_RATE_DISCRIMINATOR'])
    return optimizer_G, optimizer_D

def initialize_scalers():
    return GradScaler(), GradScaler()

def train_discriminator(discriminator, real_samples, fake_samples, criterion, optimizer_D, scaler_D, device):
    batch_size_curr = real_samples.size(0)
    real_label = 0.9
    fake_label = 0.0

    labels_real = torch.full((batch_size_curr,), real_label, dtype=torch.float, device=device)
    labels_fake = torch.full((batch_size_curr,), fake_label, dtype=torch.float, device=device)

    optimizer_D.zero_grad()
    with autocast('cuda'):
        output_real = discriminator(real_samples).view(-1)
        loss_D_real = criterion(output_real, labels_real)

        output_fake = discriminator(fake_samples.detach()).view(-1)
        loss_D_fake = criterion(output_fake, labels_fake)

        loss_D = loss_D_real + loss_D_fake

    scaler_D.scale(loss_D).backward()
    scaler_D.step(optimizer_D)
    scaler_D.update()

    return loss_D.item()

def train_generator(generator, discriminator, fake_samples, criterion, optimizer_G, scaler_G, device):
    batch_size_curr = fake_samples.size(0)
    real_label = 0.9

    labels_g = torch.full((batch_size_curr,), real_label, dtype=torch.float, device=device)

    optimizer_G.zero_grad()
    with autocast('cuda'):
        output_g = discriminator(fake_samples).view(-1)
        loss_G = criterion(output_g, labels_g)

    scaler_G.scale(loss_G).backward()
    scaler_G.step(optimizer_G)
    scaler_G.update()

    return loss_G.item()

def log_epoch_results(epoch, avg_loss_D, avg_loss_G, epoch_time, logger, experiment_tracker):
    logger.info(f"📊 Epoch {epoch} | Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f} | Time: {epoch_time:.2f}s")
    experiment_tracker.log_metric("Loss_D", avg_loss_D, step=epoch)
    experiment_tracker.log_metric("Loss_G", avg_loss_G, step=epoch)

def train(generator, discriminator, dataset, config, device, logger, experiment_tracker, early_stopping):
    """
    Train the Generator and Discriminator using the provided dataset.

    Args:
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        dataset (torch.utils.data.Dataset): Training dataset.
        config (dict): Training configuration.
        device (torch.device): Training device (CPU or GPU).
        logger (logging.Logger): Logger for training tracking.
        experiment_tracker (ExperimentTracker): Object for experiment tracking.
        early_stopping (EarlyStopping): Object to apply Early Stopping.
    """
    dataloader = initialize_dataloader(dataset, config)
    optimizer_G, optimizer_D = initialize_optimizers(generator, discriminator, config)
    criterion = nn.BCEWithLogitsLoss()
    scaler_G, scaler_D = initialize_scalers()

    G_losses, D_losses = [], []

    logger.info("🔄 Starting training loop with Early Stopping for GAN.")

    for epoch in range(1, config['EPOCHS'] + 1):
        epoch_start = time.time()
        total_loss_D = 0.0
        total_loss_G = 0.0

        for data in dataloader:
            real_samples = data[0].to(device)
            batch_size_curr = real_samples.size(0)

            noise = torch.randn(batch_size_curr, config['LATENT_DIM'], device=device)
            fake_samples = generator(noise)

            total_loss_D += train_discriminator(discriminator, real_samples, fake_samples, criterion, optimizer_D, scaler_D, device)
            total_loss_G += train_generator(generator, discriminator, fake_samples, criterion, optimizer_G, scaler_G, device)

        avg_loss_D = total_loss_D / len(dataloader)
        avg_loss_G = total_loss_G / len(dataloader)
        G_losses.append(avg_loss_G)
        D_losses.append(avg_loss_D)

        epoch_time = time.time() - epoch_start
        log_epoch_results(epoch, avg_loss_D, avg_loss_G, epoch_time, logger, experiment_tracker)

        if early_stopping(avg_loss_G):
            logger.info(f"🛑 Early Stopping activated at epoch {epoch}")
            break

    if config['SAVE_FINAL_MODEL_ONLY']:
        save_model(generator, discriminator, config)
    save_loss_plot(D_losses, G_losses, config)

    logger.info("✅ Training completed. Final models and plots saved successfully.")