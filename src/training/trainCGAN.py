import torch
import time
from torch.cuda.amp import autocast, GradScaler
import optuna

def trainCGAN(generator, discriminator, dataloader, device, config, trial):
    """
    :param generator: ConditionalGenerator
    :param discriminator: ConditionalDiscriminator
    :param dataloader: que devuelve (real_samples, real_labels)
    :param device: 'cuda' o 'cpu'
    :param config: diccionario con config['TRAINING']['EPOCHS']
    :param trial: objeto Optuna para sugerir hiperparámetros
    """
    # Hiperparámetros que optimizamos con Optuna
    learning_rate = trial.suggest_float(
        'LEARNING_RATE',
        config['TRAINING']['LEARNING_RATE_RANGE'][0],
        config['TRAINING']['LEARNING_RATE_RANGE'][1],
        log=True
    )
    beta1 = trial.suggest_float('BETA1',
                                config['TRAINING']['BETA1_RANGE'][0],
                                config['TRAINING']['BETA1_RANGE'][1])
    beta2 = trial.suggest_float('BETA2',
                                config['TRAINING']['BETA2_RANGE'][0],
                                config['TRAINING']['BETA2_RANGE'][1])
    latent_dim = trial.suggest_categorical('LATENT_DIM', config['ARCHITECTURE']['LATENT_DIM'])
    num_epochs = config['TRAINING']['EPOCHS']

    # Definir optimizadores
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # Amp grad scaler
    scaler_G = torch.amp.GradScaler(device='cuda')
    scaler_D = torch.amp.GradScaler(device='cuda')

    G_losses, D_losses = [], []
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(num_epochs):
        total_loss_G = 0.0
        total_loss_D = 0.0
        epoch_start = time.time()

        for real_samples, real_labels in dataloader:
            batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)
            real_labels = real_labels.long().to(device)  # asegurar int64 p/ embedding

            # -------------------------------------------
            # 1) Entrenar Discriminador
            # -------------------------------------------
            optimizer_D.zero_grad()
            with autocast():
                # Muestras reales
                pred_real = discriminator(real_samples, real_labels)  # shape (batch_size,1)
                targets_real = torch.full((batch_size,1), real_label, device=device)
                loss_real = criterion(pred_real, targets_real)

                # Muestras falsas
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_samples = generator(z, real_labels)
                pred_fake = discriminator(fake_samples.detach(), real_labels)
                targets_fake = torch.full((batch_size,1), fake_label, device=device)
                loss_fake = criterion(pred_fake, targets_fake)

                loss_D = loss_real + loss_fake

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            total_loss_D += loss_D.item()

            # -------------------------------------------
            # 2) Entrenar Generador
            # -------------------------------------------
            optimizer_G.zero_grad()
            with autocast():
                # Queremos que el D considere las muestras falsas como reales
                pred_fake_for_G = discriminator(fake_samples, real_labels)
                targets_gen = torch.full((batch_size,1), real_label, device=device)
                loss_G = criterion(pred_fake_for_G, targets_gen)

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            total_loss_G += loss_G.item()

        # Fin de la época
        avg_loss_D = total_loss_D / len(dataloader)
        avg_loss_G = total_loss_G / len(dataloader)
        G_losses.append(avg_loss_G)
        D_losses.append(avg_loss_D)

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{num_epochs}] LossD: {avg_loss_D:.4f} LossG: {avg_loss_G:.4f} Time: {elapsed:.2f}s")

    return G_losses, D_losses