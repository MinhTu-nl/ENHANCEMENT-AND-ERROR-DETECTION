import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator, PerceptualLoss, initialize_weights
from utils import get_data_loaders, save_image
import os

def train_gan(epochs=100, batch_size=16, lr=0.00005, beta1=0.5, start_epoch=0, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss = PerceptualLoss(device)
    
    initialize_weights(generator)
    initialize_weights(discriminator)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint loaded, starting from scratch or with initialized weights.")

    # Loss functions
    adversarial_loss = nn.BCELoss()
    pixel_loss = nn.L1Loss()

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Data loaders
    low_light_dir = "data/processed/our485/low"
    high_light_dir = "data/processed/our485/high"
    train_loader = get_data_loaders(low_light_dir, high_light_dir, batch_size)

    # Validation loader
    eval_loader = get_data_loaders("data/processed/eval15/low", "data/processed/eval15/high", batch_size)


    for epoch in range(start_epoch, start_epoch + epochs):
        generator.train()
        for i, (low_light, high_light) in enumerate(train_loader):
            low_light, high_light = low_light.to(device), high_light.to(device)

            # Calculate the output size of the Discriminator
            with torch.no_grad():
                sample_output = discriminator(low_light)
                output_size = sample_output.shape[2:]  # e.g., (29, 29)

            # Adjust labels to match the Discriminator's output size
            real_labels = torch.ones(low_light.size(0), 1, *output_size).to(device)
            fake_labels = torch.zeros(low_light.size(0), 1, *output_size).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            
            real_output = discriminator(high_light)
            d_real_loss = adversarial_loss(real_output, real_labels)

            fake_images = generator(low_light)
            fake_output = discriminator(fake_images.detach())
            d_fake_loss = adversarial_loss(fake_output, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            g_pixel_loss = pixel_loss(fake_images, high_light)
            g_perceptual_loss = perceptual_loss(fake_images, high_light)
            g_loss = g_adv_loss + 10 * g_pixel_loss + 5 * g_perceptual_loss
            g_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{start_epoch + epochs - 1}] Batch [{i}/{len(train_loader)}] "
                    f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

        # Evaluate on validation set
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                for low_light, high_light in eval_loader:
                    low_light, high_light = low_light.to(device), high_light.to(device)
                    fake_images = generator(low_light)
                    save_image(fake_images[0], f"CNN+GAN/UPDATE/eval_epoch_{epoch}.png")
                    break
            save_image(fake_images[0], f"CNN+GAN/UPDATE/epoch_{epoch}.png")
            torch.save(generator.state_dict(), f"CNN+GAN/UPDATE/generator_epoch_{epoch}.pth")

if __name__ == "__main__":
    checkpoint_path = "CNN+GAN/UPDATE/generator_epoch_300.pth"
    train_gan(epochs=30, start_epoch=251, checkpoint_path=checkpoint_path, lr=0.000005)