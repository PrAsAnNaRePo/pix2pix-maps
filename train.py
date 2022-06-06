import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import LoadData
from tqdm import tqdm
import config
from models import Generator, Discriminator

torch.backends.cudnn.benchmark = True


def train_fn(
        disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def test(data, gen):
    pred = gen(data[0][0].reshape(1, 3, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]).to('cuda'))
    plt.imshow(pred.reshape(config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3).cpu().detach().numpy())
    plt.show()


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=config.LR, betas=(0.5, 0.999), )
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config.LR, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = LoadData(img_dir=config.DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    validate_data = LoadData(img_dir=config.VAL_DIR)
    gen.train()
    for epoch in range(1, config.EPOCHS + 1):
        print(f'\n[{epoch}/{config.EPOCHS}]=========>')
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )
        if epoch % config.TEST_FREQUENCY == 0:
            test(validate_data, gen)

    torch.save(
        {
            'gen': gen.state_dict()
        },
        config.MODEL_PATH
    )


if __name__ == "__main__":
    main()
