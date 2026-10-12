import argparse
import random
from multiprocessing.spawn import freeze_support
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import Classifier


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_filterer(input, output, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    image_size = 64
    num_channels = 3
    batch_size = 128
    num_epochs = 500
    num_gpus = 1
    workers = 2
    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

    dataset = dset.ImageFolder(root=input,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    classifier = Classifier(image_size, num_channels, num_gpus).to(device)
    if (device.type == 'cuda') and (num_gpus > 1):
        classifier = nn.DataParallel(classifier, list(range(num_gpus)))
    classifier.apply(weights_init)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters())

    ckpt_path = Path(output)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_inputs = 0
        for i, data in enumerate(dataloader):
            classifier.zero_grad()
            inputs = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
            out = classifier(inputs).view(-1)
            err = criterion(out, labels)
            running_loss += err * inputs.size(0)
            num_inputs += inputs.size(0)
            err.backward()
            optimizer.step()
        running_loss = running_loss / num_inputs
        print(epoch, running_loss)

        # Save only when improved
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # optional: save these so you can recreate model exactly
                    "image_size": image_size,
                    "num_channels": num_channels,
                },
                ckpt_path,
            )
            print(f"  ✅ saved new best to {ckpt_path} (loss={best_loss:.6f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input/', help='dataroot of input images.')
    parser.add_argument('-o', '--output', type=str, default='train_out.pt', help='path to output directory.')
    parser.add_argument('-s', '--seed', type=int, default=67, help='seed for setting internal state.')
    args = parser.parse_args()

    print(f'Running train.py with arguments:')
    print(f'\tInput = {args.input}')
    print(f'\tOutput = {args.output}')
    print(f'\tSeed = {args.seed}')
    print()

    freeze_support()
    train_filterer(args.input, args.output, args.seed)