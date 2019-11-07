import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import device, grad_clip, print_freq, num_workers
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from optimizer import HNetOptimizer
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = MobileNetV2()
        model = nn.DataParallel(model)

        optimizer = HNetOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_dataset = DeepHNDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers)
    valid_dataset = DeepHNDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        model.zero_grad()
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/learning_rate', optimizer.lr, epoch)
        print('\nCurrent effective learning rate: {}\n'.format(optimizer.lr))

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterion=criterion,
                           logger=logger)

        writer.add_scalar('model/valid_loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, target) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        target = target.float().to(device)  # [N, 8]

        # Forward prop.
        out = model(img)  # [N, 8]
        out = out.squeeze(dim=1)

        # Calculate loss
        loss = criterion(out, target)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            if i % print_freq == 0:
                status = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i,
                                                                       len(train_loader),
                                                                       loss=losses,

                                                                       )
                logger.info(status)

    return losses.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()

    # Batches
    for i, (img, target) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.to(device)
        target = target.float().to(device)

        # Forward prop.
        out = model(img)
        out = out.squeeze(dim=1)

        # Calculate loss
        loss = criterion(out, target)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    status = 'Validation\t Loss {loss.avg:.5f}\n'.format(loss=losses)
    logger.info(status)

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
