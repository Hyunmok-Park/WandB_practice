import numpy as np

def train(model, train_loader, criterion, optimizer, device, config, wandb, epoch):
    model.train()
    wandb.watch(model, criterion, log="all", log_freq=10)

    loss_list = []
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        loss_list += [loss.detach().cpu().numpy().tolist()]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(loss_list)
    wandb.log({"train_loss": avg_loss}, step=epoch)
    print(f"TRAIN: EPOCH {epoch + 1:04d} / {config.epochs:04d} | Epoch LOSS {avg_loss:.4f}")


def vaild(model, vali_loader, criterion, device, wandb, epoch):
    model.eval()
    val_loss = []
    for data, target in vali_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        val_loss += [criterion(output, target).item()]

    val_loss = np.mean(val_loss)
    wandb.log({"valid_loss": val_loss}, step=epoch)
    print(f"VALID: LOSS {val_loss:.4f} | Accuracy {val_loss:.4f} ")
