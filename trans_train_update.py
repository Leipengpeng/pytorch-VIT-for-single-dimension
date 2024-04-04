import argparse
from torch import optim, nn
from dataloader import get_dataloader
import torch
from self_supervised.Transformer.trans_update_ver import VIT


def get_hyperparameters():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--root_addr', type=str, default="../../dataset/FS-SEI_4800")
    parser.add_argument('--label_num', type=int, default=30, choices=[10, 30, 50, 90])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--labeled_unlabeled_ratio', type=float, nargs=2, default=[0.8, 0.2])
    parser.add_argument('--rand_seed', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--theta', type=float, default=5)
    return parser.parse_args()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        print("using GPU")
        return torch.device(f'cuda:{i}')
    print("using CPU")
    return torch.device('cpu')


# xavier初始化
def xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def train(train_loader, val_loader, model, criterion, optimizer, device, epochs, theta):
    model.apply(xavier_init_weights)
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader, 1):  # 调整起始值为1
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), theta)  # 梯度裁剪
            optimizer.step()
            running_loss += loss.item()

            acc = calculate_accuracy(outputs, labels)

            if batch_idx % 5 == 0:  # 每训练完5个批次打印一次
                print("训练误差")
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {running_loss}")
                running_loss = 0.0  # 重置 running_loss
                print(f"GPU Memory Allocated: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
                print(f"acc: {acc}")

        running_loss = 0.0
        model.eval()
        for batch_idx, (data, labels) in enumerate(val_loader, 1):  # 调整起始值为1
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            if batch_idx % 5 == 0:
                print("泛化误差")
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {running_loss}")
                running_loss = 0.0  # 重置 running_loss
                print(f"GPU Memory Allocated: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
                print(f"acc: {acc}")


if __name__ == "__main__":
    args = get_hyperparameters()
    train_loader = get_dataloader(root_addr=args.root_addr, label_num=args.label_num,
                                  batch_size=args.batch_size, usage_type='train')
    val_loader = get_dataloader(root_addr=args.root_addr, label_num=args.label_num,
                                batch_size=args.batch_size, usage_type='test')
    theta = args.theta
    model = VIT()
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    device = try_gpu()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train(train_loader, val_loader, model, criterion, optimizer, device, epochs, theta)
