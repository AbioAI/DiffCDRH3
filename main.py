from config import *
from LDM.LatentDiffusion import *
from data.dataloader import *


def load_config_from_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(model, train_loader, val_loader, num_epochs=100):
    model.train()  # 将模型设置为训练模式
    opt, scheduler = model.configure_optimizers()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, cont) in enumerate(train_loader):
            inputs, cont = inputs.to(device), cont.to(device)
            opt.zero_grad()
            loss = model(inputs, cont)
            loss.backward()
            opt.step()

            running_loss += loss.item()

            # 每 100 个批次输出一次损失
            if i % 10 == 0:
                avg_loss = running_loss / len(train_loader)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {avg_loss.item():.4f}")

            # 每个 epoch 输出平均损
            if i % 30 == 0:
                with torch.no_grad():
                    model.eval()
                    val_running_loss = 0.0
                    for i, (inputs, cont) in enumerate(val_loader):
                        # for batch in self.val_loader:
                        inputs, cont = inputs.to(device), cont.to(device)
                        val_loss = model(inputs, cont)
                        val_running_loss += val_loss.item()

                    avg_val_loss = val_running_loss / len(val_loader)
                    print(f"Validation Loss after Epoch [{epoch + 1}]: {avg_val_loss:.4f}")

        print("Training Finished!")


if __name__ == "__main__":
    yaml_file = r""
    config = load_config_from_yaml(yaml_file)
    first_stage, cond_stage, diff_stage = instantiate_from_config(config)

    model = LatentDiffusion(first_stage, cond_stage)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cdrh3_npy_file = f''
    epitope_npy_file = f''
    # 损失函数，优化器
    # opt, scheduler = model.configure_optimizers()
    train_loader, valid_loader = get_dataloaders(cdrh3_npy_file, epitope_npy_file, batch_size=32,
                                                 test_size=0.2, shuffle=True, num_workers=0)

    train(model, train_loader, num_epochs=100)
