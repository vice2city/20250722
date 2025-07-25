import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# 设置设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载预训练模型并修改输出层
def load_model(num_classes=16):
    # 使用 ResNet50 作为基础模型
    model = models.resnet50(pretrained=True)
    
    # 替换最后的全连接层，使其适应 16 类输出
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# 初始化模型并将其移动到设备上
model = load_model(num_classes=16).to(device)

# 2. 冻结所有参数，只训练最后的全连接层
for param in model.parameters():
    param.requires_grad = False  # 冻结参数

for param in model.fc.parameters():
    param.requires_grad = True  # 只训练最后一层

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类任务的损失函数
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)  # 仅优化最后一层

# 4. 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为 Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 加载训练集和验证集（假设数据按类别放在不同文件夹中）
train_dataset = datasets.ImageFolder('path_to_train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder('path_to_val_data', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # 设置模型为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 移动到设备上

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    print("训练完成！")

# 开始训练
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 6. 保存微调后的模型
torch.save(model.state_dict(), 'finetuned_model.pth')
print("模型已保存为 'finetuned_model.pth'")

# 7. 测试模型
def test_model(model, val_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'验证集准确率: {100 * correct / total:.2f}%')

# 加载微调后的模型并进行测试
model.load_state_dict(torch.load('finetuned_model.pth'))
test_model(model, val_loader)
