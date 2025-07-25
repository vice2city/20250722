import torch
from model import OrientedRCNN

def load_model(weights_path=None):
    model = OrientedRCNN()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

def inference(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output

if __name__ == "__main__":
    # 加载模型和权重
    weights_path = "path/to/weights.pth"  # 如果有预训练权重文件
    model = load_model(weights_path)

    # 创建一个输入张量 [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 3, 224, 224)

    # 执行推理
    output = inference(model, input_tensor)
    print("Inference Output Shape:", output.shape)
