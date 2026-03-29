import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import numpy as np
import math

# 检查是否有 GPU，M1/M2 Mac 可以使用 "mps"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 调整后的超参数
num_timesteps = 1000
batch_size = 128
num_steps = 5000        # 训练迭代次数 (原变量名为 num_epochs，但逻辑上是 steps)
base_lr = 1e-3          # 初始学习率

def get_swiss_roll_data(n_samples=5000):
    # 使用 sklearn 生成瑞士卷数据，只取前两个维度用于 2D 演示
    data, _ = make_swiss_roll(n_samples, noise=0.2)
    data = data[:, [0, 2]] / 10.0 # 缩放到 [-1, 1] 附近
    return torch.from_numpy(data).float().to(device)

# 加载全部数据
dataset = get_swiss_roll_data()
print(f"Dataset shape: {dataset.shape}")

# ============================
# 2. 定义扩散过程 (Forward Process)
# ============================
# 我们需要定义噪声强度调度表 (Noise Schedule)
# beta: 每个时间步增加的噪声方差
betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
# alphas: 1 - beta，表示保留原始信息的比例
alphas = 1. - betas
# alphas_cumprod: alpha 的累乘，用于直接计算任意 t 时刻的噪声数据
alphas_cumprod = torch.cumprod(alphas, dim=0)
# 预计算平方根，方便后续使用
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def q_sample(x_0, t, noise=None):
    """
    前向加噪过程: 给定真实数据 x_0 和时间步 t，生成带有噪声的数据 x_t
    公式: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x_0) # 生成标准高斯噪声
    
    # 提取当前时间步 t 对应的系数，并调整形状以便广播计算
    sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    
    # 返回加噪后的数据 x_t
    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

# ============================
# 3. 定义神经网络模型 (Reverse Process)
# ============================

# 3.1 正弦位置编码层
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# 3.2 残差块 (Residual Block)
class Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU() # 使用 GELU 替代 ReLU

    def forward(self, x):
        return x + self.act(self.ff(x)) # 残差连接：输入直接加到输出

# 3.3 主网络
class ImprovedDiffusionNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 时间嵌入层
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # 输入投影
        self.input_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU()
        )

        # 中间残差层 (加深网络)
        self.hidden_layers = nn.Sequential(
            Block(hidden_dim),
            Block(hidden_dim),
            Block(hidden_dim)
        )

        # 输出层
        self.output_mlp = nn.Linear(hidden_dim, 2)

    def forward(self, x, t):
        # 1. 处理时间步 t -> 向量
        time_emb = self.time_mlp(t)
        
        # 2. 处理输入 x -> 向量
        x_emb = self.input_mlp(x)
        
        # 3. 融合特征 (简单的相加即可，因为维度相同)
        h = x_emb + time_emb
        
        # 4. 通过残差网络
        h = self.hidden_layers(h)
        
        return self.output_mlp(h)

model = ImprovedDiffusionNet(hidden_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=base_lr)

# 添加学习率调度器：训练后期降低学习率，细化结果
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

criterion = nn.MSELoss()

# ============================
# 4. 训练循环 (Training Loop)
# ============================
print("Training started...")
model.train() # 确保处于训练模式
for step in range(num_steps):
    # 简单的随机采样一个 batch
    indices = torch.randint(0, len(dataset), (batch_size,))
    x_0_batch = dataset[indices]
    
    # 1. 随机采样时间步 t
    t = torch.randint(0, num_timesteps, (batch_size,)).to(device)
    
    # 2. 生成要添加的随机噪声 epsilon (这是我们的训练目标 Target)
    noise = torch.randn_like(x_0_batch)
    
    # 3. 执行前向加噪过程，获得 x_t
    x_t = q_sample(x_0_batch, t, noise)
    
    # 4. 模型预测噪声
    predicted_noise = model(x_t, t)
    
    # 5. 计算 Loss：真实噪声与预测噪声的 MSE
    loss = criterion(predicted_noise, noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step() # 更新学习率
    
    if (step+1) % 500 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step [{step+1}/{num_steps}], Loss: {loss.item():.6f}, LR: {current_lr:.1e}")

print("Training finished.")

# ============================
# 5. 采样/推理过程 (Sampling)
# ============================
# 这是最激动人心的部分：从纯噪声开始，利用训练好的模型一步步还原数据
@torch.no_grad() # 推理模式，不需要计算梯度
def p_sample_loop(model, n_samples):
    # 从纯高斯噪声开始 (T时刻)
    img = torch.randn((n_samples, 2)).to(device)
    model.eval()
    for i in reversed(range(0, num_timesteps)):
        t = torch.full((n_samples,), i, dtype=torch.long, device=device)
        
        # 1. 模型预测当前时间步添加的噪声
        predicted_noise = model(img, t)
        
        # --- 以下是 DDPM 采样公式的简化实现 ---
        # 获取当前步的系数
        beta_t = betas[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(alphas[i])
        
        # 核心公式：根据预测的噪声，计算上一时刻的均值
        model_mean = sqrt_recip_alpha_t * (
            img - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise
        )

        # 添加一点点随机扰动 (除了最后一步 t=0)
        if i > 0:
            posterior_variance = beta_t * (1. - alphas_cumprod[i-1]) / (1. - alphas_cumprod[i])
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance) * noise
        else:
            img = model_mean
    return img.cpu().numpy()

print("Sampling...")
# 生成更多点以观察密度分布
generated_data = p_sample_loop(model, n_samples=2500) 

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Real Data")
plt.scatter(dataset.cpu().numpy()[:2500, 0], dataset.cpu().numpy()[:2500, 1], s=2, alpha=0.5, c='blue')
plt.xlim(-2, 2); plt.ylim(-2, 2)

plt.subplot(122)
plt.title("Generated Data (Improved Model)")
plt.scatter(generated_data[:, 0], generated_data[:, 1], s=2, alpha=0.5, c='green')
plt.xlim(-2, 2); plt.ylim(-2, 2)
plt.show()