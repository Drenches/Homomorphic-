from opacus.accountants import RDPAccountant
import pdb

def calculate_epsilon(q, sigma, delta, steps):
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=q)
        epsilon, _ = accountant.get_privacy_spent(delta=delta)
        print(epsilon)
    return epsilon

# 参数设置
batch_size = 16
num_data = 6000
q = num_data/batch_size

q = batch_size / num_data        # 采样率
sigma = 0.5       # 噪声标准差
max_norm = 1
delta = 1e-4    # 隐私损失
steps = 1000000    # 训练步骤数

epsilon = calculate_epsilon(q, sigma*max_norm, delta, steps, batch_size, num_data)
print(f"After {steps} steps, the privacy budget is (ε = {epsilon:.2f}, δ = {delta})")


