# from opacus.accountants import RDPAccountant
# import pdb

# def calculate_epsilon(q, sigma, delta, steps):
#     accountant = RDPAccountant()
#     for _ in range(steps):
#         accountant.step(noise_multiplier=sigma, sample_rate=q)
#         epsilon, _ = accountant.get_privacy_spent(delta=delta)
#         print(epsilon)
#     return epsilon

# # 参数设置
# batch_size = 16
# num_data = 6000

# q = batch_size / num_data        # 采样率
# sigma = 0.5       # 噪声标准差
# max_norm = 1
# delta = 1e-4    # 隐私损失
# steps = 1000000    # 训练步骤数

# epsilon = calculate_epsilon(q, sigma*max_norm, delta, steps)
# print(f"After {steps} steps, the privacy budget is (ε = {epsilon:.2f}, δ = {delta})")

from opacus.accountants import RDPAccountant
import pdb

def calculate_epsilon(batch_size, num_data, sigma, delta, steps):
    accountant = RDPAccountant()
    dynamic = False
    last_epsilon = 0
    # shadow_epsilon = 0
    for step in range(steps):
        if dynamic == True:
            if step % 2 == 0:
                q = batch_size / num_data
            else:
                q = 1.0
        else:
            q = batch_size / num_data
        accountant.step(noise_multiplier=sigma, sample_rate=q)
        
        epsilon, _ = accountant.get_privacy_spent(delta=delta)
        
        
        # increased_epsilon  = epsilon - last_epsilon
        # shadow_epsilon += 2*increased_epsilon
        # last_epsilon = epsilon

        
        print(epsilon)
        # print(shadow_epsilon)
    
    return epsilon

def calculate_epsilon_test(batch_size, num_data, sigma, delta, steps):
    accountant1 = RDPAccountant()
    accountant2 = RDPAccountant()
    dynamic = False
    last_epsilon = 0
    # shadow_epsilon = 0
    for step in range(steps):
        if dynamic == True:
            if step % 2 == 0:
                q = batch_size / num_data
            else:
                q = 1.0
        else:
            q = batch_size / num_data

        accountant1.step(noise_multiplier=sigma, sample_rate=q)
        epsilon1, _ = accountant1.get_privacy_spent(delta=delta)
        
        accountant2.step(noise_multiplier=sigma, sample_rate=q)
        epsilon2, _ = accountant2.get_privacy_spent(delta=delta)
        # increased_epsilon  = epsilon - last_epsilon
        # shadow_epsilon += 2*increased_epsilon
        # last_epsilon = epsilon

        
        print(epsilon1+epsilon2)
        # print(shadow_epsilon)
    
    return epsilon1+epsilon2

# 参数设置
# way 1
batch_size = 16
num_data = 6000
sigma = 0.5       # 噪声标准差
max_norm = 1
delta = 1e-4    # 隐私损失
steps = 100   # 训练步骤数

epsilon1 = calculate_epsilon(batch_size, num_data, sigma*max_norm, delta, steps)
epsilon2 =  calculate_epsilon(batch_size, num_data, sigma*max_norm, delta, steps)
epsilon3 = calculate_epsilon_test(batch_size, num_data, sigma*max_norm, delta, steps)

print('Way1')
print(f"After {steps} steps, the privacy budget is (ε = {2*epsilon1:.2f}, δ = {delta})")
print('Way2')
print(f"After {steps} steps, the privacy budget is (ε = {epsilon1+epsilon2:.2f}, δ = {delta})")
print('Way3')
print(f"After {steps} steps, the privacy budget is (ε = {epsilon3:.2f}, δ = {delta})")

