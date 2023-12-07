from scipy.stats import norm
#from scipy.stats import ztest
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import scipy.stats as stats
# SET mean and standard deviation
mean = 44.175
sample_var = 1.017
sample_size = 100
std_dev = sample_var**0.5
# set threshold
threshold_valuemax = 46.5
threshold_valuemin = 41.5
# use CDF to calculate p
probability = 1 - norm.cdf(threshold_valuemax, loc=mean, scale=std_dev)
probability = norm.cdf(threshold_valuemin, loc=mean, scale=std_dev)

print(f"P(X<{threshold_valuemin} ): {probability:.4f}")
print(f"P(X>{threshold_valuemax} ): {probability:.4f}")

# one-sample T-test

# 设置随机数种子以确保结果可复现
#np.random.seed(188)

# 生成随机样本数据（正态分布）
#random_sample = np.random.normal(loc=mean, scale=std_dev, size=sample_size)

# 假设总体均值
#population_mean_hypothesis = 44

# 执行 t-检验
#t_stat, p_value = stats.ttest_1samp(random_sample, population_mean_hypothesis)

#print(f'T Statistic: {t_stat}')
#print(f'P Value: {p_value}')

# 总体统计信息
population_mean = 44
population_std = 2.5
# 计算标准误差（standard error）
standard_error = population_std / np.sqrt(sample_size)
# 计算 Z 统计量
z_stat = (mean - population_mean) / standard_error

# 计算 p 值（双侧检验）
p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

print(f'Z Statistic: {z_stat}')
print(f'P Value: {p_value}')
# 判断是否拒绝零假设
alpha = 0.05
if p_value < alpha:
    print("reject H0, μ≠xbar")
else:
    print("fail to reject H0, μ=xbar")

#interval
from scipy.stats import norm


# 设置置信水平和计算标准正态分布上的临界值
confidence_level = 0.95
z_critical = norm.ppf((1 + confidence_level) / 2)

# 假设你知道样本大小（如果未知，无法计算）
sample_size = 100

# 计算置信区间
confidence_interval_lower = mean - z_critical * (std_dev / np.sqrt(sample_size))
confidence_interval_upper = mean + z_critical * (std_dev / np.sqrt(sample_size))

# 输出结果
print(f"95% confidence interval: ({confidence_interval_lower:.2f}, {confidence_interval_upper:.2f})")
