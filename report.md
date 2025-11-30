Patch-wise Structural Loss for Time Series Forecasting
原文链接https://arxiv.org/abs/2503.00877
github链接https://github.com/Dilfiraa/PS_Loss

1、论文总结
问题：
现在的时间序列预测模型主要依赖MSE损失函数，其过度关注局部数值精度从而忽略了序列全局结构的相似性，导致预测结果在风功率等复杂场景下出现形状失真、波动模式失准等突出问题。
创新点：
本文提出了块状结构损失（PS Loss），通过多维度结构相似性度量（趋势、波动以及水平）、基于傅里叶变换的自适应分块策略和梯度动态加权机制，在保持数值精度的同时显著提升了预测序列的结构相似性。该损失函数具备强通用性，可嵌入各类预测模型，并能自适应平衡不同结构特征的优化权重。
ps loss结构的流程图：
<img width="554" height="159" alt="image" src="https://github.com/user-attachments/assets/ad68996e-d95c-4832-b32d-f63cacbd3473" />

该方法包含三个核心组件：（1）基于傅里叶的自适应补丁划分（Fourier-based Adaptive Patching）——对真实序列Y和预测序列Ŷ进行自适应分段，生成补丁序列；（2）补丁级结构损失（Patch-wise Structural Loss）——通过融合相关性损失（LCorr）、方差损失（LVar）和均值损失（LMean），衡量补丁间的局部相似度；（3）基于梯度的动态加权（Gradient-based Dynamic Weighting）——根据各损失组件的梯度幅度，动态调整其权重（α、β、γ），确保优化过程的平衡性。最终，PS损失（LPS）与均方误差（MSE）损失（LMSE）无缝融合，以提升预测精度。2论文公式与程序代码对应
由于这个结构能够适用于目前出现的所有时间序列预测结构，所以在本次研究中我打算用itransformer模型接上ps loss来进行实验。
2.1傅里叶自适应划分
<img width="280" height="35" alt="image" src="https://github.com/user-attachments/assets/70bde5bf-af2a-4795-9f15-2b9a372710c5" />

<img width="284" height="44" alt="image" src="https://github.com/user-attachments/assets/da50a96b-6cc8-4cae-b285-9e2511cbd9f3" />

对应的代码：
def fouriour_based_adaptive_patching(self, true, pred):
# 对真实序列做FFT，提取频率特征
    true_fft = torch.fft.rfft(true, dim=1)  #使用FFT保留正频率
    frequency_list = torch.abs(true_fft).mean(0).mean(-1)  #计算各频率的平均幅值
    frequency_list[:1] = 0.0  #过滤直流分量（因为0频率无周期信息）
    top_index = torch.argmax(frequency_list)  #找到幅值最大主导频率
    period = (true.shape[1] // top_index)  #主导周期T=序列长度/主导频率
    #补丁长度=min(周期/2,阈值)，步长=补丁长度/2
patch_len = min(period // 2, self.patch_len_threshold)
stride = patch_len // 2
    # 调用划分函数，生成真实/预测序列的划分（论文的划分操作）
true_patch = self.create_patches(true, patch_len, stride=stride)  
pred_patch = self.create_patches(pred, patch_len, stride=stride)  
return true_patch, pred_patch

该代码通过快速傅里叶变换（FFT）分析真实序列的频率特征，找到主导周期T，分块长度设为patch_len=min(T/2,threshold)，步长设为stride=patch_len/2（确保补丁重叠50%，避免局部信息丢失）。

2.2线性相关性损失LCorr
论文公式：
<img width="284" height="44" alt="image" src="https://github.com/user-attachments/assets/c4c5a8e2-74db-4f9e-bdb9-12cf81a51e70" />

对应的代码：
def patch_wise_structural_loss(self, true_patch, pred_patch):
    # 计算补丁均值（用于协方差计算）
    true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)  #真实补丁均值
    pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)  #预测补丁均值
    
    # 计算补丁标准差（公式中的σ_y、σ_hat(y)）
    true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)  #真实方差
    pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)  #预测方差
    true_patch_std = torch.sqrt(true_patch_var)  #真实标准差σy
    pred_patch_std = torch.sqrt(pred_patch_var)  #预测标准差σhat(y)
    
# 计算协方差 Cov(y,hat(y))
    true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)  
    
    # 线性相关性损失
    patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)  #相关系数
linear_corr_loss = (1.0 - patch_linear_corr).mean()  #LCorr=1-相关系数（取平均）

其中：
Cov(y,ŷ)：真实序列与预测序列的协方差；σy、σŷ：真实序列与预测序列的标准差；ε：防止分母为0的微小值。
相关系数范围为[-1,1]，当预测与真实序列趋势完全一致时，相关系数=1，损失=0；趋势相反时损失接近2，损失越小越好。

2.3方差损失LVar
论文公式思想：通过KL散度（Kullback-Leibler Divergence，KL散度）衡量预测补丁与真实补丁的方差分布差异，确保两者波动幅度一致。
<img width="302" height="59" alt="image" src="https://github.com/user-attachments/assets/5c9f9564-ccf5-4fbc-8eb3-087079d898d5" />

化简为：
<img width="298" height="50" alt="image" src="https://github.com/user-attachments/assets/94a8a618-a66a-46a2-8258-de2bd466f797" />

相关代码：
# 方差损失：用softmax归一化后，通过KL散度计算分布差异
true_patch_softmax = torch.softmax(true_patch, dim=-1)  #真实补丁归一化（转为概率分布）
pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)  #预测补丁对数归一化
# KL散度：衡量两个分布的距离（公式：LVar=KL(pred_dist||true_dist)）
var_loss=self.kl_loss(pred_patch_softmax,true_patch_softmax).sum(dim=-1).mean()

核心逻辑：softmax函数将补丁内的数值转为概率分布，可突出方差差异（方差大的序列，数值分布更分散）；KL散度值越小说明两个分布越接近，预测与真实序列的波动特性越一致。

2.4均值损失 L_Mean
论文公式：
<img width="225" height="45" alt="image" src="https://github.com/user-attachments/assets/a24f2bc9-1c4a-4e8d-b745-fc6f0fb165e2" />

其中：
μy、μŷ：真实补丁与预测补丁的均值

相关代码：
# 均值损失
mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()

2.5三个结构损失之和
PS Loss由3个互补的结构损失加权组成：
<img width="310" height="28" alt="image" src="https://github.com/user-attachments/assets/a6795f9b-47dc-4da5-ad88-192dffed6800" />

论文代码：
def ps_loss(self, true, pred):
    #自适应补丁划分
    true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred) 
    #计算3个分损失
    corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)  
    #梯度动态加权
    alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)
    #PS总损失（论文公式：L_PS=α·LCorr+β·LVar+γ·LMean）
ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss 
    return ps_loss
其中：LCorr：线性相关性损失（确保趋势一致）；LVar：方差损失（确保波动一致）；LMean：均值损失（确保中心趋势一致）；α、β、γ：梯度动态加权系数。

2.6梯度动态加权（α、β、γ计算）
论文公式思想：
权重由各损失项的梯度幅度自适应调整：梯度越大，说明该损失项当前优化空间越大，权重越高；同时引入线性相似度和方差相似度，进一步优化均值损失的权重。
论文代码：
def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):
    # 调整维度，计算全局线性相似度和方差相似度
    true = true.permute(0, 2, 1)
    pred = pred.permute(0, 2, 1)
    true_mean = torch.mean(true, dim=-1, keepdim=True)
    pred_mean = torch.mean(pred, dim=-1, keepdim=True)
    true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
    pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
    true_std = torch.sqrt(true_var)
    pred_std = torch.sqrt(pred_var)
    true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
    
    #线性相似度sim_l=(1+相关系数)/2（映射到[0,1]）
    linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
    linear_sim = (1.0 + linear_sim) * 0.5 
    #方差相似度sim_v=2σ_yσ_hat(y)/(σ_y²+σ_hat(y)²)（映射到[0,1]）
    var_sim = (2*true_std*pred_std + 1e-5) / (true_var + pred_var + 1e-5) 
    
    # 计算各损失项的梯度范数（公式中的||gCorr||、||gVar||、||gMean||）
    #需通过model.projector参数反向传播梯度
    corr_gradient = torch.autograd.grad(corr_loss, self.model.projector.parameters(), create_graph=True)[0]  
    var_gradient = torch.autograd.grad(var_loss, self.model.projector.parameters(), create_graph=True)[0]  
    mean_gradient=torch.autograd.grad(mean_loss,self.model.projector.parameters(), create_graph=True)[0]
    gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0平均梯度ḡ
    # 动态权重计算
    aplha=gradiant_avg.norm().detach()/corr_gradient.norm().detach() #α=ḡ/||g_Corr||
    beta=gradiant_avg.norm().detach()/var_gradient.norm().detach()  #β=ḡ/||g_Var||
    gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()  #γ基础值
    gamma = gamma * torch.mean(linear_sim*var_sim).detach()  #γ修正
return aplha, beta, gamma

3安装说明
1.安装Pytorch和必要的依赖。
pip install -r requirements.txt
2.先配置无PS Loss的损失函数的模型初始参数
创造一个.bat文件，输入以下代码：
"E:\anaconda\python.exe" run.py --is_training 1 --root_path ../datasets/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 --model iTransformer --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --d_model 256 --d_ff 256 --learning_rate 0.0001 --train_epochs 3 --patience 3 --lradj type1 --use_ps_loss 0 --ps_lambda 0.0 --patch_len_threshold 24 --itr 1
解释：使用ETTh1电力变压器温度数据集进行多变量预测（M）。实验采用96个历史时间步长（seq_len）来预测未来的96个时间步长（pred_len），模型配置包含2层编码器（e_layers），隐藏维度（d_model）为256，前馈网络维度（d_ff）为256，输入和输出为7个特征维度（enc_in/dec_in/c_out）。训练使用0.0001的学习率（learning_rate）进行3次训练（train_epochs），采用type1学习率调整策略（lradj）和早停机制（patience=3）。由于要进行有PS Loss和没有使用PS loss的对比实验，所以暂时不用PS Loss功能（use_ps_loss=0），PS Loss权重（ps_lambda）设为0.0，仅使用传统的MSE损失函数进行训练，Patch长度阈值（patch_len_threshold）设为24，实验迭代次数（itr）为1次。这样可以建立一个纯MSE损失函数计算。
3.配置有PS Loss的损失函数的模型初始参数
创造一个.bat文件，输入以下代码：
"E:\anaconda\python.exe" run.py --is_training 1 --root_path ../datasets/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96_PS3 --model iTransformer --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --d_model 256 --d_ff 256 --learning_rate 0.0001 --train_epochs 10 --patience 3 --lradj type1 --use_ps_loss 1 --ps_lambda 3.0 --patch_len_threshold 24 --itr 1
专门测试PS Loss的有效性。实验使用ETTh1电力变压器温度数据集进行多变量预测（M），以前96个时间步长（seq_len）预测未来96个时间步长（pred_len）。模型配置包含2层编码器（e_layers），隐藏维度（d_model）和前馈网络维度（d_ff）均为256，处理7个特征维度（enc_in/dec_in/c_out）。这次启用了PS Loss功能（use_ps_loss=1）并设置PS Loss权重（ps_lambda）为3.0，同时设置Patch长度阈值（patch_len_threshold）为24。训练使用0.0001的学习率（learning_rate）进行10个训练轮次（train_epochs），采用type1学习率调整策略（lradj）和早停机制（patience=3），实验迭代次数（itr）为1次。

4运行结果说明
分别运行上面两个.bat文件，模型开始训练。
MSE：
<img width="553" height="312" alt="image" src="https://github.com/user-attachments/assets/7ea811fe-7d79-4746-9747-4b4e961079a5" />

<img width="553" height="312" alt="image" src="https://github.com/user-attachments/assets/67a6e8c0-bfd6-48a1-b184-cff02b12fb25" />

PS Loss
<img width="553" height="312" alt="image" src="https://github.com/user-attachments/assets/1ba1eeb7-957f-4c2e-86ca-a8647173b628" />

<img width="553" height="312" alt="image" src="https://github.com/user-attachments/assets/92dc12c0-d1fa-4fed-9666-ace22b32070e" />

<img width="553" height="312" alt="image" src="https://github.com/user-attachments/assets/3cca93c3-437a-4530-bc48-4b48ce16f21e" />

对比：我们可以看到，有ps loss的系统的误差函数小于无ps loss系统，并且预测的图像更加贴合真实的数据。
