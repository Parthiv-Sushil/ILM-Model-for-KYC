**中文版**|[English Version](README.md)  
![静默活体检测](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/logo.jpg)  
# 静默活体检测 (Silent-Face-Anti-Spoofing) & ILM KYC 扩展

该项目为[小视科技](https://www.minivision.cn/)的静默活体检测项目，并扩展了用于数字 KYC 身份验证的集成学习模型（ILM）。

## 简介
在工程中开源了活体模型训练架构，数据预处理方法，模型训练和测试脚本以及 ILM 集成模型与 KYC 风险评分引擎。

## 运行方法
1. 测试原版模型：
   `python test.py --image_name image_F1.jpg`
2. 运行 ILM 评估：
   `python eval_ilm.py`
3. 运行 KYC 演示：
   `python demo_kyc.py`
