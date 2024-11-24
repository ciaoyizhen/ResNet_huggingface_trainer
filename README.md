# 自写模型的huggingface的Trainer训练

说明: 为了练习使用huggingface的trainer训练自己的模型，从而写了该项目

## 该项目以ResNet50来适配
特别说明:
1. 一比一还原论文结构(通俗易懂)见src/model/py
2. 超参数没有跟原论文一致
3. 训练的代码是从[这里(原先的图片分类项目)](https://github.com/ciaoyizhen/image_classification)扒来修改的


## 步骤
0. pip install -r requirements.txt
1. python download_data.py
2. scripts/run_train.sh