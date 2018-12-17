# visualqa

## 需要的包
1. numpy
2. pandas
3. keras
4. pickle
5. skimage
6. json
7. tensorflow
8. matplotlib
## 数据来源
本文数据来源于著名的coco数据集，下载地址为http://cocodataset.org/#download
## 网络结构
### baseline model
模型结构：
![示意图](./picture/iBOWIMG.jpg)
iBOWIMG模型示意图

本文在iBOWIMG模型的基础上使用GRU对question进行编码。
![](./picture/baseline.png.jpg) 

本文使用的网络结构图

参考论文：[Simple Baseline for Visual Question Answering](https://arxiv.org/pdf/1512.02167.pdf)
### DMN+

![](./picture/DMN+.png.png)


参考论文：[Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/pdf/1603.01417.pdf)
### strong baseline model
![](./picture/strong_baseline.png)

![](./picture/strong_baseline.png.png)


参考论文：[Show, Ask, Attend, and Answer:A Strong Baseline For Visual Question Answering](https://arxiv.org/pdf/1704.03162.pdf)

## 精度比较

![](./picture/strong.jpg)
