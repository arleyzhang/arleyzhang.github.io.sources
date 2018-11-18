---
title: 目标检测-SNIPER-Efficient Multi-Scale Training-论文笔记
copyright: true
top: 100
mathjax: true
categories: deep learning
tags:
  - object detection
  - 论文笔记
description: 目标检测模型 SNIPER-Efficient Multi-Scale Training 论文笔记
abbrlink: f0c1556d
password:
---


> 论文：https://arxiv.org/abs/1805.09300
> 代码：https://github.com/MahyarNajibi/SNIPER

# 1 问题提出

这篇文章提出的SNIPER(Scale Normalization for Image Pyramids with Efficient Resampling)算法是一种多尺度（multi-scale）训练算法。提出的multi-scale算法不敢说很有创造性，但是是一种比较新颖的思想。

为了实现对图片上不同大小目标的检测，需要针对multi-scale提出有效方法。现在很多先进的目标检测算法，比如Faster R-CNN/Mask R-CNN，它们实现multi-scale的思路除了使用anchor的思想之外，一般还都包括一种常用的手段，那就是 image pyramid，这方法在早期的目标检测算法中是经常使用的方法。

如下图，image pyramid 和 anchor 方法实现multi-scale的思路展现的很清楚了。

![mark](/images/Paper-notes-sniper.assets/multi-scale.png)

现在大多数目标检测算法为了提高精度，除了使用anchor的思想外，一般都会把image pyramid的方法加进去，但是我们知道的，这种方法肯定会严重影响速度，不管是训练还是inference，不过但凡是用了这种方法的一般都是为了追求高精度，而暂时不注重速度。

image pyramid 影响速度的原因也很明了，每一个scale的图片的所有像素都参与了后续的计算。比如一个3scale的image pyramid（分别是原图的1倍，2倍，3倍），它要处理相当于原图14倍的像素.

在CVPR2018上有这样一篇文章，An Analysis of Scale Invariance in Object Detection – SNIP 。这篇文章曾经表述这样一个观点，在multi-scale training时，最好是将极端分辨率情况下的目标的梯度忽略。比如在 3scales的情况下，应该将较大分辨率图像上的较大目标和较小分辨率图像上的较小目标的梯度忽略。

关于scale的问题先回顾一下RCNN和Fast R-CNN

**R-CNN**

![](/images/Paper-notes-sniper.assets/RCNN.png)



R-CNN是比较老的目标检测框架了，但是很多现代新型的目标检测算法都是从它演变而来的。在R-CNN检测的第二步里对通过selective search生成的候选区域（region proposals）做了warped操作，一般是warp成224x224大小的图片，这个是CNN中经常使用的输入图片尺寸。因为R-CNN相当于一个分类问题，因此将不同大小的候选区域缩放成固定大小，实际上引入了尺度不变性 (scale invariant )，这也解决了 multi-scale 的问题，任何一个唯一尺寸的候选区域都相当于一个scale。由于候选区域是由selective search生成的，因此有一定的先验知识，换句话说每个不同大小的候选区域，其scale也代表了object的scale。虽然第3步输入的是固定大小的region proposals（看起来是没有考虑scale的），但是scale已经在提取候选区域时和缩放时完成了，实际上输入的warped region proposals是有scale的。它最主要的问题在于提取特征时，CNN需要在每一个候选区域上跑一遍，候选区域之间的交叠使得特征被重复提取, 造成了严重的速度瓶颈, 降低了计算效率。



**Fast R-CNN** 

![](/images/Paper-notes-sniper.assets/Fast-RCNN.png)

Fast RCNN的提出主要是解决了R-CNN的速度问题，它通过将selective search生成的候选区域映射到feature map上，实现了候选区域间的卷积计算共享，同时由于共同参与计算，Fast R-CNN也能更多的获取到全局信息（contex）。而同时提出的 ROI pooling也将对应到feature map上的候选区域池化到相同的大小，ROI pooling 在某种程度上也可以引进尺度不变性 (scale invariant )，因为可以将不同大小的region proposal（连同 object）在feature map层面池化到固定大小。但它的问题是前面的卷积层部分是只在一个scale上进行的，这正是为了实现卷积计算共享所带来的一点负面影响，因为是卷积计算共享，所以不同大小的目标没有办法进行不同比例的缩放操作，所以相对于原图大小来说，它们都处于一个scale。这影响了尺度不变性 (scale invariant )，因此为了弥补这一点，在训练或者inference时，会使用image pyramid的操作。

但是image pyramid有一个问题，就是在缩放原图时，object也会跟着缩放，这就导致原来比较大的object变得更大了，原来更小的object变得更小了。这在R-CNN中是不存在这个问题的。



**SNIPER 相当于综合了R-CNN 在scale上的优点 和Fast R-CNN在速度上的优点。** 



# 2 SNIPER

SNIPER(Scale Normalization for Image Pyramids with Efficient Resampling) 

**chips** 

首先介绍一下这篇文章提出的 chips  的概念，原文的解释是 scale specific context-regions (chips) that cover maximum proposals at a particular scale. 

**chips是某个图片的某个scale上的一系列固定大小的（比如KxK个像素）的以恒定间隔（比如d个像素）排布的小窗（window） ，每个window都可能包含一个或几个objects。**

这个跟sliding window有点像。



**Chip Generation**

首先是chips的生成：

假设有n个scale，{s1,s2,…,si,…sn}，每个scale的大小为 $W_i × H_i$ ，将原图缩放至对应比例后，$K×K$ 大小的chip（对于COCO数据集，论文用的512×512）以等间隔（d pixels）的方式排布，注意是每个scale都会生成这些chips，而且chip的大小是固定的，变化的是图片的尺寸。这跟anchor的思想刚好相反，因为anchor中，不变的是feature map，变化的是anchor。

实际上这篇论文输进网络用于训练的是这些chip，每个chip都是原图的一部分，包含了若干个目标。

一张图的每个scale，会生成若干个Positive Chips 和 若干个 Negative Chips 。每个Positive Chip都包含若干个ground truth boxes，所有的Positive Chips则覆盖全部有效的ground truth boxes。每个 Negative Chips 则包含若干个 false positive cases。这些在下面详细讲解。



**Positive Chip Selection** 

每个scale，都会有个area range $R^i =[r_{min}^i, r_{max}^i]，i \in [1,n]$ ，这个范围决定了这个scale上哪些ground truth box是用来训练的。所有ground truth box中落在 $R^i$ 范围内的ground truth box 称为有效的（对某个scale来说），其余为无效的，有效的gt box集合表示为 $G^i$ 。从所有chips中选取包含（完全包含）有效 gt box最多的chip，作为Positive Chip，其集合称为 $C^i_{pos}$  ，每一个gt box都会有chip包含它。因为 $R^i$ 的区间会有重叠，所以一个gt box可能会被不同scale的多个chip包含，也有可能被同一个scale的多个chip包含。被割裂的gt box（也就是部分包含）则保持残留的部分。

如下图，左侧是将gt box和不同scale上的 chip都呈现在了原图上，这里为了展示，将不同scale的图片缩放回原图大小，所以chip才有大有小，它们是属于不同scale的。右侧才是训练时真正使用的chips，它们的大小是固定的。可以看出这张图片包含了三个scale，所以存在三种chip，chip中包含有效的gt box（绿色）和无效的gt box（红色）。

![mark](/images/Paper-notes-sniper.assets/postive-chip-selection.png)

从原图也可以看出，原图中有一大部分的背景区域（没被postive chip覆盖的区域）被丢弃了，而且实际上网络真正输入的是chip（比原图分辨率小很多），这对于减少计算量来说很重要。

SNIPER 只用了4张低分辨率的chips 就完成了所有objects的全覆盖和multi-scale的训练方式。

这个地方很巧妙，在不同的scale上截取相同大小的chip，而且包含的是不完全相同的目标。这个方式很接近 RCNN。对较小的目标起到一种 zoom in 的作用，而对较大的目标起到一种 zoom out的作用。而对于网络来说，输入的还是固定大小的chip，这样的结构更加接近于分类问题网络的拓扑。

所以说 **SNIPER 相当于综合了R-CNN 在scale上的优点 和Fast R-CNN在速度上的优点。** 

这个地方太精妙了。



**Negative Chip Selection** 

如果至包含上述的 postive chip那么由于还是有大部分的区域背景，所以对背景的识别效果不好，导致假正例率比较高。因此需要生成 negative chip，来提高网络对背景的识别效果。现在的目标检测算法因为在multi-scale训练时，所有的像素都参与了计算，所以假正例率的问题相对这个算法没那么严重（但是计算量大），但是这里因为抛弃了很多背景，所以会使假正例率增加。为了降低假正例率和维持计算量不至于上涨太多，一个简单的解决办法就是使用一个精度不是很高的rpn生成一些不太准确的proposals，这些proposals中肯定有一大部分都是假正例。连这个精度不是很高的rpn都没有生成proposals的地方则有很大可能是背景区域，那么训练时把这一部分抛弃即可。

negative chip 的生成是这样的：首先使用训练集训练一个不是很精确的RPN，生成一系列 proposals，对每个scale i，移除那些被 positive chip 所覆盖的proposals，因为这部分proposal与gt box重合在一起了。选取包含至少M个在 $R^i$ 范围内的proposals的所有chips作为  negative chip，其集合称为 $C^i_{neg}$ 。

训练时，每张图片在每个epoch中只从   negative chip 池中选择固定数量的 negative chip。这个地方训练时怎么弄的，文章解释不是很清楚，恐怕得看代码才能明白。

如下图是   negative chip 的生成示例：

![mark](/images/Paper-notes-sniper.assets/negative-chip-selection.png)



**Label Assignment** 

回归时标签的赋值与 Faster R-CNN是类似的，文章说是端到端的训练，但是我个人觉得是没有把生成 positive/negative chip的过程算进去，这一部分好像是独立进行的，等chip生成后才在chip上进行 类似于 Fster R-CNN的训练。因为生成chip时用的RPN是不太精确的，跟训练检测网络时的RPN应该不能共享。不过这个地方具体是怎样的也只能看代码才能明白了。

其他的设置基本跟  Faster R-CNN 差别不大，只是正样本比例啊，OHEM啊，损失函数啊有所不同，这里不再赘述。



**Benefits** 

许多网络为了增加精度，都会使用image pyramid而且会使用较高分辨率的图片。SNIPER 则摆脱了对较高分辨率图像的依赖。而且由于SNIPER 训练时使用的是较低分辨率的chip，所以batch也可以做的很大，比如可以做到每张卡20的batch（不过大神用的都是tesla V100）。



# 3 实验结果

![mark](/images/Paper-notes-sniper.assets/AR.png)



![mark](/images/Paper-notes-sniper.assets/AP.png)



![mark](/images/Paper-notes-sniper.assets/ablation-study.png)



# 4 总结

- 这篇文章的 multi-scale 训练方式很新奇，但是可能在实现上会有点复杂
- 证明了在高分辨率图片上进行训练这一常用技巧并不是提升精度必须要使用的策略