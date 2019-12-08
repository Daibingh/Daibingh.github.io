---
layout: post
title:  "tensorflow 安装教程及版本对应关系"
categories: 环境配置
tags: tensorflow
author: hdb
comments: true
excerpt: "tensorflow 安装教程，以及版本对应关系。"
mathjax: true
---

* content
{:toc}

## 版本对应关系

Linux GPU:

<center><img src="https://i.stack.imgur.com/Laiii.png" width="700px"></center>
Linux:

<center><img src="https://i.stack.imgur.com/TR2iI.png" width="700px"></center>
Windows:

<center><img src="https://i.stack.imgur.com/dOZtR.png" width="700px"></center>
## 安装方法

```
conda create -n tf-gpu python=3.6
conda activate tf-gpu
pip install --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade tensorflow-gpu==1.12
```
