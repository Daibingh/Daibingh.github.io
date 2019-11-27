---
layout: post
title:  "Lenovo Y7000 装 win 10 Ubuntu 双系统"
categories: 环境配置
tags: 装机 Windows Ubuntu 
author: hdb
comments: true
excerpt: "Lenovo Y7000 装 win 10 Ubuntu 双系统配置过程。"
mathjax: true
---

* content
{:toc}

## Win 10

### 下载系统镜像

下载 win 10 镜像，win 10 企业版 2019 . 
- [MSDN I tell you](https://msdn.itellyou.cn/)
- 迅雷地址
	```
ed2k://|file|cn_windows_10_enterprise_ltsc_2019_x64_dvd_d17070a8.iso|4290967552|9E80DED85693E8E4E0D76E55B1207221|/
	```

### 制作 U 盘启动盘

使用 UltraISO 加载镜像到虚拟光驱，插入优盘，写入硬盘镜像到优盘。注意优盘模式确保为 FAT32.

### 笔记本 BIOS 设置

重启电脑，按 F2 进入 BIOS 设置。

- secure boot 设置为 disable
- usb boot 设置为 enable
- 开启 uefi 模式

### 装系统

重启电脑，从优盘引导启动，进入装机程序

### 分区（可选）

到分区界面，按 shift + F10，进入命令行界面，输入以下命令，

```
diskpart # 启动分区管理工具
list disk # 显示所有磁盘和编号
select disk n # 选择要分区的磁盘
clean # 清除磁盘所有数据
convert gpt # 硬盘转成 gpt 格式
```

再回到之前的界面，点击“新建”进行分区。分区的大小为 1024 的整数倍。

### 激活系统


以管理员身份打开 dos 窗口，
```bat
@echo off
title Activate Win10
slmgr.vbs /upk
slmgr /ipk M7XTQ-FN8P6-TTKYV-9D4CC-J462D
slmgr /skms kms.03k.org
slmgr /ato
slmgr /dlv
echo ---------: Your OS has ben successfully activated! :---------
pause
```

### 参考

>https://www.landiannews.com/archives/51131.html<br>https://blog.csdn.net/baidu_38432732/article/details/80896980#commentBox

## Ubuntu

### 步骤

- 下载镜像，写入硬盘镜像到优盘
- 设置 BIOS 为 uefi 模式
- 安装系统
  
    - boot loader 选择整个硬盘（让 Ubuntu 引导 Windows） 
- 修复显卡驱动冲突造成卡顿
    - 重启，当出现引导选择时，按 “e”，进入 grub 编辑界面。
    - 在 splash 后面空格，再添加以下内容
      
        - ... quiet splash **nouveau.modeset=0** $vt_handoff
    - 进入系统，`sudo gedit /etc/default/grub`，修改其中一行
        ```
        GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nouveau.modeset=0"
        ```
    - `sudo update-grub`

### 解决 WiFi 禁用问题
    ```
    rfkill list all
    sudo modprobe -r ideapad_laptop
    # 以下设置开机自动执行，避免每次开机手动执行以上命令
    sudo gedit /etc/rc.local
    # 添加
    echo <password> | sudo modprobe -r ideapad_laptop
    # 以上措施若不管用，尝试以下步骤
    sudo gedit /etc/modprobe.d/ideapad.conf
    # 添加
    blacklist ideapad_laptop
    ```

### 备注：Ubuntu 分区
    - `/` : 30G
    - `swap` : == RAM
    - `home` : rest of the disk