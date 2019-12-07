---
layout: post
title:  "基于 OpenCV PCA 人脸识别"
categories: ML
tags: OpenCV PCA C++
author: hdb
comments: true
excerpt: "PCA 是一种用于数据降维的方法，常用于图像的压缩、人脸识别等。其原理并不复杂，但是其中的思想还是很有用的。"
mathjax: true
---

* content
{:toc}


## 数据准备

40 组人脸图像，类标号为 1-40，每组 6 张，总计 240 张。训练样本为每组的 5 张，共 200 张。测试样本为每组一张，共40 张。

## 函数设计

我编写的代码用到了以下函数：

```c++
//获取文本文件路径，将每一行的文字作为一个元素添加到vector中
vector<string> getList(const string& file);   
//用于显示Mat数据的函数（调试用）
void printEle(Mat_<float> m, int x, int y, int n);
//保存pca训练数据
void savePCAdata(PCA pca,Mat eigenface, Size size, vector<string> type, string file);   
//pca训练函数
void pcaTrain(const string& data, const string& path, const string& namelist, Size size, const string& typelist, int num = 0);   
 //pca测试函数
vector<string> pcaTest(const string& data, const string& path, const string& namelist, const string truetype = "");  
 //读取图像整理成标准的矩阵
Mat arrMat(const string& path, const string& namelist, Size size);  
//原图像与pca重建后图像显示
void compareFace(int i, const string& path,const string& file);   
```

## 函数详解

```c++
vector<string> getList(const string& file);
```

- 输入参数：
  - file：txt 文本文件，里面是提前做好的用于训练或测试的图像名称。

- 输出：
  - `vector<string>`的向量：该函数将每一个图像名称作为一个元素放进 vector 中，为了以后读取图像时提供方便。

```c++
void savePCAdata(PCA pca,Mat eigenface, Size size, vector<string> type, string file);
```

- 输入：
  - pca：PCA 对象
  - eigenface：特征脸
  - size：缩减后的图像尺寸，因为原图像尺寸很大，不进行缩减，运算量会很大
  - type：与每一幅图像对应，为类标号
  - file：保存的想XML文件名

> 该函数将输入的参数保存在 XML 文件中，因此，只需训练一次就可以。

```c++
void pcaTrain(const string& data, const string& path, const string& namelist, Size size, const string& typelist, int num = 0);  
```

- 该函数会调用 arrMat 函数、getList 函数、savePCAdata 函数。

执行过程：

- 调用 arrMat 函数，将图像整理成 PCA 所需要的矩阵
- 调用 getList 函数，获取每幅图像的类别
- 调用 OpenCV 的 PCA 函数，构造 PCA 对象
- 调用 PCA::project 函数，将原始图像投影到特征空间
- 调动 savePCAdata 函数保存训练数据

输入：

data：XML 文件名

path：图像存储路径

namelist：txt 文件，即图像名称列表

size：缩减后图像尺寸

typelist：存储类别的 txt 文件

num：要保留的主成分个数，默认 num=0 表示全部保留

```c++
vector<string> pcaTest(const string& data, const string& path, const string& namelist, const string truetype = "");  
```

函数执行过程：

- 构造 FileStorage 对象，加载 XML 中的数据
- 调用 arrMat 函数，将测试图像整理成标准矩阵
- 重构 PCA 对象
- 对测试数据投影到特征空间
- 构造测试样本特征脸与训练样本特征脸的距离矩阵
- 按照最邻近原则归类

输入参数：

data：XML 文件名

path：测试图像路径

namelist：测试图像名称 txt 清单

truetype：可选的测试样本的真实类别，若有该参数，会根据测试结果与真实结果计算识别的准确度

主要函数就这些了。

## 源代码

```c++
// cv3_pca.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "iostream"
#include "fstream"
#include "string"
#include "vector"

using namespace cv;
using namespace std;

vector<string> getList(const string& file);   //获取文本文件路径，将每一行的文字作为一个元素添加到vector中
void showEle(Mat_<float> m);   //用于显示Mat数据的函数（调试用）
void savePCAdata(PCA pca,Mat eigenface, Size size, vector<string> type, string file);   //保存pca训练数据
void pcaTrain(const string& data, const string& path, const string& namelist, Size size, const string& typelist, int num = 0);   //pca训练函数
vector<string> pcaTest(const string& data, const string& path, const string& namelist, const string truetype = "");   //pca测试函数
Mat arrMat(const string& path, const string& namelist, Size size);   //读取图像整理成标准的矩阵
void compareFace(int i, const string& path,const string& file);   //原图像与pca重建后图像显示

int main()
{
	string trainNameList = "D:\\我的文档\\Visual Studio 2013\\Projects\\cv3_pca\\cv3_pca\\trainNameList.txt"; //图像名称列表路径
	string typelist = "D:\\我的文档\\Visual Studio 2013\\Projects\\cv3_pca\\cv3_pca\\trainClassType.txt";
	string path1 = "D:\\我的文档\\Visual Studio 2013\\Projects\\cv3_pca\\cv3_pca\\train\\";
	Size size;
	size.width = size.height = 50; //缩减后的图像尺寸,因图像尺寸很大，不进行缩减，运算量太大
	string data = "pca.xml";
	pcaTrain(data, path1, trainNameList, size, typelist,0); //只需训练一遍，之后调用时，不用在执行，因为训练的数据已经保存
	string testNameList = "D:\\我的文档\\Visual Studio 2013\\Projects\\cv3_pca\\cv3_pca\\testNameList.txt";
	string testTrueType = "D:\\我的文档\\Visual Studio 2013\\Projects\\cv3_pca\\cv3_pca\\testTrueType.txt";
	string path2 = "D:\\我的文档\\Visual Studio 2013\\Projects\\cv3_pca\\cv3_pca\\test\\";
	vector<string> testType = pcaTest(data, path2, testNameList, testTrueType);
	compareFace(1, path1, trainNameList);
	return 0;
}

vector<string> getList(const string& file)
{
	ifstream ifs(file);
	string temp;
	vector<string> list;
	for (;!ifs.eof();)
	{
		getline(ifs, temp);
		list.push_back(temp);
	}
	ifs.close();
	return list;
}

void showEle(Mat_<float> m)
{
	float** tempt = new float*[m.rows];
	for (int i = 0; i < m.rows; i++)
	{
		float* num = m.ptr<float>(i);
		tempt[i] = new float[m.cols];
		for (int j = 0; j < m.cols; j++)
		{
			tempt[i][j] = num[j];
		}
	}

	for (int i = 0; i < m.rows; i++)
	{
		
		delete[] tempt[i];

	}
	delete[] tempt;
}


void savePCAdata(PCA pca, Mat eigenface, Size imgsize, vector<string> traintype, string file)
{
	FileStorage fs(file, FileStorage::WRITE); //创建XML文件  

	if (!fs.isOpened())
	{
		cerr << "failed to open " << "pca.xml" << endl;
	}	
	fs <<"eigenvalues" << pca.eigenvalues;
	fs << "eigenvectors" << pca.eigenvectors;
	fs << "mean" << pca.mean;
	fs << "eigenface" << eigenface;
	fs << "imgsize" << imgsize;
	fs << "traintype" << "[";
	for (int i = 0; i < traintype.size(); i++)
	{
		fs << traintype[i];
	}
	fs << "]";
	fs.release();
}

void pcaTrain(const string& data, const string& path, const string& namelist, Size size, const string& typelist, int num)
{
	//step1：加载训练图像进行预处理（线性变换，灰度化，类型转换）
	Mat trainMat = arrMat(path, namelist, size);
	vector<string> type = getList(typelist);

	//step2：调用pca的构造函数，
	PCA pca(trainMat, Mat(), CV_PCA_DATA_AS_ROW,num);
	string outfile = data;
	//求特征脸
	Mat traineigenface = pca.project(trainMat);

	savePCAdata(pca,traineigenface, size, type, outfile);

}

Mat arrMat(const string& path, const string& namelist, Size size)
{
	vector<string> list;
	list = getList(namelist);

	//step1：加载训练图像进行预处理（线性变换，灰度化，类型转换）
	int Mat_rows = list.size();	 //图片总数，亦trainMat的行
	int Mat_cols = size.height * size.width;		 //trainMat的列
	Mat xMat(Mat_rows, Mat_cols, CV_32FC1);  //为trainMat开辟空间
	for (int i = 0; i < list.size(); i++)
	{
		Mat temp = imread(path + list[i], 0); //加载图像单通道
		Mat temp_s;
		cv::resize(temp, temp_s, size, 0, 0, CV_INTER_AREA);
		Mat temp2;
		cv::normalize(temp_s, temp2, 0, 255, cv::NORM_MINMAX, CV_8UC1); //归一化处理
		Mat temp3;
		temp2.convertTo(temp3, CV_32FC1, 1.0 / 255.0); //转化为浮点数
		Mat temp4 = temp3.reshape(0, 1); //reshape
		xMat.row(i) = temp4 + 0;  //注意！！！！
	}
	return xMat;
}


vector<string> pcaTest(const string& data, const string& path, const string& namelist, const string truetype)
{
	FileStorage f(data, FileStorage::READ);
	Mat eigenvalues, eigenvectors, mean, traineigenface;
	Size imgsize;
	vector<string> traintype;
	f["eigenvalues"] >> eigenvalues;
	f["eigenvectors"] >> eigenvectors;
	f["mean"] >> mean;
	f["eigenface"] >> traineigenface;
	f["imgsize"] >> imgsize;
	FileNode n = f["traintype"];
	if (n.type() != FileNode::SEQ)
	{
		cerr << "发生错误，字符串不是一个序列" << endl;
		exit(1);
	}
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		traintype.push_back((string)*it);
	}
	f.release();

	Mat testMat = arrMat(path, namelist,imgsize);
	PCA pca;
	pca.eigenvalues = eigenvalues;
	pca.eigenvectors = eigenvectors;
	pca.mean = mean;
	Mat testeigenface = pca.project(testMat);

	vector<string> testTrueType;
	if (truetype != "")
	{		
		testTrueType = getList(truetype);
	}

	vector<string> testType;
	for (int i = 0; i < testeigenface.rows; i++)
	{
		double min_dis  = cv::norm(testeigenface.row(i), traineigenface.row(0), NORM_L2);
		int min_index = 0;
		for (int j = 0; j < traineigenface.rows; j++)
		{
			double dis = cv::norm(testeigenface.row(i), traineigenface.row(j), NORM_L2);
			if (dis < min_dis)
			{
				min_dis = dis;
				min_index = j;
			}
		}
		testType.push_back(traintype[min_index]);
	}
	int count = 0;
	for (int i = 0; i < testType.size(); i++)
	{
		if (testType[i] == testTrueType[i])
		{
			count++;
		}
	}
	float rate = (float)count / testType.size();
	int k = 1;
	for (vector<string>::iterator it = testType.begin(); it < testType.end(); it++)
	{
		cout << "第" << k << "张人脸属于第" << *it << "个人" << endl;
		k++;
	}
	cout << "The accurate rate is " << rate << endl;
	return testType;
}

void compareFace(int i,const string& path, const string& file)
{
	vector<string> trainNameList = getList(file);
	FileStorage fs("pca.xml",FileStorage::READ);
	Size size;
	PCA pca;
	Mat trianEigenface;
	fs["imgsize"] >> size;
	fs["eigenface"] >> trianEigenface;
	fs["eigenvalues"] >> pca.eigenvalues;
	fs["eigenvectors"] >> pca.eigenvectors;
	fs["mean"] >> pca.mean;
	
	fs.release();
	Mat trainOriginFace = imread(path+trainNameList[i]);
	namedWindow("trainOriginFace");
	imshow("trainOriginFace",trainOriginFace);
	Mat trainReconstFaceV, trainReconstFace;
	pca.backProject(trianEigenface.row(i), trainReconstFaceV);
	cv::resize(trainReconstFaceV.reshape(0,size.height), trainReconstFace, Size(trainOriginFace.cols, trainOriginFace.rows), 0.0, 0.0, cv::INTER_LINEAR);
	namedWindow("trainReconstFace");
	imshow("trainReconstFace", trainReconstFace);
	char key = cv::waitKey(0);
	if (key == 27)
	{
		return;
	}
}
```

## 总结

通过编写代码，熟悉了 OpenCV 的一些函数的用法：

### 关于 PCA 类

- 构造函数  

  ```c++
  PCA (InputArray data, InputArray mean, int flags, int maxComponents=0)
  ```

  输入参数依次是：

  输入的 PCA 样本矩阵，

  样本均值（可以不计算，写成 Mat()）

  flags 标志是一行是一个样本（DATA_AS_ROW），还是一列 （DATA_AS_COL ）

  主成分个数，0 为全部保留，即等于样本数

- PCA 成员：

  public 属性 eigenvalues 特征值 $N\times{1}$ 列，eigenvectors 特征向量（按行排列），mean 样本均值

  常用函数：

  ```c++
  Mat project(InputArray vec) const //将原始样本投影到特征空间
  Mat backProject(InputArray vec) const //将特征脸重构原图像，肯定跟原图像有差别
  ```

### 关于 `normalize` 函数

```c++
void cv::normalize	(	InputArray 	src,
						InputOutputArray 	dst,
						double 	alpha = 1,
						double 	beta = 0,
						int 	norm_type = NORM_L2,
						int 	dtype = -1,
						InputArray 	mask = noArray() 
					)	//对矩阵中的元素进行归一化，如果是矩阵的化是对整个矩阵归一化
```

- $ normType=NORM\_INF, NORM\_L1, or NORM\_L2$，使 $src$ 的无穷范数、1 范数、或 2 范数等于 $alpha$
- $normType=NORM\_MINMAX$，进行区间的归一化到$[alpha, beta]$

### 关于 `imread` 函数

```c++
Mat cv::imread(const string& filename, int MODE = IMREAD_COLOR);
MODE=IMREAD_UNCHANGE //不经处理加载原图像
IMREAD_GRAYSCALE	//强制转化为单通道灰度图
IMREAD_COLOR	//强制转化成3通道BGR图
...
```

### 关于 `norm` 函数

```c++
double cv::norm	(	InputArray 	src1,
					int 	normType = NORM_L2,
					InputArray 	mask = noArray() 
				)	//求src1的范数，2范数相当于欧式距离
  
double cv::norm	(	InputArray 	src1,
						InputArray 	src2,
						int 	normType = NORM_L2,
						InputArray 	mask = noArray() 
					)	//求src1与src2差的范数
```
### 关于 `Mat:: convertTo` 函数

```c++
void cv::Mat::convertTo	(OutputArray 	m,
						int 	rtype,
						double 	alpha = 1,
						double 	beta = 0 
						)		const
//对矩阵元素进行线性变换按照如下公式：
```


$$
m(x,y)=saturate_cast<rType>(α(∗this)(x,y)+β)
$$

### 关于 XML 数据的存储

```c++
//存储数据
Mat img = imread("lena.jpg", IMREAD_COLOR);
FileStorage fs("xxx.xml", FileStorage::WRITE);
if (!fs.isOpened())
{
  cerr << "failed to open " << "xxx.xml" << endl;
}
fs<<"img"<<img;	//存Mat
//存vector
fs<<"vec"<<"[";	
for (int i = 0; i < vec.size(); i++)
{
  fs << vec[i];
}
fs << "]";
//释放
fs.release();

 //读取xml数据
FileStorage fs("xxx.xml", FileStorage::READ);

Mat img;
fs["img"]>>img;	//读取Mat

//读取vector,要借助FileNode和FileNodeIterator
FileNode n = fs["traintype"];
if (n.type() != FileNode::SEQ)
{
  cerr << "发生错误，字符串不是一个序列" << endl;
  exit(1);
}
FileNodeIterator it = n.begin(), it_end = n.end();
for (; it != it_end; ++it)
{
  vec.push_back((string)*it);
}
fs.release();
```

### 关于 `resize` 函数

```c++

void cv::resize	(	InputArray 	src,
					OutputArray 	dst,
					Size 	dsize,//输出图像尺寸，若为Size(0,0),则根据以下两个参数确定
					double 	fx = 0,						//宽度放大倍数
					double 	fy = 0,						//高度放大倍数
					int 	interpolation = INTER_LINEAR 	//图像插值算法
				)	
```

- $interpolation=INTER\_AREA,INTER\_LINEAR,INTER\_CUBIC$ 
  缩小用area，放大用 linear

### 关于 `Mat:: reshape` 函数

```c++
//矩阵元素不变，改变行数、列数
Mat cv::Mat::reshape	(	int 	cn,		//通道
							int 	rows = 0 //返回矩阵的行数
						)		const
```
### 关于 `Mat:: row​` 函数

```c++
Mat cv::Mat::row	(	int y	)	const
//注意该函数只创建一个Mat头信息，不复制数据
Mat A;
...
A.row(i) = A.row(j); // 不会改变第i行的值
A.row(i) = A.row(j) + 0;	// 会改变第i行的值
A.row(j).copyTo(A.row(i));  // 会改变第i行的值
```
### C++中 `ifstream​` 读取文件

```c++
ifstream ifs(file);
string temp;
vector<string> list;
for (;!ifs.eof();)
{
  getline(ifs, temp); 	//调用string全局函数getline
  list.push_back(temp);
}
ifs.close();
```