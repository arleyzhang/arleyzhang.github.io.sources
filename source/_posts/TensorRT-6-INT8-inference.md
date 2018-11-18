---
title: TensorRT(6)-INT8 inference
copyright: true
top: 100
mathjax: true
categories: deep learning
tags:
  - TensorRT
  - inference
description: TensorRT INT8 inference 官方例程
abbrlink: 95d15d89
date: 2018-09-03 01:23:53
password:
---

这一节通过官方例程 介绍 INT8 inference mode.

例程位于 `/usr/src/tensorrt/samples/sampleINT8` ，是基于mnist的，大体流程是一致的。

流程同样是 build(Calibration )->deploy，只不过在build时多了一个校准的操作。

注意以下几点：

# 1 网络定义

定义网络时，注意这个地方传进去的dataType，如果使用FP16 inference 则传进去的是FP16，也就是kHALF；但如果是使用INT8 inference的话，这个地方传进去的是kFLOAT，也就是 FP32，这是因为INT8 需要先用FP32的精度来确定转换系数，TensorRT自己会在内部转换成INT8。

```c++
const IBlobNameToTensor* blobNameToTensor = 
    parser->parse(locateFile(deployFile).c_str(),
                  locateFile(modelFile).c_str(),
                   *network,
                   DataType::kFLOAT);
```

这个看起来就跟使用FP32是一样的流程，INT8 MODE inference的输入和输出都是 FP32的。

（After the network has been built, it can be used just like an FP32 network, for example, inputs and outputs remain in 32-bit floating point.）

# 2 校准网络-Calibrating The Network 

校准网络时，比较麻烦的是校准集的构建，作者定义了一个BatchStream  class来完成这个操作。BatchStream类有个成员函数getBatch ()是为了依次读取 batch file 中的数据的。

还有个校准类 Int8EntropyCalibrator，继承自 NvInfer.h 中的 IInt8EntropyCalibrator

```c++
class Int8EntropyCalibrator : public IInt8EntropyCalibrator
```

这个类里面也有个 getBatch () 成员函数，实际上调用的是 BatchStream类的getBatch () ，然后将数据从内存搬到了显存，如下：

```c++
bool getBatch(void* bindings[], const char* names[], int nbBindings) override
{
    if (!mStream.next())
        return false;

    CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], INPUT_BLOB_NAME));
    bindings[0] = mDeviceInput;
    return true;
}
```

这个getBatch () 成员函数在校准时会被反复调用。

生成校准集时，校准集的样本应该是已经进行过一系列预处理的图片而不是原始图片。

校准类 Int8EntropyCalibrator 和 BatchStream 类的实现说起来比较麻烦，在后面源码解读部分直接结合注释看源码吧。

# 3 builder的配置-Configuring The Builder  

只需要在原来builder的基础上添加以下：

```c++
builder->setInt8Mode(true);
builder->setInt8Calibrator(calibrator);
```

# 4 batch file的生成-Batch Files For Calibration  

例程使用的batch file 已经制作好了，位于`<TensorRT>/data/mnist/batches`  这是一系列二进制文件，每个文件包含了 N 个图片样本，格式如下：

- 首先是4个32 bit的整形值，代表 {N, C, H, W},batchsize和图片dims
- 然后是N个 {C, H, W}维度的浮点数据，代表N个样本

batch file二进制文件的生成有两种方式：

## 4.1 使用caffe生成

主要对于使用caffe的用户，这里干脆直接将官方文档上的说明拷贝过来好了，比较简单：

> 1. Navigate to the samples data directory and create an INT8 mnist directory:
>
>    ```shell
>    cd <TensorRT>/samples/data
>    mkdir -p int8/mnist
>    cd int8/mnist
>    ```
>
>    Note: If Caffe is not installed anywhere, ensure you clone, checkout, patch, and build Caffe at the specific commit:
>
>    ```shell
>    git clone https://github.com/BVLC/caffe.git
>    cd caffe
>    git checkout 473f143f9422e7fc66e9590da6b2a1bb88e50b2f
>    patch -p1 < <TensorRT>/samples/mnist/int8_caffe.patch
>    mkdir build
>    pushd build
>    cmake -DUSE_OPENCV=FALSE -DUSE_CUDNN=OFF ../
>    make -j4
>    popd
>    ```
>
> 2. Download the mnist dataset from Caffe and create a link to it:
>
>    ```shell
>    bash data/mnist/get_mnist.sh
>    bash examples/mnist/create_mnist.sh
>    cd .. 
>    ln -s caffe/examples .
>    ```
>
> 3. Set the directory to store the batch data, execute Caffe, and link the mnist files:
>
>    ```shell
>    mkdir batches
>    export TENSORRT_INT8_BATCH_DIRECTORY=batches
>    caffe/build/tools/caffe test -gpu 0 -iterations 1000 -model examples/mnist/lenet_train_test.prototxt -weights
>    <TensorRT>/samples/mnist/mnist.caffemodel
>    ln -s <TensorRT>/samples/mnist/mnist.caffemodel .
>    ln -s <TensorRT>/samples/mnist/mnist.prototxt .
>    ```
>
> 4. Execute sampleINT8 from the bin directory after being built with the following command:
>
>    ```shell
>     ./sample_int8 mnist
>    ```



## 4.2 其他方式生成

对于不用caffe或者模型难以转换成caffemode的用户，首先要进行一系列预处理，然后按照前面提到的batch file格式生成二进制batch file文件，但这个生成过程要自己写了，不过写的话应该也比较简单，可以参考caffe中的patch文件中的核心部分：

```c++
#define LOG_BATCHES_FOR_INT8_TESTING 1
#if LOG_BATCHES_FOR_INT8_TESTING
  static int sBatchId = 0;
  char* batch_dump_dir = getenv("TENSORRT_INT8_BATCH_DIRECTORY");
  if(batch_dump_dir != 0)
  {
    char buffer[1000];
    sprintf(buffer, "batches/batch%d", sBatchId++);
    FILE* file = fopen(buffer, "w");    
    if(file==0)
      abort();

    int s[4] = { top_shape[0], top_shape[1], top_shape[2], top_shape[3] };
    fwrite(s, sizeof(int), 4, file);
    fwrite(top_data, sizeof(float), top_shape[0]*top_shape[1]*top_shape[2]*top_shape[3], file);
    fwrite(&top_label[0], sizeof(int), top_shape[0], file);
    fclose(file);
  }
+#endif
```

添加上数据集的读取，划分和预处理就可以了。

# 5 校准算法

从INT8的例程来看，TensorRT 支持两种方式的校准，一种就是上节我们讲过的使用相对熵的方式，还有一种是废弃的校准算法，校准时需要设置两个参数 cutoff 和 quantile，以下是 在GTC2017 上对INT8校准原理进行讲解的 Szymon Migacz 对废弃的校准算法的解读：

> https://devtalk.nvidia.com/default/topic/1015108/cutoff-and-quantile-parameters-in-tensorrt/
>
> Parameters cutoff and quantile have to be specified only for "legacy" calibrator. It's difficult to set values of cutoff and quantile without running experiments. Our recommended way was to run 2D grid search and look for optimal combination of (cutoff, quantile) for a given network on a given dataset. This was implemented in sampleINT8 shipped with TensorRT 2.0 EA.
>
> New entropy calibrator doesn't require any external hyperparameters, and it determines quantization thresholds automatically based on the distributions of activations on calibration dataset. In my presentation at GTC I was talking only about the new entropy calibrator, it's available in TensorRT 2.1 GA.

Szymon Migacz并没有充分的解释这两个参数，而是说这是 "legacy" calibrator中才会用到的参数，而且在没有做充分的试验的情况下，是很难合理地设置这两个参数的。他推荐的做法是 针对特定的网络结构和数据集使用 2D 网格搜索 来确定这两个参数的取值。而 entropy calibrator ，就是使用相对熵的校准方法，不需要任何超参数，而且能够根据校准集上的激活值分布自动确定量化阈值。NVIDIA官方也推荐使用使用相对熵校准的方式。所以 "legacy" calibrator 就不深入研究了。

# 6 源码解读

sampleINT8.cpp:

```c++
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <chrono>
#include <iterator>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

#include "BatchStream.h"
#include "LegacyCalibrator.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;

static Logger gLogger;

// stuff we know about the network and the caffe input/output blobs

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const char* gNetworkName{nullptr};

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs;
    dirs.push_back(std::string("data/int8/") + gNetworkName + std::string("/"));
    dirs.push_back(std::string("data/") + gNetworkName + std::string("/"));
    return locateFile(input, dirs);
}

bool caffeToTRTModel(const std::string& deployFile,		// name for caffe prototxt
	const std::string& modelFile,						// name for model
	const std::vector<std::string>& outputs,			// network outputs
	unsigned int maxBatchSize,							// batch size - NB must be at least as large as the batch we want to run with)
	DataType dataType,
	IInt8Calibrator* calibrator,
	nvinfer1::IHostMemory *&trtModelStream)
{
	//创建一个builder，传入自己实现的 gLogger 对象，为了打印信息用
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	//创建一个 network 对象，并创建一个 ICaffeParser 对象，这个对象是用来进行模型转换的；此时的 network 对象里面还是空的
	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();

	//判断当前的硬件平台是否支持 INT8 精度和 FP16 精度，两者都不支持的话，直接返回 false
	if((dataType == DataType::kINT8 && !builder->platformHasFastInt8()) || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
		return false;
    // caffemodel到tensorrt的转换, 注意这个地方传进去的dataType，
	// 如果使用FP16 inference 则传进去的是FP16，也就是kHALF
	// 如果是使用INT8 inference的话，这个地方传进去的是kFLOAT也就是 FP32,
	// 因为INT8 需要先用FP32的精度来确定转换系数，TensorRT自己会在内部转换成INT8
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
		locateFile(modelFile).c_str(),
		*network,
		dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);

	//标志输出tensor
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	// 设置最大 batchsize和工作空间大小 2^30 ,这里是1G
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 30);
	// 设置平均迭代次数和最小迭代次数，这是测量每一层时间的一种策略，即多次迭代求平均值，不过这里只迭代一次
	builder->setAverageFindIterations(1);
	builder->setMinFindIterations(1);
	//同步调试
	builder->setDebugSync(true);
	//INT8 MODE or/and FP16 MODE
	builder->setInt8Mode(dataType == DataType::kINT8);
	builder->setFp16Mode(dataType == DataType::kHALF);
	//设置INT8校准接口
	builder->setInt8Calibrator(calibrator);

	// 创建engine
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	//销毁无用对象
	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	//序列化到磁盘上，这里实际上是在内存中，没有保存到磁盘
	// serialize the engine, then close everything down
	trtModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	return true;
}

float doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	//从context恢复engine
	const ICudaEngine& engine = context.getEngine();
	//创建engine的时候，会把输入blob和输出blob指针放进去，engine.getNbBindings() 就是为了获取输入和输出的blob数目，以便于做检查
	//比如这里，就只有一个输入和一个输出，所以 检查时可以这样检查 assert(engine.getNbBindings() == 2);
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	//每个输入和输出blob都需要申请显存，故：void* buffers[engine.getNbBindings()];
	void* buffers[2];
	float ms{ 0.0f };

	//为了将 buffer中的成员(指针或者地址)分别与输入/输出的blob相关联，需要分别获取输入输出blob在engine中的索引
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	//计算输入输出shape
	// create GPU buffers and a stream
	Dims3 inputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(INPUT_BLOB_NAME)));
	Dims3 outputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));

	//计算实际的输入输出大小,申请显存
	size_t inputSize = batchSize*inputDims.d[0]*inputDims.d[1]*inputDims.d[2] * sizeof(float), outputSize = batchSize *
    outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);
	CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
	CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

	//从Host (CPU) 拷贝输入数据到 Device(GPU)，也就是从内存到显存
	CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice));

	//创建一个 cuda 异步流
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	//创建一个cuda事件
	cudaEvent_t start, end;
	CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
	//标记stream流，start
	cudaEventRecord(start, stream);
	//异步执行inference，//标记stream流，end
	context.enqueue(batchSize, buffers, stream, nullptr);
	cudaEventRecord(end, stream);
	//事件同步
	cudaEventSynchronize(end);
	//计算start事件和end事件之间的运行时间
	cudaEventElapsedTime(&ms, start, end);
	//销毁事件
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//从Device(GPU) 拷贝输出数据到 Host (CPU)，也就是从显存到内存
	CHECK(cudaMemcpy(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost));
	//释放显存
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	//销毁流对象
	CHECK(cudaStreamDestroy(stream));
	//返回inference时间
	return ms;
}

//计算一个batch 中 top-1或top-5的正确的图片数量
//对于输出来说，一张图片的输出对应一个 outputSize 维的向量（比如mnist是10维的）
//然而对于标签来说一张图片的标签是一个0-9之间的数字
//batchProb是一个batch中的标签向量按顺序叠加到一个vector中的，10个数字一组对应一张图片
//label就这这个batch的标签向量，一个数字对应一张图片
//outputsize是输出维度（比如mnist的outputsize=10）
//threshold：两个取值：1，对应top-1；5对应top-5
int calculateScore(float* batchProb, float* labels, int batchSize, int outputSize, int threshold)
{
	int success = 0;
	for (int i = 0; i < batchSize; i++)
	{
		//获取每个batch的地址，并获取预测向量中与标签相同位置上的真实概率
		//举个例子：假设threshold=1
		//i=0时，prob[0]-prob[9]是batch中的第一张图片的预测输出向量，
		//假设prob[0]-prob[9]的值为{0.1, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05}，这张图片的label是1.
		//那么correct = prob[(int)labels[i]]=prob[1]=0.5，之后判断的是这个correct是否在top-1或者top-5范围内
		//做法是：统计 prob[0]-prob[9]之间比correct更大的值的个数 better，因为如果比correct大的话，最终输出的肯定是错的预测结果；
		//但是由于top-1，top-5允许你出错的次数分别为1次和5次，所以只要 better < threshold，就认为预测准确，success++；
		//最后返回success，代表这个batch中按照 top-1 或 top-5的精度来算，预测对了几张图片。
		float* prob = batchProb + outputSize*i, correct = prob[(int)labels[i]];

		int better = 0;
		for (int j = 0; j < outputSize; j++)
			if (prob[j] >= correct)
				better++;
		if (better <= threshold)
			success++;
	}
	return success;
}



class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
	Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true)
		: mStream(stream), mReadCache(readCache)
	{
		DimsNCHW dims = mStream.getDims();
		mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();
		//为 mDeviceInput 申请显存，跳过前面 firstBatch 个batch
		CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
		mStream.reset(firstBatch);
	}

	/**
	 * 析构函数，释放显存
	 */
	virtual ~Int8EntropyCalibrator()
	{
		CHECK(cudaFree(mDeviceInput));
	}

	int getBatchSize() const override { return mStream.getBatchSize(); }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		if (!mStream.next())
			return false;

		//将mStream.getBatch()获取到的数据拷贝到 mDeviceInput 中，也就是从内存到显存
		CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
		assert(!strcmp(names[0], INPUT_BLOB_NAME));
		bindings[0] = mDeviceInput;
		return true;
	}

	/**
	 * 从文件中读取校准数据，返回校准表缓存地址
	 * @param length 读取长度
	 */
	const void* readCalibrationCache(size_t& length) override
	{
		//首先清空mCalibrationCache
		mCalibrationCache.clear();
		//从文件中读取内容并放到 mCalibrationCache vector中
		std::ifstream input(calibrationTableName(), std::ios::binary);
		input >> std::noskipws;
		if (mReadCache && input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		//返回 mCalibrationCache 地址或 空指针
		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	/**
	 * 将校准数据存储到文件中
	 * @param cache 校准数据内存地址
	 * @param length 数据长度
	 */
	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(calibrationTableName(), std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

private:
	/**
	 * 存储校准数据的文件
	 * @return 文件名称
	 */
    static std::string calibrationTableName()
    {
        assert(gNetworkName);
        return std::string("CalibrationTable") + gNetworkName;
    }
    //batch流
	BatchStream mStream;
	//是否从文件中读取校准数据
	bool mReadCache{ true };

	//校准时 GPU接受 的 数据量mInputCount 和 数据内容 mDeviceInput
	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	//存放从文件中读取到的校准数据，也就是scale_factor 缩放系数
	std::vector<char> mCalibrationCache;
};

/**
 * 用于模型评分，包含了caffe模型向ensorRT的转化以及inference的执行
 * @param batchSize 批尺寸
 * @param firstBatch 跳过初始的一些batch
 * @param nbScoreBatches 测试的 batch总数
 * @param datatype 以何种精度inference
 * @param calibrator 校准接口
 * @param quiet 是否输出调试信息
 */
std::pair<float, float> scoreModel(int batchSize, int firstBatch, int nbScoreBatches, DataType datatype, IInt8Calibrator* calibrator, bool quiet = false)
{
	IHostMemory *trtModelStream{ nullptr };

	// 调用 caffeToTRTModel 将caffe模型解析为TensorRT
	bool valid = false;
    if (gNetworkName == std::string("mnist"))
        valid = caffeToTRTModel("deploy.prototxt", "mnist_lenet.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, batchSize, datatype, calibrator, trtModelStream);
    else
        valid = caffeToTRTModel("deploy.prototxt", std::string(gNetworkName) + ".caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, batchSize, datatype, calibrator, trtModelStream);

    // 如果GPU不支持某种精度类型，比如FP16/INT8，则返回（0,0）
	if(!valid)
	{
		std::cout << "Engine could not be created at this precision" << std::endl;
		return std::pair<float, float>(0,0);
	}

    assert(trtModelStream != nullptr);

    // 恢复创建engine，创建上下文环境
	// Create engine and deserialize model.
	IRuntime* infer = createInferRuntime(gLogger);
    assert(infer != nullptr);
	ICudaEngine* engine = infer->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
	trtModelStream->destroy();
	IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    //创建 batch 流对象，并跳过开始的一些batch，共firstBatch个，此处等于100
	BatchStream stream(batchSize, nbScoreBatches);
	stream.skip(firstBatch);

	// output tensor 维度
	Dims3 outputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));
	//确定输出 tensor 数据量大小
	int outputSize = outputDims.d[0]*outputDims.d[1]*outputDims.d[2];
	int top1{ 0 }, top5{ 0 };
	float totalTime{ 0.0f };

	//每张图片都有一个 outputSize 大小的向量(比如 mnist 分类大小为10)，那么一个batch的输出应该为 batchSize * outputSize
	std::vector<float> prob(batchSize * outputSize, 0);

	//依次对不同的batch进行inference，stream.next()获取下一个batch
	while (stream.next())
	{
		//输入数据：stream.getBatch()，输出数据：prob 每循环一次就对一个batch的数据进行测试，这个batch的输出放在 prob 中
		totalTime += doInference(*context, stream.getBatch(), &prob[0], batchSize);

		//对每个batch，按照top-1和top-5精度来计算准确率
		top1 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 1);
		top5 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 5);

		//读取10个batch输出一个点，读取800个输出一个换行符
		std::cout << (!quiet && stream.getBatchesRead() % 10 == 0 ? "." : "") << (!quiet && stream.getBatchesRead() % 800 == 0 ? "\n" : "") << std::flush;
	}

	//统计总共读到了多少张图片，并计算top-1和top-5正确率
	int imagesRead = stream.getBatchesRead()*batchSize;
	float t1 = float(top1) / float(imagesRead), t5 = float(top5) / float(imagesRead);

	// 精度和时间，结果输出
	if (!quiet)
	{
		std::cout << "\nTop1: " << t1 << ", Top5: " << t5 << std::endl;
		std::cout << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and " << totalTime / stream.getBatchesRead() << " ms/batch." << std::endl;
	}

	//销毁无用对象，返回准确率
	context->destroy();
	engine->destroy();
	infer->destroy();
	return std::make_pair(t1, t5);
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Please provide the network as the first argument." << std::endl;
		exit(0);
	}
	gNetworkName = argv[1];

	//前 firstScoreBatch 个 batch是用来作为校准集的，因此在测试时这些是不进行测试的
	int batchSize = 100, firstScoreBatch = 100, nbScoreBatches = 400;	// by default we score over 40K images starting at 10000, so we don't score those used to search calibration
	//search变量是LEGACY_CALIBRATION校准算法中使用的变量，具体作用要看 LegacyCalibrator.h 源码，因为这个校准算法nvidia已经不推荐使用了，所以这里不深究了
	bool search = false;

	//校准算法 选择参考 Nvinfer.h 文件，kENTROPY_CALIBRATION：使用信息熵进行校准；kLEGACY_CALIBRATION，使用以前遗留下来的算法进行校准
	// 	enum class CalibrationAlgoType : int
	// {
	// 	kLEGACY_CALIBRATION = 0,
	// 	kENTROPY_CALIBRATION = 1
	// };
	CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION;

	// 处理命令行参数
	for (int i = 2; i < argc; i++)
	{
		if (!strncmp(argv[i], "batch=", 6))
			batchSize = atoi(argv[i] + 6);
		else if (!strncmp(argv[i], "start=", 6))
			firstScoreBatch = atoi(argv[i] + 6);
		else if (!strncmp(argv[i], "score=", 6))
			nbScoreBatches = atoi(argv[i] + 6);
		else if (!strncmp(argv[i], "search", 6))
			search = true;
		else if (!strncmp(argv[i], "legacy", 6))
			calibrationAlgo = CalibrationAlgoType::kLEGACY_CALIBRATION;
		else
		{
			std::cout << "Unrecognized argument " << argv[i] << std::endl;
			exit(0);
		}
	}

	if (calibrationAlgo == CalibrationAlgoType::kENTROPY_CALIBRATION)
	{
		search = false;
	}

	//batchsize不能大于128，这是为何？
	if (batchSize > 128)
	{
		std::cout << "Please provide batch size <= 128" << std::endl;
		exit(0);
	}

	//感觉这里写错了，应该是 50000
	if ((firstScoreBatch + nbScoreBatches)*batchSize > 500000)
	{
		std::cout << "Only 50000 images available" << std::endl;
		exit(0);
	}

	//设置标准输出流输出的精度
	std::cout.precision(6);

	//用于构建校准集的batch流
	//CAL_BATCH_SIZE = 50;NB_CAL_BATCHES = 10; 定义在 LegacyCalibrator.h文件中, 既然废弃了 LegacyCalibrator，为什么不把常量定义在本文件中
	BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);

	//FP32精度不需要校准集，因此最后一个参数传入 nullptr
	std::cout << "\nFP32 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
	scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kFLOAT, nullptr);

	//FP16精度不需要校准集，因此最后一个参数传入 nullptr
	std::cout << "\nFP16 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
	scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kHALF, nullptr);

	std::cout << "\nINT8 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
	if (calibrationAlgo == CalibrationAlgoType::kENTROPY_CALIBRATION)
	{
		//先构建校准集，然后调用scoreModel进行模型评估，创建engine时传入了Int8EntropyCalibrator对象calibrator
		//FIRST_CAL_SCORE_BATCH = 100; 定义在 LegacyCalibrator.h文件中
		Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH);
		scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kINT8, &calibrator);
	}
	else
	{
        //被废弃的校准算法，不解释了
		std::pair<double, double> parameters = getQuantileAndCutoff(gNetworkName, search);
		Int8LegacyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, parameters.first, parameters.second);
		scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kINT8, &calibrator);
	}

	shutdownProtobufLibrary();
	return 0;
}
```



BatchStream.h，这个源码看起来还是稍微有点费劲的，还是我C++功底不够啊，得补。。。

```c++
#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include "NvInfer.h"

std::string locateFile(const std::string& input);

class BatchStream
{
public:
	//构造函数，使用 batchSize 和 maxBatches 初始化 BatchStream 中的 mBatchSize(批尺寸) 和 mMaxBatches(批数量)
	BatchStream(int batchSize, int maxBatches) : mBatchSize(batchSize), mMaxBatches(maxBatches)
	{
		//读取第一个batch文件的shape，用于一系列初始化操作
		FILE* file = fopen(locateFile(std::string("batches/batch0")).c_str(), "rb");
		int d[4];
		fread(d, sizeof(int), 4, file);
		mDims = nvinfer1::DimsNCHW{ d[0], d[1], d[2], d[3] };
		fclose(file);
		//单张图片的大小（总的像素个数）
		mImageSize = mDims.c()*mDims.h()*mDims.w();
		//根据batch文件中的单张图片大小mImageSize初始化 BatchStream 中的 mBatch 的内存空间，初值为0；同理根据mBatchSize初始化mLabels
		//mBatch指的是BatchStream中的batch，batch的个数为mBatchSize，所以数据量总数为mBatchSize*mImageSize，
		//mLabels是BatchStream中的label，总数就是 mBatchSize
		mBatch.resize(mBatchSize*mImageSize, 0);
		mLabels.resize(mBatchSize, 0);

		//有两块专门的内存区域用于存储读取到的batch{i} 文件内容，就是下面两个。这两块内存区域里的内容在后面会被复制到 mBatch和mLabels中
		//mFileBatch指的是读取到的 batch{i} 文件中的batch，因此总数为mDims.n()*mDims.c()*mDims.h()*mDims.w()=mDims.n()*mImageSize
		//mFileLabels指的是读取到的 batch{i} 文件中的label，因此总数为 mDims.n()
		mFileBatch.resize(mDims.n()*mImageSize, 0);
		mFileLabels.resize(mDims.n(), 0);
		reset(0);
	}

	// reset操作
	void reset(int firstBatch)
	{
		mBatchCount = 0;
		mFileCount = 0;
		mFileBatchPos = mDims.n();
		skip(firstBatch);
	}

	/**
	 * stream.next()每调用一次，就使用batch file中的数据(读取后首先是变量名为mFileBatch的buffer)填充一个mBatch
	 * @return 是否填充成功
	 */
	bool next()
	{
		//已经读取到 最大 批数量 了，返回false
		if (mBatchCount == mMaxBatches)
			return false;

		// 将mFileBatch（相当于buffer）中的内容拷贝到mBatch中，
        //由于mFileBatch和mBatch大小有可能不一样,所以才这么写
		for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
		{
			assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
			//调用update函数，读取batches文件夹中的 batch{i} 文件，读取失败的话直接在这里返回false，
			//调用update函数会使 mFileBatchPos=0，这是合理的，因为还没有开始往 mBatch 拷贝数据
			if (mFileBatchPos == mDims.n() && !update())
				return false;

			//一次从batch文件中读取 csize 张图片，
			//由于mFileBatch和mBatch大小有可能不一样所以借助 mFileBatchPos 和 batchpos 来指示batch文件和mbatch中的当前操作(读取或存储)位置
			//所以csize取二者之间较小值
			// copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
			csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
			//将 mFileBatch 和 mFileLabels 中存放的batch文件的内容复制到 mBatch 和 mLabels 中
			std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
			std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
		}
		// mBatchCount自增，指示当前填充了多少个mBatch
		mBatchCount++;
		return true;
	}

	/**
	 * 跳过前面多少个batch
	 * @param skipCount 跳过的batch的个数
	 */
	void skip(int skipCount)
	{
		//如果mBatchSize 大于等于 mDims.n()，并且 mBatchSize%mDims.n() == 0，
		//换句话说batchsteam中的batchsize(比如100)，比batch{i}文件的batchsize(比如50)大，并且能整除.
		//那么batchstream中一个 batch， 相当于 mBatchSize / mDims.n()个batch 个batch{i}文件
		//举个例子：batchsteam中batchsize=100，batch{i}文件中batchsize=50，那么batchsteam中一个batch相当于 两个batch{i}文件
		//那么在batchstream中跳过一个 batch， 相当于跳过 mBatchSize / mDims.n() 个 batch{i}文件
		//所以才有 mFileCount += skipCount * mBatchSize / mDims.n();
		//这时直接通过修改mFileCount的数值来读取剩下的batch文件
		if (mBatchSize >= mDims.n() && mBatchSize%mDims.n() == 0 && mFileBatchPos == mDims.n())
		{
			mFileCount += skipCount * mBatchSize / mDims.n();
			return;
		}

		//其他情况：batchsteam中的batchsize不能整除batch{i}文件的batchsize
		//循环调用 next() 读取batch{i}文件，读取skipCount个，由于next() 会改变 mBatchCount 的值，所以先暂存，再取出
		int x = mBatchCount;
		for (int i = 0; i < skipCount; i++)
			next();
		mBatchCount = x;
	}

	//获取batchsteam中的 batch 和 label 的首地址， batch文件中的内容读取后首先是放在 mFileBatch 和 mFileLabels 中，
	//但最终会被复制到 mBatch和mLabels中，校准使用的就是 mBatch 和mLabels，而不是直接从batch file中读取进来的mFileBatch和mFileLabels
	float *getBatch() { return &mBatch[0]; }
	float *getLabels() { return &mLabels[0]; }
	//mBatchCount表示填充了多少个 mBatch 的数量
	//mBatchSize表示填充mBatch时使用的batchsize
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	//获取图片的shape信息，这个在mBatch和mFileBatch中是一样的
	nvinfer1::DimsNCHW getDims() const { return mDims; }
private:
	//batch文件（如batch0）中的图像数据和标签数据存放在 mFileBatch 和 mFileLabels 中，此处返回他们的地址
	float* getFileBatch() { return &mFileBatch[0]; }
	float* getFileLabels() { return &mFileLabels[0]; }

	//此函数用于依次读取 batches文件夹下的 batch{i} 文件，并将读取到的内容存放在mFileBatch和mFileLabels中，读取成功返回true，否则返回false
	bool update()
	{
		//依次读取 batches文件夹下的 batch{i} 文件，mFileCount变量自增，指向下一个batch文件也就是 batch{i+1} 文件
		std::string inputFileName = locateFile(std::string("batches/batch") + std::to_string(mFileCount++));
		FILE * file = fopen(inputFileName.c_str(), "rb");
		if (!file)
			return false;

		//从batch文件读取当前 batch 的 shape 信息（图像数据的shape）
		int d[4];
		fread(d, sizeof(int), 4, file);
		assert(mDims.n() == d[0] && mDims.c() == d[1] && mDims.h() == d[2] && mDims.w() == d[3]);

		//从batch文件读取图像数据（精度为float，大小为mDims.n()*mImageSize ），存放到 mFileBatch 中
		//从batch文件读取标签数据（精度为float，大小为mDims.n()），存放到mFileLabels中
		size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.n()*mImageSize, file);
		size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.n(), file);;
		assert(readInputCount == size_t(mDims.n()*mImageSize) && readLabelCount == size_t(mDims.n()));

		fclose(file);
		//每读取一个batch文件，mFileBatchPos置零，也就是说新读取的batch文件内容 mFileBatch 还没有开始往 mBatch 拷贝
		mFileBatchPos = 0;
		//读取成功返回true
		return true;
	}

	//stream中的批尺寸和最大批数量，每填充一个mBatch，mBatchCount 自增1
	int mBatchSize{ 0 };
	int mMaxBatches{ 0 };
	int mBatchCount{ 0 };

	//mFileCount指向batches文件夹中的batch文件，就跟指针一样，读完一个batch，自增1
	//mFileBatchPos在一个batch中当前操作的位置
	int mFileCount{ 0 }, mFileBatchPos{ 0 };
	//batchstream中的图片大小，一般要求跟batch文件中的大小一致，初值为0
	int mImageSize{ 0 };

	//batch文件中的数据的shape
	nvinfer1::DimsNCHW mDims;
	// 从 batch文件 中读到的图像数据和标签数据最终要放到这里来，这个是最终校准时使用的
	std::vector<float> mBatch;
	std::vector<float> mLabels;
	//用以存取 从 batch文件 中读到的图像数据和标签数据，相当于buffer
	std::vector<float> mFileBatch;
	std::vector<float> mFileLabels;
};


#endif
```

# 7 结果

```shell
myself@admin:~/workspace/study/tensorrt/bin$ ./sample_int8 mnist
FP32 run:400 batches of size 100 starting at 100
........................................
Top1: 0.9904, Top5: 1
Processing 40000 images averaged 0.00167893 ms/image and 0.167893 ms/batch.

FP16 run:400 batches of size 100 starting at 100
Engine could not be created at this precision

INT8 run:400 batches of size 100 starting at 100
........................................
Top1: 0.9908, Top5: 1
Processing 40000 images averaged 0.0013438 ms/image and 0.13438 ms/batch.
```

从这例程中也忽然发现在TensorRT中 1080ti GPU竟然不支持 FP16 mode，虽然1080ti官方的参数上是支持 float16的，但是在TensorRT中竟然不能使用。查了一下，是因为 1080ti的float16 吞吐量太低（throughput），效率太低，应该是TensorRT对float16也进行了条件限制，吞吐量太低的不支持。

从资料中得知，只有 Tesla P100, Quadro GP100, and Jetson TX1/TX2 支持 full-rate FP16 performance，应该也就只有这些才支持 TensorRT的FP16吧。新出的 TITAN V 加了tensor core，float16半精度性能有很大提升，应该也支持？不过有意思的是jetson TX1和 TX2 却能支持 FP16，反而不支持INT8.

可以参考下面资料：

> [FP16 --half=true option doesn't work on GTX 1080 TI although it runs ./sample_int8 INT8](https://devtalk.nvidia.com/default/topic/1023096/gpu-accelerated-libraries/fp16-half-true-option-doesn-t-work-on-gtx-1080-ti-although-it-runs-sample_int8-int8-/)
> [FP16 support on gtx 1060 and 1080](https://devtalk.nvidia.com/default/topic/1023708/gpu-accelerated-libraries/fp16-support-on-gtx-1060-and-1080/) 
>
> The only GPUs with full-rate FP16 performance are Tesla P100, Quadro GP100, and Jetson TX1/TX2. All GPUs with compute capability 6.1 (e.g. GTX 1050, 1060, 1070, 1080, Pascal Titan X, Titan Xp, Tesla P40, etc.) have low-rate FP16 performance. It's not the fast path on these GPUs. All of these GPUs should support "full rate" INT8 performance, however.

从结果上看：

INT8 MODE：Top 1 0.9908， 速度：0.0013438 ms/image ；

FP32 MODE : Top 1 0.9904，速度：0.00167893 ms/image；

准确率竟然还高那么一点点，速度上大概快了20%。



# 参考

1. [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#int8_sample)
2. [cutoff and quantile parameters in TensorRT](https://devtalk.nvidia.com/default/topic/1015108/cutoff-and-quantile-parameters-in-tensorrt/) 
3. [FP16 --half=true option doesn't work on GTX 1080 TI although it runs ./sample_int8 INT8](https://devtalk.nvidia.com/default/topic/1023096/gpu-accelerated-libraries/fp16-half-true-option-doesn-t-work-on-gtx-1080-ti-although-it-runs-sample_int8-int8-/)
4. [FP16 support on gtx 1060 and 1080](https://devtalk.nvidia.com/default/topic/1023708/gpu-accelerated-libraries/fp16-support-on-gtx-1060-and-1080/) 