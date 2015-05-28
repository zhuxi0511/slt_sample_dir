# 有监督机器学习平台工程简单目录样例

本例子为使用[supervise learning tool](https://github.com/zhuxi0511/supervised_learning_tool)的工程提供了一个样例目录，工程可参照此样例来规划目录。

* algorithm目录：放置学习算法以及预处理模块，请保持其中\__init__.py, pretreat.py extreat.py和pre的名称不变。
* conf目录：放置配置文件
* data目录：放置数据文件，注意原始数据文件需要以`*.train.raw`，`*.test.raw`为文件名存储在配置文件的data_dir目录下。
* log目录：放置日志文件
* model目录：放置模型文件
* output目录：放置输出文件
