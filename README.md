# 运行环境
```
python                        3.11.2
sklearn                       0.0.post5
transformers                  4.47.1
torch                         2.1.0
together                      1.2.7
spacy                         3.7.5
numpy                         1.24.3
pandas                        2.1.1
joblib                        1.2.0
```

# 程序入口

程序入口位于仓库根目录下的`code\main.py`中，在`code`文件夹下执行`python main.py`即可运行程序。

可选择的操作如下：
```
if __name__ == '__main__':
    # bert_train()
    # RQ1()
    # RQ2()
    # RQ4()
    # example()
    pass
```
包括对比`LQPR`和其他主流方法的性能、消融实验以及`LQPR`调用示例。
