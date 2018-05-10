# 说明
使用gensim的LDA模型，参数分别使用默认参数与某公司给出的参数，主题数在代码中给出

代码写的很乱很丑，随意使用。

# 使用
将分词后的语料库放入corpus中，运行lda_train.py将训练LDA模型，在model目录下生成模型。同时会得到该模型LDAvis的可视化界面。

模型训练好后，运行write_list.py将会读取model下所有模型，并根据corpus提供的语料输出分类，分类文件生成在THETA目录下。
