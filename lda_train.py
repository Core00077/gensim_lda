#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
import pyLDAvis.gensim
import time

# 语料目录
corpus_path = "corpus/"
# files = os.listdir(corpus_path)
files = ["tfidf_seg_0.1_0.2_full_reduce_digit.txt"]
# 主题数
topics = [10, 15, 20, 25]
# 参数
parameters = ["default", "company"]

total_start = time.time()
for filename in files:
    # 基路径
    save_base = "model/" + filename[:-4] + "/"
    for topic in topics:
        for parameter in parameters:
            # 模型路径
            save_path = str(topic) + "_by_" + parameter + "/"
            # 模型名字
            save_name = save_base + save_path + str(topic) + "_by_" + parameter
            if not os.path.exists(save_base + save_path):
                os.makedirs(save_base + save_path)
            start = time.time()
            print("训练主题数=" + str(topic))
            print("从" + filename + "读取文本...")
            with open(corpus_path + filename, encoding="utf-8") as f:
                train = f.readlines()
            train = [line.split(" ") for line in train]

            print("文本转语料...")
            dictionary = Dictionary(train)
            print("语料转词袋...")
            corpus = [dictionary.doc2bow(text) for text in train]
            print("开始训练...")
            lda = None
            if parameter == "default":
                lda = LdaModel(corpus, id2word=dictionary,
                               num_topics=topic)
            if parameter == "company":
                lda = LdaModel(corpus, id2word=dictionary,
                               num_topics=topic,
                               alpha=0.10, eta=0.02, iterations=5000)

            print("开始存储模型...")
            lda.save(save_name + ".lda")
            print("开始存储语料...")
            dictionary.save(save_name + ".dict")
            print("开始存储词袋化语料稀疏矩阵...")
            MmCorpus.serialize(save_name + ".mm", corpus=corpus)
            print("开始生成LDAvis可视化界面...")
            # lda = LdaModel.load(save_base + save_path + ".lda")
            # dictionary = Dictionary.load(save_base + save_path + ".dict")
            # corpus = MmCorpus(save_base + save_path + ".mm")
            vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
            pyLDAvis.save_html(vis, save_name + ".html")

            end = time.time()
            print("用时：" + str(end - start) + "s")

total_end = time.time()
print("总计用时：" + str(total_end - total_start) + "s")
