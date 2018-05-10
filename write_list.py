import os
import re
import time
from gensim.models import LdaModel
from gensim.corpora import Dictionary

start = time.time()
corpus_base = "corpus/"
model_base = "model/"
THETA_base = "THETA/"

corpus_list = os.listdir(corpus_base)
print("要分类的语料库:" + str(corpus_list))
for corpus in corpus_list:
    print("开始读取语料库:" + corpus)
    with open(corpus_base + corpus, encoding="utf-8") as f:
        train = f.readlines()
    print("读取完成,开始词袋化语料库……")
    train = [line.split() for line in train]
    dictionary = Dictionary(train)
    print("完成，开始加载语料库对应的LDA模型……")
    models = os.listdir(model_base + corpus[:-4] + "/")
    print("要加载的模型为:" + str(models))
    for model_name in models:
        print("开始加载模型:" + corpus[:-4] + "/" + model_name)
        model = LdaModel.load(
            model_base + corpus[:-4] + "/" +
            model_name + "/" +
            model_name + ".lda")  # type:LdaModel
        print("加载完成")
        topics = model.num_topics
        print("模型主题数为:" + str(topics))

        if not os.path.exists(THETA_base + corpus[:-4] + "/"):
            os.makedirs(THETA_base + corpus[:-4] + "/")
        output = open(THETA_base + corpus[:-4] + "/"
                      + model_name + ".THETA", "w", encoding="utf-8")
        print("生成THETA文件的路径为:" +
              THETA_base + corpus[:-4] + "/" + model_name + ".THETA")
        count = 0
        results = []
        print("开始生成THETA文件……")
        for doc in train:
            result = model[dictionary.doc2bow(doc)]
            resultList = []
            x = 0
            for i in range(topics):
                if x >= len(result) or result[x][0] != i:
                    resultList.append(0)
                else:
                    resultList.append(result[x][1])
                    x += 1
            results.append(" ".join(str(r) for r in resultList) + "\n")
            if count % 10000 == 0:
                output.writelines(results)
                results.clear()
                print(count)
            count += 1
        output.writelines(results)
        print(count)
        output.close()
end = time.time()
print("共计用时：" + str(end - start) + "s")
