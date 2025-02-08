# 余弦相似度是一种常见的相似度计算方法，但它也有一些不足之处。以下是五种改进版本：

# 带TF-IDF权重的余弦相似度：这个版本考虑了词项在文档中的重要性，根据它们在文档集合中出现的频率来赋予它们不同的权重。
# 带BM25权重的余弦相似度：这个版本是基于BM25算法优化的，BM25算法考虑了查询词在文档中的分布情况以及它们在全局中的出现频率，然后根据这些因素给每个词项赋予不同的权重。
# 带LDA主题模型的余弦相似度：这个版本使用LDA主题模型将文档表示为多个主题的混合，然后计算主题之间的余弦相似度。
# 带LSI/LDA降维的余弦相似度：这个版本使用LSI或LDA降维方法将高维向量压缩到低维空间，然后计算压缩后的向量之间的余弦相似度。
# 带词向量模型的余弦相似度：这个版本使用预训练的词向量模型将文档表示为向量，然后计算向量之间的余弦相似度。


#%%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 构建文本集合
documents = ["I love machine learning", 
             "I hate studying",
             "Machine learning is fun",
             "Studying is boring"]

# 计算TF-IDF权重矩阵
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# # 计算余弦相似度矩阵
cosine_similarities = tfidf_matrix* tfidf_matrix.T
cosine_similarities.toarray()
#%%
from sklearn.metrics.pairwise import cosine_similarity
# 计算余弦相似度矩阵
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_similarities