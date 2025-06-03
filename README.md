#### 修改的部分有
`./beir/retrieval/search/dense/exact_search.py`:
1. 增加函数 `raw_score_text_no_batch`, `raw_score_text_no_batch_sym`
2. `DenseRetrievalExactSearch.search`中`# yzl_code`后到第一个`return`的部分
   这一段中使用了hnsw

在主文件夹中增加了hnsw.py，相对于原版的hnsw.py进行了一些修改，使得能够在search中自定义距离函数

#### 测试
使用`Try=1 python qa_webq.py`可以调用hnsw进行检索
