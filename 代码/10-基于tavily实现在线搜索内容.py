import os
os.environ["TAVILY_API_KEY"] = "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"

# 1. 初始化搜索引擎客户端
from tavily import TavilyClient
# client = TavilyClient(api_key="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")
client = TavilyClient() # 如果不填写，则默认从本地系统变量中提取TAVILY_API_KEY的值作为key

# 2. 执行搜索
response = client.search(
    query="2024年第三季度中，中国与美国的GDP是多少?",
    search_depth="advanced",  # 搜索类型：advanced深度搜索/高级搜索，basic简单搜索
    max_results=5, # 返回搜索结果的文档数量
    days=3, # 如果指定了search_depth=advanced，则表示搜索指定天数内容的搜索结果
    include_domains=[],  # 指定包含的搜索站点域名
    exclude_domains=['qq.com'], # 指定排除的搜索站点域名
    include_answer=True, # 是否包含答案
    include_raw_content=True, # 是否包含原始内容
    include_images=True, # 是否包含图片
)

# 3. 获取结果，后续可以提供给大模型进行推理
print(response)