import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

# llm
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 加载文档,可换成PDF、txt、doc等其他格式文档
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = TextLoader('../大模型应用开发与私有化部署.md', encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_language(language="markdown", chunk_size=250, chunk_overlap=80)
texts = text_splitter.create_documents(
    [documents[0].page_content]
)

# 选择向量模型，并灌库
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
db = FAISS.from_documents(texts, OpenAIEmbeddings(model="text-embedding-ada-002"))
# 获取检索器，选择 top-2 相关的检索结果
retriever = db.as_retriever(search_kwargs={"k": 2})

# 创建带有 system 消息的模板
from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个AI管家。
               你的任务是根据下述给定的已知信息回答用户问题。
               确保你的回复完全依据下述已知信息。不要编造答案。
               请用中文回答用户问题。
               已知信息:
               {context} """),
    ("user", "{question}")
])

# 定义RetrievalQA链
from langchain.chains import RetrievalQA
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 使用stuff模式将上下文拼接到提示词中
    chain_type_kwargs={"prompt": prompt_template}, # 自定义的提示词参数
    retriever=retriever # 检索算法
)

# 构建 FastAPI 应用，提供服务
app = FastAPI()

# 可选，前端报CORS时
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# 定义请求格式模型
class QuestionRequest(BaseModel):
    question: str

# 定义响应格式模型
class AnswerResponse(BaseModel):
    answer: str

# 提供查询接口 http://127.0.0.1:8000/ask
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # 获取用户问题
        user_question = request.question
        print(user_question)
        # 通过RAG链生成回答
        answer = rag_chain.run(user_question)
        # 返回答案
        answer = AnswerResponse(answer=answer)
        print(answer)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)