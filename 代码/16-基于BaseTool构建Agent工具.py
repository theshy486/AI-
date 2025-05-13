from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4', temperature=0)


from tavily import TavilyClient
client = TavilyClient(api_key="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")


from pydantic import BaseModel, Field
from typing import Type

class SearchQuery(BaseModel):
    query: str = Field(description="要查询的query")


# 自定义Agent工具
from langchain_core.tools import BaseTool
class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = """通过搜索引擎来查询信息"""
    args_schema: Type[BaseModel] = SearchQuery

    def _run(self, query: str) -> dict:
        """调用工具"""
        return client.search(query, max_results=2)

tools = [TavilySearchTool()]

# 提示词
from langchain.prompts import PromptTemplate
prompt = PromptTemplate.from_template(
"""尽可能简约和准确地使用中文回应如下问题。您可以使用以下工具:
{tools}
使用以下格式：
Question：您必须回答的输入问题
Thought：你应该始终思考该做什么
Action：要采取的操作，应该是〔｛tool_names｝〕之一
Action Input：动作的输入
Observation：行动的结果
…（这个 Thought/Action/Action Input/Observation 可以重复N次）
Thought：我现在知道最终答案了
Final Answer：原始输入问题的最终答案
开始!
Question: {input}
Thought:{agent_scratchpad}
提醒！务必使用中文回答。
""")

from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, agent_kwargs={"prompt": prompt})
response = agent.invoke({"input": "美国现任总统是谁？"})
print(response)