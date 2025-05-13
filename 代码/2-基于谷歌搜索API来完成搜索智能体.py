import os  # 引入os模块
os.environ['SERPER_API_KEY'] = 'd8c90cb04db399c8057cd9f02f580c284e016ea4'



# 实例化大模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

# 原生加载搜索工具
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

# 提示词
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template(
"""Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}""")

# create_react_agent 创建一个ReAct风格智能体
from langchain.agents import create_react_agent, AgentExecutor
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input": "2021年~2024年，北京大学这三年中的总招生人数？"})
print(response)

