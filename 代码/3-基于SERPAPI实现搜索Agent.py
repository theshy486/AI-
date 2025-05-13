import os  # 引入os模块
os.environ['SERPAPI_API_KEY'] = 'e14b42e676f3e2d06a9403d943f8bed11e4f45a05ad49c0ff11a03289629e7b1'

# 实例化大模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

# load_tools加载所需工具
from langchain_community.agent_toolkits.load_tools import load_tools
tools = load_tools(["serpapi"], llm=llm)


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

