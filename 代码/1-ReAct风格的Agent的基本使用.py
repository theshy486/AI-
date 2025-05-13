# 实例化大模型
# cursor编辑器
# # 获取当前langchian中内置提供的所有Agent工具函数
# from langchain_community.agent_toolkits.load_tools import get_all_tool_names
# print(get_all_tool_names())

# 实例化大模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

# # 原生调用tools工具
from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]

# 通过load_tools工具函数加载wikipedia代理工具
from langchain_community.agent_toolkits.load_tools import load_tools
tools = load_tools(['wikipedia'])

# 提示词
from langchain.prompts import PromptTemplate

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

# 实例化ReAct智能体
# create_react_agent 创建一个ReAct风格智能体
from langchain.agents import create_react_agent, AgentExecutor
agent = create_react_agent(llm, tools, prompt)

# 基于create_react_agent提供的Agent创建代理执行器，只支持使用一个参数
# verbose=True列出大模型执行过程的执行细节
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input": "2024年北京大学的招生人数？"})
print(response)
