import os
os.environ["TAVILY_API_KEY"] = "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"

from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=5)]

# 导入 LangChain Hub
from langchain import hub
# 从 LangChain Hub中获取 ReAct的提示
prompt = hub.pull("hwchase17/react")

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:0.5b")

# 导入 create_react_agent 功能
from langchain.agents import create_react_agent
# 构建 ReAct Agent
agent = create_react_agent(llm, tools, prompt)
# 导入 AgentExecutor
from langchain.agents import AgentExecutor
# 创建 Agent 执行器并传入 Agent 和工具
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 调用 AgentExecutor
response = agent_executor.invoke({"input": "韩国目前总统是谁?"})
print(response)