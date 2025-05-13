import os
os.environ["TAVILY_API_KEY"] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 固定为'true'
os.environ['LANGCHAIN_API_KEY'] = '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'

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)]

from langchain import hub
prompt = hub.pull("hwchase17/react-chat")

# 实例化Agent
from langchain.agents import create_react_agent, AgentExecutor
agent = create_react_agent(llm, tools, prompt)
# 创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 创建对话历史管理对象
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    query = input("问题：")
    if query == "exit": break
    response = agent_with_chat_history.invoke(
        {"input": query},
        config={"configurable": {"session_id": 'session-10086'}},
    )
    print(response)