from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

from langchain_community.agent_toolkits.load_tools import load_tools
tools = load_tools(['wikipedia'])

# 从 LangChain Hub 中拉取预训练的 ReAct 提示词模板：https://smith.langchain.com/hub
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
    user = input("用户名：")
    if user == "exit": break
    while True:
        query = input("问题：")
        if query == "exit": break
        response = agent_with_chat_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": user}},
        )
        print(response)