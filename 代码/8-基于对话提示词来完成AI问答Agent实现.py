# 实例化大模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

# 加载工具
from langchain_community.agent_toolkits.load_tools import load_tools
tools = load_tools(['wikipedia'])

# 直接拉取别人写好的提示词
from langchain import hub
prompt = hub.pull("sayhi12345/react-chat")

# 实例化Agent
from langchain.agents import create_react_agent, AgentExecutor
agent = create_react_agent(llm, tools, prompt)
# 执行Agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []
from langchain_core.messages import AIMessage, HumanMessage
while True:
    query = input("请输入您的问题：")
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    chat_history.append(HumanMessage(query))
    chat_history.append(AIMessage(response['output']))
    print(response)