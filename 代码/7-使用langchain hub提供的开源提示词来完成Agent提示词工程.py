import os  # 引入os模块
os.environ['SERPAPI_API_KEY'] = 'e14b42e676f3e2d06a9403d943f8bed11e4f45a05ad49c0ff11a03289629e7b1'

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

# 加载所需工具
from langchain_community.agent_toolkits.load_tools import load_tools

# python
from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
python_repl_ast = Tool(
    name="python_repl_ast",
    description="一个Python shell。使用它来执行python命令。输入应该是一个有效的python命令。如果你想看到一个值的输出，你应该用`print(...)`打印出来。",
    func=PythonAstREPLTool().run,
)

tools = load_tools(["serpapi"], allow_dangerous_tools=True)
tools += [python_repl_ast]

# 提示词
from langchain import hub
prompt = hub.pull("zqh/hwchase17_react_cn")

from langchain.agents import create_react_agent, AgentExecutor
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
response = agent_executor.invoke({"input": "基于FastAPI生成一个学生信息管理项目的目录结构，其中run.py完成FastAPI应用初始化，view.py提供RestAPI接口，这个过程基于python代码生成并调用python解释器来执行代码过程。"})
print(response)
