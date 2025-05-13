from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool

# 创建Python解释器
python_repl_tool = PythonAstREPLTool()

# python代码
query_value = """
def add(a,b):
    return a+b
print(add(10,20))
"""

# 基本使用
repl_tool = Tool(
    name="python_repl_ast",
    description="一个Python shell。使用它来执行python命令。输入应该是一个有效的python命令。如果你想看到一个值的输出，你应该用`print(...)`打印出来。",
    func=python_repl_tool.run,
)
response = repl_tool.invoke(query_value)
print(response)