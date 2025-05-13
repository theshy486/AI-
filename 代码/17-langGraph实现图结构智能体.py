import os
os.environ["TAVILY_API_KEY"] = "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"

from typing import Annotated
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# 定义工具
tool = TavilySearchResults(max_results=2)
tools = [tool]

# 初始化大模型
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    """
    定义一个字典类型 State（继承自 TypeDict）
    包含一个键 messages
    值是一个 list，并且列表的更新方式由 add_messages 函数定义
    add_message 将新消息追加到列表中，而不是覆盖原有列表
    """
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def stream_graph_updates(user_input: str):
    """返回大模型输出结果"""
    for event in graph.stream({"messages": [("user", user_input)]}): # stream 表示流式输出
        for value in event.values():
            # 访问最后一个消息的内容，并将其打印出来
            print("Assistant:", value["messages"][-1].content)
            print("Test:", value)


if __name__ == '__main__':
    # 1. 创建一个 StateGraph 对象
    graph_builder = StateGraph(State)

    # 2. 添加 node
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)
    # 2.1 添加条件边
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # 2.2 添加普通边
    # tools 节点调用结束，会又回到 chatbot 节点
    graph_builder.add_edge("tools", "chatbot")

    # 3. 定义 StateGraph 的入口
    # graph_builder.add_edge(START, "chatbot")
    graph_builder.set_entry_point("chatbot")

    # 4. 定义 StateGraph 的出口
    # graph_builder.add_edge("chatbot", END)
    # graph_builder.set_finish_point("chatbot")

    # 5. 创建一个 CompiledGraph，以便后续调用
    graph = graph_builder.compile()

    # 6. 可视化 graph
    try:
        # graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        graph.get_graph().draw_mermaid_png(output_file_path="graph_with_tools.png")
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    # 7. 运行 graph
    # 通过输入"quit", "exit", "q"结束对话
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
            # 如果在 try 块中的代码执行时发生任何异常，将执行 except 块中的代码
        except:
            # 在异常情况下，这行代码将 user_input 变量设置为一个特定的问题
            user_input = "error?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
