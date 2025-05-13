import os
os.environ["TAVILY_API_KEY"] = "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"

# 实例化大模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

# 加载Agent工具
from langchain_community.tools.tavily_search import TavilySearchResults
tools = [TavilySearchResults(max_results=1)]


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_structured_chat_agent, AgentExecutor

def structured_chat(input):
    ## 提示词
    system = '''你需要尽可能地帮助和准确地回答人类的问题。你可以使用以下工具:

    {tools}

    使用json blob通过提供action key（工具名称）和action_input key（工具输入）来指定工具。
    "action"的有效取值为: "Final Answer" or {tool_names}

    每个$JSON_BLOB只提供一个action，如下所示：
    ```
    {{
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }}
    ```

    遵循此格式:

    Question: 用户输入的问题
    Thought:  回答这个问题我需要做些什么，尽可能考虑前面和后面的步骤
    Action:   回答问题所选取的工具
    ```
    $JSON_BLOB
    ```
    Observation: 工具返回的结果
    ... (这个思考/行动/行动输入/观察可以重复N次)
    Thought: 我现在知道最终答案
    Action: 工具返回的结果信息
    ```
    {{
      "action": "Final Answer",
      "action_input": "原始输入问题的最终答案"
    }}
    ```
    开始！提醒始终使用单个操作的有效json blob进行响应。必要时使用工具. 如果合适，直接回应。格式是Action：“$JSON_BLOB”然后是Observation'''

    human = '''{input}

    {agent_scratchpad}

    (提醒:无论如何都要在JSON blob中响应!)'''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
        ]
    )

    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    print(agent)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    inputs = {"input": input}
    response = agent_executor.invoke(inputs)
    return response.get("output")


response = structured_chat("2024年中国最流行的电影？")
print(response)
