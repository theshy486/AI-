import requests, json
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4', temperature=0)

from langchain_core.tools import tool
from tavily import TavilyClient

# 自定义tool
@tool("tavily")
def tavily_search(query: str) -> dict:
    """tavily search."""
    client = TavilyClient(api_key="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")
    return client.search(query, max_results=2)

@tool("mairui")
def mairui_api(dm: str) -> dict:
    """mairui api.
    dm: 股票代码(如000001)
    字段名称	数据类型	字段说明
    fm	number	五分钟涨跌幅（%）
    h	number	最高价（元）
    hs	number	换手（%）
    lb	number	量比（%）
    l	number	最低价（元）
    lt	number	流通市值（元）
    o	number	开盘价（元）
    pe	number	市盈率（动态，总市值除以预估全年净利润，例如当前公布一季度净利润1000万，则预估全年净利润4000万）
    pc	number	涨跌幅（%）
    p	number	当前价格（元）
    sz	number	总市值（元）
    cje	number	成交额（元）
    ud	number	涨跌额（元）
    v	number	成交量（手）
    yc	number	昨日收盘价（元）
    zf	number	振幅（%）
    zs	number	涨速（%）
    sjl	number	市净率
    zdf60	number	60日涨跌幅（%）
    zdfnc	number	年初至今涨跌幅（%）
    t	string	更新时间YYYY-MM-DD HH:MM
    """
    api_key = "3F380102-3DFE-4F89-A749-6E000A02A2EA"
    url = f"http://api.mairui.club/hsrl/ssjy/{dm}/{api_key}"
    result = requests.get(url).json()
    return result


tools = [mairui_api, tavily_search]


# 提示词
from langchain.prompts import PromptTemplate
prompt = PromptTemplate.from_template(
"""尽可能简约和准确地使用中文回应如下问题。您可以使用以下工具:
{tools}
使用以下格式：
Question：您必须回答的输入问题
Thought：你应该始终思考该做什么
Action：要采取的操作，应该是〔｛tool_names｝〕之一
Action Input：动作的输入
Observation：行动的结果
…（这个 Thought/Action/Action Input/Observation 可以重复N次）
Thought：我现在知道最终答案了
Final Answer：原始输入问题的最终答案
开始!
Question: {input}
Thought:{agent_scratchpad}
提醒！务必使用中文回答。
""")

# 自定义tool工具必须使用initialize_agent来创建智能体，不能使用新版本create_xxx_agent，会报错
from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, agent_kwargs={"prompt": prompt})

response = agent.invoke({"input": "良品铺子的股票行情？"})
print(response)