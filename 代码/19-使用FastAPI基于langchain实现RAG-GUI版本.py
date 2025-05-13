import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from typing import Any
import queue
import threading
import requests
from tavily import TavilyClient
from datetime import datetime
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义工具
@tool("tavily_search")
def tavily_search(query: str) -> dict:
    """使用Tavily搜索引擎搜索信息.
    query: 搜索查询词
    """
    client = TavilyClient(api_key="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")
    return client.search(query, max_results=3)


@tool("mairui")
def mairui_api(code: str) -> dict:
    """查询股票实时行情.
    code: 股票代码(如000001)
    字段说明：
    fm: 五分钟涨跌幅（%）
    h: 最高价（元）
    hs: 换手（%）
    lb: 量比（%）
    l: 最低价（元）
    lt: 流通市值（元）
    o: 开盘价（元）
    pe: 市盈率
    pc: 涨跌幅（%）
    p: 当前价格（元）
    sz: 总市值（元）
    cje: 成交额（元）
    ud: 涨跌额（元）
    v: 成交量（手）
    yc: 昨日收盘价（元）
    zf: 振幅（%）
    zs: 涨速（%）
    """
    api_key = "3F380102-3DFE-4F89-A749-6E000A02A2EA"
    url = f"http://api.mairui.club/hsrl/ssjy/{code}/{api_key}"
    try:
        response = requests.get(url)
        result = response.json()

        if not isinstance(result, dict):
            return {"error": "API返回格式错误"}

        if result.get('msg') == 'ok' and result.get('data'):
            data = result['data']
            return {
                "股票名称": data.get('name', '未知'),
                "当前价格": f"{data.get('p', 0)}元",
                "涨跌幅": f"{data.get('pc', 0)}%",
                "涨跌额": f"{data.get('ud', 0)}元",
                "成交量": f"{data.get('v', 0)}手",
                "成交额": f"{data.get('cje', 0)}元",
                "振幅": f"{data.get('zf', 0)}%",
                "最高": f"{data.get('h', 0)}元",
                "最低": f"{data.get('l', 0)}元",
                "今开": f"{data.get('o', 0)}元",
                "昨收": f"{data.get('yc', 0)}元",
                "量比": f"{data.get('lb', 0)}",
                "换手率": f"{data.get('hs', 0)}%",
                "市盈率": data.get('pe', 0),
                "总市值": f"{data.get('sz', 0)}元",
                "流通市值": f"{data.get('lt', 0)}元",
                "涨速": f"{data.get('zs', 0)}%",
                "60日涨跌幅": f"{data.get('zdf60', 0)}%",
                "年初至今涨跌幅": f"{data.get('zdfnc', 0)}%",
                "更新时间": data.get('t', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            }
        return {"error": "获取数据失败，API返回错误"}

    except requests.RequestException as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"数据解析失败: {str(e)}"}
    except Exception as e:
        return {"error": f"未知错误: {str(e)}"}


class QueueCallback(BaseCallbackHandler):
    """自定义回调处理器"""

    def __init__(self, q: queue.Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)


def create_agent():
    """创建Agent"""
    q = queue.Queue()
    callback = QueueCallback(q)

    # from vllm import LLM
    # llm = LLM(model="qwen2.5:7b")
    llm = ChatOllama(
        model="qwen2.5:7b",
        callback_manager=CallbackManager([callback])
    )

    tools = [mairui_api, tavily_search]

    prompt = PromptTemplate.from_template(
        """尽可能简约和准确地使用中文回应如下问题。您可以使用以下工具:
{tools}

使用以下格式：
Question: 您必须回答的输入问题
Thought: 你应该始终思考该做什么
Action: 要采取的操作名称（可用工具：{tool_names}）
Action Input: 要传递给工具的参数
Observation: 工具返回的结果
... (这个思考/行动/观察可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 原始输入问题的最终答案

对于股票查询，你可以：
1. 使用mairui工具获取股票的实时行情数据
2. 使用tavily_search工具搜索相关新闻和分析

注意事项：
- 股票代码必须是6位数字，不要加任何引号或其他字符
- A股代码：上海主板以6开头，深圳主板以000开头，创业板以300开头，科创板以688开头
- 如果遇到股票名称查询，请先使用tavily_search搜索股票代码

示例格式：
Action: mairui
Action Input: 600519

开始!
Question: {input}
Thought: {agent_scratchpad}

提醒！务必使用中文回答，并对数据进行合理的解读和总结。
""")

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3  # 限制最大迭代次数
    )

    return agent_executor, q


def bot_response(history):
    if not history:
        return history
    last_user_message = history[-1][0]

    try:
        agent_executor, q = create_agent()

        def run_agent():
            try:
                response = agent_executor.invoke({
                    "input": last_user_message,
                    "handle_parsing_errors": True  # 添加错误处理
                })
                if response and "output" in response:
                    # 处理可能的错误消息
                    output = response["output"]
                    if "PARSING_ERROR" in output:
                        q.put("抱歉，我理解有误。请使用更清晰的方式描述您的问题。")
                    else:
                        # 按字符分割输出
                        for char in output:
                            q.put(char)
                else:
                    q.put("无法获取有效响应")
            except Exception as e:
                print(f"Agent error: {e}")
                q.put("处理请求时发生错误，请稍后重试。")
            finally:
                q.put(None)  # 结束标记

        # 启动代理线程
        thread = threading.Thread(target=run_agent)
        thread.daemon = True
        thread.start()

        # 收集响应
        response_text = ""
        while True:
            try:
                token = q.get(timeout=30.0)
                if token is None:
                    break
                response_text += token
                history[-1][1] = response_text
                yield history
            except queue.Empty:
                print("Response timeout")
                break
            except Exception as e:
                print(f"Stream error: {e}")
                break

        if not response_text:
            history[-1][1] = "抱歉，我现在无法回答。请稍后再试。"
            yield history

    except Exception as e:
        print(f"Bot response error: {e}")
        history[-1][1] = "处理消息时发生错误，请稍后重试。"
        yield history


# 自定义CSS样式
custom_css = """
    .contain { 
        display: flex; 
        flex-direction: column;
        height: 90vh;
        min-height: 500px;
        margin: 2vh auto;
        width: 90%;
        max-width: 1200px;
    }
    .gradio-container { 
        height: 100vh;
        background-color: #f7f7f8;
        margin: auto;
        padding: 0;
    }
    #component-0 { 
        height: 100%;
        background: #ffffff;
        border-radius: 1rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
    }
    #chatbot { 
        flex: 1 1 auto;
        overflow: auto;
        background-color: #ffffff;
        border-radius: 1rem 1rem 0 0;
        min-height: 0;
    }
    #chatbot .message.user {
        background-color: #ffffff;
        padding: 1.5rem clamp(1rem, 4%, 2rem);
        border-bottom: 1px solid #e5e7eb;
    }
    #chatbot .message.bot {
        background-color: #f7f7f8;
        padding: 1.5rem clamp(1rem, 4%, 2rem);
        border-bottom: 1px solid #e5e7eb;
    }
    #chatbot .message {
        color: #374151;
        font-size: clamp(14px, 1.5vw, 16px);
        line-height: 1.6;
    }
    #chatbot .message-wrap {
        width: 100% !important;
        max-width: min(800px, 90%) !important;
        margin: auto;
    }
    #input-box { 
        min-height: 60px;
        height: auto;
        max-height: 200px;
        margin: 0.75rem 1rem;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        color: #374151;
        font-size: clamp(14px, 1.5vw, 16px);
        line-height: 1.5;
        resize: vertical;
    }
    #input-box::placeholder {
        color: #9ca3af;
    }
    .input-row {
        margin: 0;
        padding: 0.5rem clamp(0.5rem, 2%, 1rem);
        background-color: #ffffff;
        border-top: 1px solid #e5e7eb;
        border-radius: 0 0 1rem 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .submit-btn, .clear-btn {
        font-size: clamp(12px, 1.2vw, 14px) !important;
        padding: 8px clamp(12px, 2vw, 16px) !important;
        height: auto !important;
        flex-shrink: 0;
    }
    .submit-btn {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 0.5rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .submit-btn:hover {
        background-color: #1d4ed8 !important;
        transform: translateY(-1px);
    }
    .clear-btn {
        background-color: #ffffff !important;
        color: #374151 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.5rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .clear-btn:hover {
        background-color: #f3f4f6 !important;
        transform: translateY(-1px);
    }
    footer {display: none !important;}
    .message-wrap img {
        width: clamp(24px, 3vw, 32px) !important;
        height: clamp(24px, 3vw, 32px) !important;
        margin-right: clamp(0.5rem, 2%, 1rem);
        border-radius: 0.375rem;
    }
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
    @media (max-width: 768px) {
        .contain {
            height: 95vh;
            width: 95%;
            margin: 2.5vh auto;
        }
        #input-box {
            margin: 0.5rem;
        }
    }
"""

# Gradio界面配置
with gr.Blocks(css=custom_css) as interface:
    with gr.Column(elem_classes="contain"):
        chatbot = gr.Chatbot(
            value=[],
            elem_id="chatbot",
            bubble_full_width=True,
            avatar_images=(
                "https://api.dicebear.com/7.x/bottts/svg?seed=user",
                "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"
            ),
            height=600,
            show_label=False,
        )
        with gr.Row():
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="请输入股票相关问题...",
                container=False,
                elem_id="input-box"
            )
            submit_btn = gr.Button("发送", scale=1, variant="primary")

        clear = gr.Button("清空对话")


    def user_submit(message, history):
        if message.strip() == "":
            return "", history
        return "", history + [[message, None]]


    # 更新事件处理
    submit_btn.click(
        user_submit,
        [txt, chatbot],
        [txt, chatbot],
        queue=False
    ).then(
        bot_response,
        chatbot,
        chatbot
    )

    txt.submit(
        user_submit,
        [txt, chatbot],
        [txt, chatbot],
        queue=False
    ).then(
        bot_response,
        chatbot,
        chatbot
    )

    clear.click(lambda: [], None, chatbot, queue=False)

app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)