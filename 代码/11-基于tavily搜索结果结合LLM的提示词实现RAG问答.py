import os
os.environ["TAVILY_API_KEY"] = "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"


from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:0.5b")
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model='gpt-4', temperature=0)

from tavily import TavilyClient
client = TavilyClient()

query = "北京今天的天气以及温度?"
content = client.search(query, search_depth="advanced", max_results=2)["results"]

prompt = [
    {
    "role": "system",
    "content":  f'You are an chinese\'s AI critical thinker research assistant. '
                f'Your sole purpose is to write well written, critically acclaimed,'
                f'objective and structured reports on given text.'
    }, {
    "role": "user",
    "content": f'Information: """{content}"""\n\n'
               f'Using the above information, answer the following'
               f'query: "{query}" in a detailed report --'
               f'Please use markdown format and markdown syntax and chinese language.'
}]

report = llm.invoke(prompt).content

print(report)