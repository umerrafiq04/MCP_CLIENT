import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
from langchain_core.messages import ToolMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# create client with dummy API key
llm = ChatOpenAI(model="gpt-4o-mini")
servers = {
    "expense": {
        "transport": "stdio",

        "command": r"C:\Users\user\Desktop\CLIENTS\.venv\Scripts\fastmcp.exe",

        "args": [
            "run",
            r"C:\Users\user\Desktop\Servers\main.py",
            "--no-banner",
        ],
    }
}

async def main():
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    named_tools={tool.name:tool for tool in tools}
    llm =ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools=llm.bind_tools(tools)
    prompt="add the following expenses:physics subscription of rupees 200 last sunday"
    response=await llm_with_tools.ainvoke(prompt)
    print(f"tool called: {response.tool_calls}")
    if not getattr(response,"tool_calls",None):
        print(f"LLM REPLY: {response.content}")
        return
    tool_messages=[]
    for tc in response.tool_calls:
        tool_name=tc["name"]
        tool_args=tc.get("args") or {}
        tool_call_id=tc["id"]
        tool_result = await named_tools[tool_name].ainvoke(tool_args)
        tool_messages.append(
            ToolMessage(
                content=json.dumps(tool_result),
                tool_call_id=tool_call_id
            )
        )
    final_response = await llm_with_tools.ainvoke(
        [prompt, response] + tool_messages
    )
    print("Final response:", final_response.content)

    print("\n✅ MCP CONNECTED. TOOLS FOUND:")
    for t in tools:
        print("-", t.name)

if __name__ == "__main__":
    asyncio.run(main())
