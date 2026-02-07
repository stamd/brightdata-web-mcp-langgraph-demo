import asyncio
import os
from typing import Literal

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState


# ------------------------------------------------------------
# Agent rules: what the agent should do and how it should use tools
# ------------------------------------------------------------

SYSTEM_PROMPT = """You are a web research assistant.

Task:
- Research the user's topic using Google search results and a few sources.
- Return 6–10 simple bullet points.
- Add a short "Sources:" list with only the URLs you used.

How to use tools:
- First call the search tool to get Google results.
- Select 3–5 reputable results and scrape them.
- If scraping fails, try a different result.

Constraints:
- Use at most 5 sources.
- Prefer official docs or primary sources.
- Keep it quick: no deep crawling.
"""


# ------------------------------------------------------------
# Routing logic: decide whether to keep looping or stop
# ------------------------------------------------------------

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """
    If the last LLM message requested tool calls,
    continue to the tool execution node.
    Otherwise, end the graph and return the final answer.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


# ------------------------------------------------------------
# Node 1: Ask the LLM what to do next
# ------------------------------------------------------------

def make_llm_call_node(llm_with_tools):
    async def llm_call(state: MessagesState):
        """
        Sends the conversation (plus system rules) to the LLM.
        The model can either:
        - return a final answer, or
        - request tool calls (search, scrape, etc.)
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        ai_message = await llm_with_tools.ainvoke(messages)
        return {"messages": [ai_message]}
    return llm_call


# ------------------------------------------------------------
# Node 2: Execute MCP tools requested by the LLM
# ------------------------------------------------------------

def make_tool_node(tools_by_name: dict):
    async def tool_node(state: MessagesState):
        """
        Executes each tool call requested by the LLM and
        returns the results as ToolMessage objects.
        """
        last_ai_msg = state["messages"][-1]
        tool_results = []

        for tool_call in last_ai_msg.tool_calls:
            tool = tools_by_name.get(tool_call["name"])

            if not tool:
                tool_results.append(
                    ToolMessage(
                        content=f"Tool not found: {tool_call['name']}",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            # MCP tools are typically async
            observation = (
                await tool.ainvoke(tool_call["args"])
                if hasattr(tool, "ainvoke")
                else tool.invoke(tool_call["args"])
            )

            tool_results.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_results}
    return tool_node


# ------------------------------------------------------------
# Main: wire everything together and run the agent
# ------------------------------------------------------------

async def main():
    # Load env variables
    load_dotenv()

    # Load Bright Data token
    bd_token = os.getenv("BRIGHTDATA_TOKEN")
    if not bd_token:
        raise ValueError("Missing BRIGHTDATA_TOKEN")

    # Connect to Bright Data Web MCP and load available tools
    client = MultiServerMCPClient({
        "bright_data": {
            "url": f"https://mcp.brightdata.com/mcp?token={bd_token}",
            "transport": "streamable_http",
        }
    })

    tools = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    # Create an LLM and allow it to call MCP tools
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Build the LangGraph agent
    graph = StateGraph(MessagesState)

    graph.add_node("llm_call", make_llm_call_node(llm_with_tools))
    graph.add_node("tool_node", make_tool_node(tools_by_name))

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")

    agent = graph.compile()

    # Run the agent with a research topic
    topic = "What is Model Context Protocol (MCP) and how is it used with LangGraph?"

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Research this topic:\n{topic}")]}
        , config={"recursion_limit": 12}
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())