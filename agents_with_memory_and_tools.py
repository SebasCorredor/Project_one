
from typing import Annotated
from typing import Literal
from typing_extensions import TypedDict
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()




@tool
def sumar(a: int, b: int) -> int:
    """Suma dos enteros.
    Args:
        a: Primer entero
        b: Segundo entero
    """
    return a + b

@tool
def multiplicar(a: int, b: int) -> int:
    """multiplica dos enteros.
    Args:
        a: Primer entero
        b: Segundo entero
    """
    return a * b


members = ["sumador", "multiplicador"]


options = members + ["FINISH"]

system_prompt = (
    f'''
    Eres un supervisor encargado de gestionar una conversación entre los
    siguientes trabajadores: {members}. Dada la siguiente solicitud del usuario,
    responde con el trabajador que debe actuar a continuación. Cada trabajador realizará una
    tarea y responderá con sus resultados y estado.
     
    Las funciones de cada trabajador son las siguientes : 

    *sumador: Realiza sumas lo usas solo para operaciones de sumas
    *multiplicador : Realiza multiplicaciones lo usas solo para operaciones de multiplicaciones

    cuando un {members} termine te dira `RESPUESTA FINAL` y asi `deberas` ir vas a FINISH

    '''
)

class Router(TypedDict):
    next: Literal["sumador", "multiplicador","FINISH"]


llm = ChatVertexAI(model="gemini-2.0-flash-001",temperature=0)

class State(MessagesState):
    next: str

def supervisor_node(state: State) -> Command[Literal["sumador", "multiplicador","__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

sum_agent = create_react_agent(
    llm, tools=[sumar], prompt="Eres un agente de sumas nada mas, cuando termines dices RESPUESTA FINAL"
)

def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = sum_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="sumador")
            ]
        },
        goto="supervisor",
    )

multi_agent = create_react_agent(llm, tools=[multiplicar],prompt="Eres un agente de multiplicaciones nada mas, cuando termines dices RESPUESTA FINAL")

def code_node(state: State) -> Command[Literal["supervisor"]]:
    result = multi_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="multiplicador")
            ]
        },
        goto="supervisor",
    )

builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("sumador", research_node)
builder.add_node("multiplicador", code_node)
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "2"}}


for s in graph.stream(
    {"messages": [("user", "Cuanto es dos mas dos")]},config,subgraphs=True,
):
    print(s)
    print("----")

for s in graph.stream(
    {"messages": [("user", "Al resultado anterior multiplicale 2")]},config,subgraphs=True
):
    print(s)
    print("----")