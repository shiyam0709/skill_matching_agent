import dotenv
dotenv.load_dotenv()

import pandas as pd
from typing_extensions import TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

df = pd.read_csv("./MOCK_DATA.csv")

documents = []
for _, row in df.iterrows():
    product_text = str(row["product"])
    documents.append(
        Document(
            page_content=product_text,
            metadata=row.to_dict()
        )
    )

vector_store = InMemoryVectorStore(GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
vector_store.add_documents(documents)

class State(TypedDict):
    query: str
    result: list

def search_query(state: State):
    query = input("Type your question below:\n")
    result = vector_store.similarity_search(query)
    response_lines = []
    for doc in result:
        row = doc.metadata
        row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
        response_lines.append(row_str)
    response = "\n\n".join(response_lines)
    print(response)

graph_builder = StateGraph(State)

graph_builder.add_edge(START, "search_query")
graph_builder.add_node("search_query", search_query)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"query": "", "result": ""}, config)
