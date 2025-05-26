import dotenv
dotenv.load_dotenv()

from langchain_core import embeddings
import pandas as pd
from typing_extensions import TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS

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

embeddings =GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 10})

class State(TypedDict):
    query: str
    result: list

def search_query(state: State):
    result = retriever.invoke(state["query"])
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
while True:
    user_input = input("Type your question below (or 'exit' to quit):\n")
    if user_input.lower() == "exit":
        break
    state = {"query": user_input, "result": []}
    graph.invoke(state, config)
