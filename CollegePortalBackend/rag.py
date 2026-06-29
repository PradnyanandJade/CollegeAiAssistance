from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# llm model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# parser
parser = StrOutputParser()

# Embedding model 
embedding = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

# document loaders 
loader = DirectoryLoader(
    path = 'data',
    glob = '*.txt',
    loader_cls = TextLoader
)
documents = loader.load()

# splitting the data
splitters = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitters.split_documents(documents)

# vector store FAISS
vector_store = FAISS.from_documents(
    documents = chunks,
    embedding = embedding
)

# retrievers 
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10
    }
)
# convert into context
def documents_to_text(documents):
    return '\n\n'.join(doc.page_content for doc in documents)


# Structure Output
class CollegeAnswer(BaseModel):
    answer : str = Field(description= "Answer the user's question")
    source_found : bool  = Field(description= "Whether the answer was found in retrieved document")

# pydantic output parser 
pydantic_parser = PydanticOutputParser(pydantic_object = CollegeAnswer)

# template
prompt = PromptTemplate(
    template="""
You are a helpful AI Assistant at a College Portal.

Answer only using the provided context.

If the answer exists in the context,
summarize it clearly.

Do not omit important details.

If the answer is not present in the context:
- answer = "Information not found"
- source_found = false

Otherwise:
- answer = actual answer
- source_found = true

{format_instruction}

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
    partial_variables = {'format_instruction':pydantic_parser.get_format_instructions()}
)


# chain structure
# retriver --> context     = context   \
#                                        ----> prompt -->  model --> parser 
# question --> passthrough = question  /

parallel_chain = RunnableParallel(
    context = retriever | RunnableLambda(documents_to_text),
    question = RunnablePassthrough() 
)

main_chain = parallel_chain | prompt | model | pydantic_parser

# result = main_chain.invoke("What is the admission process ? ")

# print(result.answer)
# print(result.source_found)


def ask_question(question):

    result = main_chain.invoke(question)

    return {
        "answer": result.answer,
        "source_found": result.source_found
    }