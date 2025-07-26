# import google.generativeai as genai
from urllib.parse import quote_plus
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from few_shots import few_shots
import os
from dotenv import load_dotenv
load_dotenv()
def get_few_Shot_db_chain():
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        google_api_key=os.environ["api_key"],
        temperature=0.2,
    )
    db_user = "root"
    raw_password = "S@qlain1234"
    db_password = quote_plus(raw_password)  
    db_host = "localhost"
    db_name = "atliq_tshirts"
 
    print(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
 
    db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
    sample_rows_in_table_info=3)
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   
    to_vectorize=[" ".join([str(value) for value in example.values()]) for example in few_shots]

    vectorstore=Chroma.from_texts(texts=to_vectorize, embedding=embeddings,metadatas=few_shots, persist_directory="chroma_db")
    example_selector=SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2
    )
    exmaple_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
    template="\nQuestion: {Question}\nSQL Query: {SQLQuery}\nSQL Result: {SQLResult}\nAnswer: {Answer}\n"
    )
    few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=exmaple_prompt,
    prefix="You are a helpful assistant that answers questions about t-shirts.",
    suffix=PROMPT_SUFFIX,
    input_variables=["input","table_info","top_k"]
    )
    chain=SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,
        prompt=few_shot_prompt)
    return chain
if __name__=="__main__":
   chain=get_few_Shot_db_chain()
   response=chain.run("If i sell all levi t-shirts in stock, how much revenue?")
   print(response)          
                       
                       