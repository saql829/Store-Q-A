from langchain_helper import get_few_Shot_db_chain

import streamlit as st
st.title("T-Shirt Retail Q&A Tool")
question = st.text_input("Ask a question about t-shirts:")

if question:
    chain = get_few_Shot_db_chain()
    response = chain.run(question)
    st.header("Response")
    st.write("Response:", response)