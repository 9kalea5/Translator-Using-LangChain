from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

groq_api_key=os.environ.get("GROQ_API_KEY")


## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
       (
        "system",
        "You are a helpful assistant that translates a given language to English. Translate the user sentence.",
    ),
        ("user","Question:{question}")
    ]
)

## streamlit framework

st.title('Langchain Demo With GROQ API')
input_text=st.text_input("Input the text you want to to translate")

# openAI LLm 
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Gemma-7b-It")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
