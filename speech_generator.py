import os
import io # NEW: For handling audio in-memory
from gtts import gTTS # NEW: Google Text-to-Speech

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)

topic_prompt= PromptTemplate(
    input_variables=["topic"],
    template ="""You are a professional blogger.
    Create an outline for a blog post on the following topic: {topic}
    The outline should include:
        - Introduction
        - 3 main points with subpoints
        - Conclusion: {topic}
    """
    )

blog_prompt= PromptTemplate(
    input_variables=["outline"],
    template ="""You are a professional blogger.
    Write an engaging introduction paragraph based on the following
    outline:{outline}
    The introduction should hook the reader and provide a brief
    overview of the topic.
    """
    )

first_chain = topic_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = blog_prompt | llm

#second_chain = speech_prompt | llm | StrOutputParser() # Without StrOutputParse,  response is an AIMessage object (and not a raw string), you extract the actual speech.
final_chain = first_chain | second_chain

st.title("Blog Post Generator")
topic = st.text_input("Enter a topic ")

if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content) # No need for .content anymore! as we added StrOutputParser against second chain


