from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_ollama.llms import OllamaLLM
import requests
import streamlit as st

model = OllamaLLM(model="llama3.1:8b")

st.set_page_config(
    page_title="News",
    page_icon=":newspaper:",
    layout="centered"
)

st.title(":newspaper: News")
categories = ["Technology","Sports","Entertainment","Weather","politics"]
topic = st.text_input("Enter topic")

url = f"https://newsapi.org/v2/everything?q={topic}&apiKey=7c7dc30d36c2426f93a98c3260b75bb8"

if st.button("Get news and Summarize it"):
    response = requests.get(url)
    data = response.json()
    articles = data.get("articles", [])

    if articles:
        for article in articles[:5]:
            title = article.get("title","Not Found")
            source_name = article.get("source","Not Found").get("name","Not Found")
            content = article.get("content","Not Found")

            template1 = """Summarize the following text: {text}"""
            first_prompt = ChatPromptTemplate.from_template(template1)
            chain_one = LLMChain(llm=model, prompt=first_prompt,output_key="summary_text")

            template2 = """Just tell me the category in two words, text: {summary_text}"""
            second_prompt = ChatPromptTemplate.from_template(template2)
            chain_two = LLMChain(llm=model, prompt=second_prompt,output_key="Category")

            overall_simple_chain = SequentialChain(chains=[chain_one, chain_two],
                                                   input_variables=["text"],
                                                   output_variables=["summary_text","Category"],
                                                         verbose=True
                                                         )
            output = overall_simple_chain(content)
            st.markdown(f"## {title}")
            st.markdown(f"**Source:** {source_name}")
            # st.markdown(f"**Summary:** {output['choices'][0]['message']['content']}")
            st.markdown(f"**Content:** {output['summary_text']}")
            st.markdown(f"**Category:** {output['Category']}")
    else:
        st.write("No articles found for the given topic.")