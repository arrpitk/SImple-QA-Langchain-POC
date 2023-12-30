import streamlit as st

from langchain.llms import HuggingFaceHub  

def load_answers(question):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large")
    answer = llm(question, options={"use_gpu": False, "use_cache": False}, parameters={"return_full_text": False, "num_return_sequences": 1, "temperature": 1.0, "top_p": 0.9, "max_new_tokens": 250})
    return answer

st.set_page_config(page_title="LangChain LLM LAB", page_icon="🔗")
st.header("LangChain LLM LAB")

def get_text():
    input_text = st.text_input("Enter your question here...",key="question")
    return input_text

user_input = get_text()

response = load_answers(user_input)

submit = st.button("Generate Answer")

if submit:
    st.subheader("Answer:")
    st.write(response)


