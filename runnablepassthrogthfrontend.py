import streamlit as st



# LangChain imports
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Initialize model
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Prompts
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words:\n{code}")
])

# Chains
seq = code_prompt | model | parser

seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation": explain_prompt | model | parser
})

chain = seq | seq2

# ---------------- UI ---------------- #

st.set_page_config(page_title="AI Code Generator + Explainer", layout="wide")

st.title("💻 AI Code Generator & Explainer")

# Mode selection
mode = st.radio("Choose Mode:", ["Generate Code", "Explain My Code"])

# ---------------- Generate Code ---------------- #
if mode == "Generate Code":
    topic = st.text_input("Enter what code you want:")

    if st.button("Generate"):
        if topic:
            with st.spinner("Generating..."):
                result = chain.invoke({"topic": topic})

            st.subheader("🧠 Generated Code")
            st.code(result['code'], language="python")

            st.subheader("📖 Explanation")
            st.write(result['explanation'])
        else:
            st.warning("Please enter a topic!")

# ---------------- Explain User Code ---------------- #
elif mode == "Explain My Code":
    user_code = st.text_area("Paste your code here:")

    if st.button("Explain"):
        if user_code:
            with st.spinner("Explaining..."):
                explanation = (explain_prompt | model | parser).invoke({
                    "code": user_code
                })

            st.subheader("📖 Explanation")
            st.write(explanation)
        else:
            st.warning("Please paste some code!")
            
#python -m streamlit run runnablepassthrogthfrontend.py
