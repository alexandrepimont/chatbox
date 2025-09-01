from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
Geologist Assistant for exploration in pre-salt Brazil

Here are some relevant reviews: {reviews}

Here is the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Enter your question ('q' to quit): ")
    if question == 'q':
        break
    result = chain.invoke({
        "reviews": [],
        "question": question
    })


    print(result)