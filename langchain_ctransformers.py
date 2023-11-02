from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate


def create_prompt():
    template = """You are a question answering Large Language Model.
    Answer the following question.
    USER: {question}
    ASSISTANT:"""

    prompt = PromptTemplate.from_template(template)
    return prompt

def create_llm(model_path='./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf', model_type="mistral"):
    llm = CTransformers(model=model_path, model_type=model_type)
    return llm

def create_chain(prompt, llm):
    chain = prompt | llm.bind(stop=["USER:"])
    return chain

if __name__ == "__main__":
    chain = create_chain(create_prompt(), create_llm())

    keep_running = True
    while keep_running:
        print("input something...")
        question = input()
        if question != "exit":
            print(chain.invoke({"text": question}))
        else:
            keep_running = False