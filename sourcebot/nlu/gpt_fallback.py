from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)

def gpt_parse(text):
    return llm.predict(f"Extract product,max_price,location,certification as JSON: {text}")
