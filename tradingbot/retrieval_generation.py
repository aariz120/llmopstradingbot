from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from tradingbot.data_ingestion import ingestdata


def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your finance bot is an expert in finance related advice.
    Ensure your answers are relevant to the query context and refrain from straying off-topic.
    Your responses should be concise and informative.if you are getting any hi hello how are you message
    you just have to reply normally dont add any additional information.if you are getting any other question 
    which is not related to product try to answer that question using your own knowledge

    CONTEXT:C
    CONTEXT:C
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """


    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    llm = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("None")
    chain  = generation(vstore)
    print(chain.invoke("what is Market For Registrant’s Common Equity?"))
    