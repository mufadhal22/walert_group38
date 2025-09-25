import streamlit as st
from langchain_ollama import ChatOllama
import pandas as pd
from pyserini.search import FaissSearcher
from langchain_core.output_parsers import StrOutputParser


#constants
MODEL_NAME = "llama3.2:latest"
DATA_DIR = "../../data"

COLLECTION = DATA_DIR + "/collection.csv"


# Dense Retrieval
INDEX = "../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = 'facebook/dpr-question_encoder-multiset-base'




searcher = FaissSearcher(
    INDEX,
    QUERY_ENCODER
)



def get_context_passages(query):
    num_hits = 10
    hits = searcher.search(query, num_hits)
    top_K = 3
    collection_df = pd.read_csv(COLLECTION)
    context_passages = []
    for d in hits[:top_K]:
        temp_passage = list(collection_df[collection_df['passage_id'] == d.docid]['passage'])[0]
        context_passages.append(temp_passage)
    return context_passages


def generate_answer(query, context, llm):

    prompt = f"""Answer the question \"{query}\" based ONLY on the following context:\n
    context 1: {context[0]}\n
    context 2: {context[1]}\n
    context 3: {context[2]}\n

    If the context is not related to the question or no relevant information is found, then answer with ONLY "I don't have enough information to answer that.".

    """
    
    
    llm_result = llm.invoke(prompt).content
    # llm_result = StrOutputParser().parse(llm_result)
    return llm_result



def main():
    # Initialize the language model
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.5,
        system = "You are a tax expert assistant called 'Tax Compass', provides accurate answers based on the provided context."
        )



    st.title("Welcome to Tax Compass Assistant")

    # User input
    query = st.text_input("Enter your question:", "")

    if query:
        with st.spinner("Answering your question..."):
            try:

                # Retrieval
                context_passages = get_context_passages(query)

                st.markdown("**Context Passages Retrieved:**")
                for i, passage in enumerate(context_passages, 1):
                    st.markdown(f"**Document {i}:** {passage}")
                st.write("---") 

                

                # Generate answer
                response = generate_answer(query, context_passages, llm)

                st.markdown("**According to my knowledge base:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()