import streamlit as st
from langchain_ollama import ChatOllama
import pandas as pd
from pyserini.search.faiss import FaissSearcher
from langchain_core.output_parsers import StrOutputParser


#constants
MODEL_NAME = "llama3.2:latest"
DATA_DIR = "../../data"

COLLECTION = DATA_DIR + "/collection.csv"


# Dense Retrieval
INDEX = "../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss-TaxCompass"
QUERY_ENCODER = 'castorini/tct_colbert-v2-hnp-msmarco'




searcher = FaissSearcher(
    INDEX,
    QUERY_ENCODER
)



def get_context_passages(query, top_K=3, num_hits=10):
    hits = searcher.search(query, num_hits)
    collection_df = pd.read_csv(COLLECTION, encoding="latin1")
    context_passages = []
    for h in hits[:top_K]:
        docid = h.docid              # Passage reference (eg PO1)
        score = h.score              # FAISS similarity score
        text = collection_df.loc[collection_df['passage_id']==docid, 'passage'].iloc[0]
        context_passages.append({"id": docid, "score": score, "text": text})
    return context_passages


def generate_answer(query, context, llm):
    # context is a list of dicts with id/score/text
    bullet_context = "\n".join(
        [f"Document {i} [{c['id']}]: {c['text']}" for i, c in enumerate(context, 1)]
    )
    prompt = f"""Answer **only** using the provided documents, and **cite** passage IDs in square brackets (e.g., [P01]) next to each claim you use, but dont cite the document only the passage id. If the documents are insufficient,  then answer with ONLY "I'm sorry, but I don't have enough information to answer your question."
    
    Question: {query}
    
    {bullet_context}
    
    Answer (with citations):
    """.strip()
    return llm.invoke(prompt).content



def main():
    # Initialize the language model
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.5,
        system = "You are a tax expert assistant called 'Tax Compass', provide accurate answers based on the provided context."
        )



    st.title("Welcome to Tax Compass Assistant")

    # User input
    query = st.text_input("Enter your question:", "")

    if query:
        with st.spinner("Answering your question..."):
            try:

                # Retrieval
                context_passages = get_context_passages(query)

                st.markdown("**Context Passages Retrieved (for testing purposes):**")
                for i, ctx in enumerate(context_passages, 1):
                    st.markdown(f"**Rank {i} — {ctx['id']} (score {ctx['score']:.4f})**")
                    st.write(ctx['text'])
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