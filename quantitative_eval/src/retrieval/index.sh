ENCODER="tct_colbert-v2-hnp-msmarco"
EMBEDDINGS="../../target/embeddings/$ENCODER-TaxCompass"

INDEX="../../target/indexes/$ENCODER-faiss-TaxCompass"
python -m pyserini.index.faiss \
  --input $EMBEDDINGS \
  --output $INDEX \
  --hnsw