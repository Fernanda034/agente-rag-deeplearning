Se crearon 10273 fragmentos de texto.
Cargando modelo de embeddings (esto puede tardar la primera vez que se descarga)...
Traceback (most recent call last):
  File "C:\Users\PC\miniconda3\Lib\site-packages\langchain_huggingface\embeddings\huggingface.py", line 68, in __init__
    import sentence_transformers  # type: ignore[import]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'sentence_transformers'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\PC\Documents\ChatBot\agente-rag-deeplearning\01_ingesta\ingestion.py", line 62, in <module>
    main()
  File "C:\Users\PC\Documents\ChatBot\agente-rag-deeplearning\01_ingesta\ingestion.py", line 44, in main
    embeddings = HuggingFaceEmbeddings(
                 ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PC\miniconda3\Lib\site-packages\langchain_huggingface\embeddings\huggingface.py", line 74, in __init__
    raise ImportError(msg) from exc
ImportError: Could not import sentence_transformers python package. Please install it with `pip install sentence-transformers`.