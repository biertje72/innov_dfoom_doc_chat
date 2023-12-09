class DpPromptEngineSettings():
    # Amount of documents to return (Default: 4)
    K_VALUE = 3
    # score_threshold: Minimum relevance threshold for similarity_score_threshold    
    SCORE_THRESHOLD = 0.8
    # Flag to determine whether to use chat history or not.
    USE_HISTORY = False #MBI custom, Maurice added this
    TEMPERATURE = 0.000  #default was 0.2
