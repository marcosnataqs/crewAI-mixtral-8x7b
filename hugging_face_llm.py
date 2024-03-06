from langchain_community.llms import HuggingFaceHub

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

hf_llm = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.7,
        "repetition_penalty": 1.03,
    },
)
