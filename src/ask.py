from openai import OpenAI

def ask_with_file_search(client: OpenAI, model: str, vector_store_id: str, question: str, include_search_results: bool = False):
    kwargs = {}
    # Cookbook shows include=["output[*].file_search_call.search_results"] for deeper inspection of retrieved chunks.
    if include_search_results:
        kwargs["include"] = ["output[*].file_search_call.search_results"]

    resp = client.responses.create(
        model=model,
        input=question,
        tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
        **kwargs,
    )
    return resp