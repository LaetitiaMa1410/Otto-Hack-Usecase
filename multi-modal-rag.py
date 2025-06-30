from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
 
from dotenv import load_dotenv
from openai import AzureOpenAI
import os
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Azure AI Search config
load_dotenv()
azure_oai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_oai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
azure_search_index = os.getenv("AZURE_SEARCH_INDEX")
 
client = AzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version="2024-08-01-preview")
 
        # Get the prompt
 
# Step 1: Retrieve relevant documents from Azure Search
def retrieve_documents(query, top_k=5):
    search_client = SearchClient(endpoint=azure_search_endpoint,
                                 index_name=azure_search_index,
                                 credential=AzureKeyCredential(azure_search_key))
    results = search_client.search(query, top=top_k)

    res_str = ""
    for doc in results:
        res_str += doc['chunk']

    return res_str
 
# Step 2: Generate answer using Azure OpenAI
def generate_answer(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    prompt = f"""You are a helpful data agent that tries to answer questions based on a datasource.
 
    Context:
    {context}
 
    Question: {query}
    Answer:"""
   
    response = client.chat.completions.create(
            model=azure_oai_deployment,
            messages=[
                {"role": "system", "content": "You are a helpful data agent that tries to answer questions based on a datasource."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
        )
    return response.choices[0].message.content
 

if __name__ == '__main__':
    # Example usage
    query = "what are the must haves?"
    docs = retrieve_documents(query)
    answer = generate_answer(query, docs)
    print("Answer:", answer)
 