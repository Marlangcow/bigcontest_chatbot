from transformers import (
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import GOOGLE_API_KEY


def load_embedding_model():
    model_name = "jhgan/ko-sroberta-multitask"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model


def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        top_p=0.85,
        frequency_penalty=0.1,
    )


def create_chain(llm, prompt_template):
    return LLMChain(
        prompt=prompt_template,
        llm=llm,
        output_parser=StrOutputParser(),
    )
