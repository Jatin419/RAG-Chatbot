"""
llm.py - LLM Provider selection and instantiation for Ollama, OpenAI, and Groq.
"""

import os
from typing import Any
from config import LLM_PROVIDERS, DEFAULT_TEMPERATURE
from utils import logger


def get_llm(
    provider: str = "Ollama",
    model_name: str = "llama3.2",
    temperature: float = DEFAULT_TEMPERATURE
) -> Any:
    """
    Instantiate and return the requested LLM Chat Model based on provider settings.
    
    Args:
        provider: Provider name ("Ollama", "OpenAI", or "Groq").
        model_name: Model identifier string.
        temperature: LLM sampling temperature (0.0 to 1.0).
        
    Returns:
        LangChain ChatModel instance.
        
    Raises:
        ValueError: If provider is invalid or required API keys are missing.
    """
    logger.info(f"Instantiating LLM - Provider: {provider}, Model: {model_name}, Temp: {temperature}")

    if provider == "Ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_name,
                temperature=temperature
            )
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=model_name,
                    temperature=temperature
                )
            except ImportError as e:
                raise ValueError("`langchain-ollama` is required. Please install it using `pip install langchain-ollama`.")

    elif provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing. Please set it in `.env` file or Settings.")
        
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )
        except ImportError:
            raise ValueError("`langchain-openai` is required. Please install it using `pip install langchain-openai`.")

    elif provider == "Groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Please set it in `.env` file or Settings.")
        
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_name,
                temperature=temperature,
                groq_api_key=api_key
            )
        except ImportError:
            raise ValueError("`langchain-groq` is required. Please install it using `pip install langchain-groq`.")

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
