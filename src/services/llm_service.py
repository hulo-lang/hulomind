"""LLM service for RAG Q&A."""

import logging
from typing import Dict, Any
from abc import ABC, abstractmethod

from ..utils.logger import info, success, error

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, context: str, query: str) -> str:
        """Generate response based on context and query."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class OpenAIService(LLMProvider):
    """OpenAI API service."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        
    def generate(self, prompt: str, context: str, query: str) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            full_prompt = f"""You are a helpful assistant for the Hulo programming language. 
Based on the following documentation context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error(f"OpenAI API error: {e}")
            return f"Error generating response: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "OpenAI",
            "model": self.model,
            "type": "cloud"
        }


class QwenService(LLMProvider):
    """Qwen API service."""
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
        self.api_key = api_key
        self.model = model
        
    def generate(self, prompt: str, context: str, query: str) -> str:
        """Generate response using Qwen API."""
        try:
            import dashscope
            
            full_prompt = f"""You are a helpful assistant for the Hulo programming language. 
Based on the following documentation context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
            
            response = dashscope.Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": full_prompt}
                ],
                result_format='message'
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"API Error: {response.message}"
                
        except Exception as e:
            error(f"Qwen API error: {e}")
            return f"Error generating response: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Qwen",
            "model": self.model,
            "type": "cloud"
        }


class LocalLLMService(LLMProvider):
    """Local LLM service using Ollama."""
    
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        
    def generate(self, prompt: str, context: str, query: str) -> str:
        """Generate response using local Ollama."""
        try:
            import requests
            
            full_prompt = f"""You are a helpful assistant for the Hulo programming language. 
Based on the following documentation context, answer the user's question.

Context:
{context}

Question: {query}

Answer:"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Ollama API Error: {response.text}"
                
        except Exception as e:
            error(f"Local LLM error: {e}")
            return f"Error generating response: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Local (Ollama)",
            "model": self.model,
            "type": "local"
        }


class LLMService:
    """Main LLM service for RAG Q&A."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        
    def answer_question(self, context: str, query: str) -> str:
        """Generate answer based on context and query."""
        info(f"Generating answer for: {query}")
        
        system_prompt = """You are an expert assistant for the Hulo programming language. 
Your task is to answer questions based on the provided documentation context.

Guidelines:
1. Answer based only on the provided context
2. If the context doesn't contain enough information, say so
3. Provide clear, concise explanations
4. Include code examples when relevant
5. Be helpful and accurate

Always respond in the same language as the user's question."""
        
        answer = self.provider.generate(system_prompt, context, query)
        
        success("Answer generated successfully")
        return answer
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return self.provider.get_model_info() 