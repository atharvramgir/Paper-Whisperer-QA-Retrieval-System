from openai import OpenAI
import os
from typing import List, Dict

class RAGPipeline:
    def __init__(self):
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        self.model = "gpt-4o"
        
    def generate_response(self, query: str, contexts: List[str]) -> Dict[str, str]:
        """Generate a response using the retrieved contexts."""
        prompt = f"""Based on the following contexts from the 'Attention Is All You Need' paper, 
        answer the question: {query}\n\nContexts:\n{' '.join(contexts)}\n\n
        Please provide a response that directly answers the question and includes citations 
        to specific parts of the paper. Format your response as a JSON object with 'answer' 
        and 'citations' fields."""
        
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content