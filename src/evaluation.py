from typing import List
import numpy as np

class Evaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_retrieval(self, query: str, retrieved_contexts: List[str], 
                          response: str) -> dict:
        """Evaluate the quality of retrieved contexts."""
        # Calculate basic metrics
        avg_context_length = np.mean([len(ctx) for ctx in retrieved_contexts])
        query_terms = set(query.lower().split())
        
        # Calculate term overlap between query and contexts
        context_term_overlap = [
            len(query_terms.intersection(set(ctx.lower().split()))) / len(query_terms)
            for ctx in retrieved_contexts
        ]
        
        avg_term_overlap = np.mean(context_term_overlap)
        
        metrics = {
            "avg_context_length": avg_context_length,
            "avg_term_overlap": avg_term_overlap,
            "num_contexts": len(retrieved_contexts)
        }
        
        self.metrics[query] = metrics
        return metrics