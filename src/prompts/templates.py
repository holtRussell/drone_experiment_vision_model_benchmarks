"""
Prompt Templates - Version Controlled
These prompts MUST NOT vary across pipelines.
"""
from typing import Dict, List
from src.utils.config import load_config


class PromptTemplates:
    """Manage prompt templates"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            config = load_config(config_path)
            self.prompts = config.get('prompts', {})
        else:
            self.prompts = {
                "atomic": {
                    "cars": "How many cars are there?",
                    "pedestrians": "How many pedestrians are there?",
                    "bicycles": "How many bicycles are there?"
                },
                "composite": """Describe the scene. Include:
- Number of cars
- Number of pedestrians
- Number of bicycles
- One sentence describing the environment""",
                "system": "You are a vision analysis assistant. Analyze the provided scene information and answer questions accurately.",
                "ir_prompt": """Based on the following vision processing output, answer the user's question.

Vision Output:
{intermediate_representation}

Question: {query}

Provide a structured response with counts and description as requested."""
            }
    
    def get_atomic_query(self, object_type: str) -> str:
        """Get atomic query for object type"""
        return self.prompts.get("atomic", {}).get(object_type, "")
    
    def get_atomic_queries(self) -> List[str]:
        """Get all atomic queries"""
        atomic = self.prompts.get("atomic", {})
        return list(atomic.values())
    
    def get_composite_query(self) -> str:
        """Get composite query"""
        return self.prompts.get("composite", "")
    
    def get_system_prompt(self) -> str:
        """Get system prompt"""
        return self.prompts.get("system", "")
    
    def get_ir_prompt(self, ir: Dict, query: str) -> str:
        """Get prompt for processing intermediate representation"""
        template = self.prompts.get("ir_prompt", "")
        ir_text = f"Cars: {ir.get('cars', 0)}, Pedestrians: {ir.get('pedestrians', 0)}, Bicycles: {ir.get('bicycles', 0)}"
        if ir.get('raw_description'):
            ir_text += f"\nDescription: {ir['raw_description']}"
        return template.format(intermediate_representation=ir_text, query=query)
