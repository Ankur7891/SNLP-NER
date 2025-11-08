from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class Entity(BaseModel):
    name: str = Field(..., description="Minimal canonical term for the entity.")
    mention: Optional[str] = Field(None, description="Exact phrase from the text.")
    type: str = Field(
        ..., description="Type of entity (Gene, Protein, Mutation, Drug, Disease)."
    )


class EntityExtractionOutput(BaseModel):
    entities: Dict[str, List[str]] = Field(
        ...,
        example={
            "Gene": ["BRCA1"],
            "Disease": ["Breast Cancer"],
            "Drug": ["Olaparib"],
            "Protein": ["PARP"],
        },
    )


def extract_entities(
    llm: ChatGoogleGenerativeAI, text: str, system_prompt: str
) -> dict:
    """
    Extract biomedical entities categorized by type.
    """
    prompt = ChatPromptTemplate.from_template(
        """
{system_prompt}

Extract all biomedical entities mentioned in the following text
and organize them under their types (Gene, Protein, Mutation, Drug, Disease).

Return only the "entities" JSON part as per the format above.

Text:
{text}
"""
    )
    chain = prompt | llm
    response = chain.invoke({"text": text, "system_prompt": system_prompt})
    return response.content
