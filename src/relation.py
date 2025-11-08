from pydantic import BaseModel, Field
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class Relation(BaseModel):
    subject: str = Field(..., description="The source entity.")
    relation: str = Field(
        ..., description="The relation type, e.g., inhibits, causes, treats."
    )
    object: str = Field(..., description="The target entity.")


class RelationExtractionOutput(BaseModel):
    relations: List[Relation] = Field(
        ...,
        example=[
            {
                "subject": "BRCA1 mutation",
                "relation": "causes",
                "object": "Breast Cancer",
            },
            {
                "subject": "Olaparib",
                "relation": "treats",
                "object": "BRCA1-related cancers",
            },
            {"subject": "Olaparib", "relation": "inhibits", "object": "PARP proteins"},
        ],
    )


def extract_relations(
    llm: ChatGoogleGenerativeAI, text: str, system_prompt: str
) -> dict:
    """
    Extract biomedical relations as subject–relation–object triplets.
    """
    prompt = ChatPromptTemplate.from_template(
        """
{system_prompt}

Now extract all biomedical relations from the following text.
Represent them as triplets: subject–relation–object.
Return only the "relations" JSON part as per the format above.

Text:
{text}
"""
    )
    chain = prompt | llm
    response = chain.invoke({"text": text, "system_prompt": system_prompt})
    return response.content
