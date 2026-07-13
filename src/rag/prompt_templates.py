ZERO_SHOT_TEMPLATE = """
You are a helpful RAG assistant. Use ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{question}
"""

FEW_SHOT_TEMPLATE = """
You are a helpful RAG assistant.

EXAMPLE 1:
Question: What is the net profit?
Answer: The net profit is X according to page Y.

EXAMPLE 2:
Question: What is EBITDA?
Answer: EBITDA is defined as ....

NOW ANSWER THE USER QUESTION USING ONLY THE CONTEXT BELOW.

CONTEXT:
{context}

QUESTION:
{question}
"""

COT_TEMPLATE = """
You are a helpful RAG assistant. Think step-by-step.

CONTEXT:
{context}

QUESTION:
{question}

Let's reason this out step-by-step:
"""
