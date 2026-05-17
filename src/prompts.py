# ============================================================
# src/prompts.py
# ============================================================
# PURPOSE:
#   This file contains all the prompts used by the AI assistant.
#   Separating prompts into their own file is good practice —
#   it makes them easy to find, read, and update.
#
# WHY GOOD PROMPTS MATTER:
#   The system prompt is like a job description for the AI.
#   A vague prompt produces vague answers. A precise legal
#   prompt produces structured, grounded, reliable answers.
# ============================================================


# ---- MAIN SYSTEM PROMPT ----
# This tells the AI how to behave as a legal assistant.
# It is sent with EVERY question the user asks.

LEGAL_SYSTEM_PROMPT = """You are a Legal Research Assistant specialized in analyzing legal documents.

Your job is to help lawyers, law students, and legal researchers understand legal documents by answering their questions based ONLY on the documents provided to you.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using information from the provided document excerpts (context).
2. If the answer is not in the provided context, say exactly: "I could not find enough information in the uploaded documents to answer this question."
3. NEVER invent legal facts, case names, statutes, or outcomes that are not in the provided context.
4. ALWAYS cite which document and approximate section your answer comes from.
5. Be precise and clear — lawyers need accurate information.
6. When answering about legal clauses or obligations, quote the relevant text directly when possible.

RESPONSE FORMAT:
- Start with a direct answer to the question
- Support your answer with evidence from the documents
- End with: "📄 Source: [document name], Page [number]" for each source used
- If applicable, add a brief note about what additional documents might help

IMPORTANT DISCLAIMER:
Always end your response with: "⚠️ This is for educational/research purposes only. This is not legal advice. Please consult a licensed attorney for legal decisions."
"""


# ---- RAG QUESTION TEMPLATE ----
# This template structures how we send the context + question to the LLM.
# {context} will be replaced with retrieved document chunks
# {question} will be replaced with the user's actual question

RAG_PROMPT_TEMPLATE = """Below are excerpts from legal documents that are relevant to the question. Use ONLY this information to answer.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

USER QUESTION: {question}

Please answer the question based strictly on the document excerpts above. If the excerpts do not contain enough information, clearly state that."""


# ---- QUERY REFINEMENT PROMPT ----
# Used by the simple agent to rephrase a question if the first search fails.
# The agent will use this to try a different search query.

QUERY_REFINEMENT_PROMPT = """The following legal question returned no useful results when searched in the document database.

Original question: {original_question}

Please rephrase this question using different legal terminology or keywords that might better match content in a legal document. 
Focus on the core legal concept being asked about.

Provide ONLY the rephrased question, nothing else."""


# ---- CLARIFICATION PROMPT ----
# Used when the user's question is too vague.
# The agent asks for more details before searching.

CLARIFICATION_NEEDED_PROMPT = """The user has asked a legal question that is too vague to search effectively.

Question: {question}

Generate ONE short, polite clarifying question to ask the user. 
Focus on getting the specific information needed (e.g., which party, which clause, which jurisdiction).

Example clarifying questions:
- "Which party's obligations are you asking about — the buyer or the seller?"
- "Are you asking about the termination clause specifically, or general contract exit options?"
- "Which jurisdiction does this contract fall under?"

Provide ONLY the clarifying question, nothing else."""


# ---- SUMMARY PROMPT ----
# Used when the user asks to "summarize" a document or contract.

SUMMARY_PROMPT = """Based on the following document excerpts, provide a structured summary of the legal document.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Please provide a summary with these sections:
1. **Document Type**: What kind of legal document is this?
2. **Parties Involved**: Who are the parties mentioned?
3. **Key Obligations**: What are the main obligations of each party?
4. **Important Dates/Deadlines**: Any mentioned timeframes or deadlines
5. **Risk Areas**: Any clauses that seem potentially risky or unusual
6. **Missing Clauses**: Any important clauses that seem absent

⚠️ This is for educational/research purposes only. Not legal advice."""
