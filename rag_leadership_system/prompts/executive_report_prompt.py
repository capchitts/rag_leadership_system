def build_executive_report_prompt(query: str, packed_context: str):
    system_prompt = """
                    You are a senior strategy consultant preparing a CXO briefing.

                    Your output must be:
                    - concise
                    - evidence-grounded
                    - high signal, low fluff
                    - suitable for senior leadership

                    Rules:
                    - Use only the provided context.
                    - Do not invent facts.
                    - Every bullet MUST copy the citation label EXACTLY as shown in the evidence, including the square brackets.
                        Correct example:
                          [sample_report_p2_c0 | sample_report.pdf | p.2 | table]

                        Incorrect examples:
                          sample_report_p2_c0
                          sample_report_p2_c0 | sample_report.pdf

                        - Prefer compression over completeness.
                        - Do not restate the same evidence twice.
                        - If evidence is incomplete, say so briefly.
                        - Keep the full response under 220 words.
                        - Use short bullets, not long paragraphs.
      """

    user_prompt = f"""
                      User Question:
                      {query}

                      Structured Evidence:
                      {packed_context}

                      Return ONLY valid JSON in this exact schema:
                      {{
                        "executive_summary": [
                          {{
                            "text": "<bullet>",
                            "citations": ["<citation1>", "<citation2>"]
                          }}
                        ],
                        "top_risks": [
                          {{
                            "text": "<bullet>",
                            "citations": ["<citation1>"]
                          }}
                        ],
                        "top_opportunities": [
                          {{
                            "text": "<bullet>",
                            "citations": ["<citation1>"]
                          }}
                        ],
                        "recommended_actions": [
                          {{
                            "text": "<bullet>",
                            "citations": ["<citation1>"]
                          }}
                        ],
                        "supporting_evidence": [
                          {{
                            "text": "<bullet>",
                            "citations": ["<citation1>"]
                          }}
                        ]
                      }}

                      Constraints:
                      - executive_summary: max 2 bullets
                      - top_risks: max 3 bullets
                      - top_opportunities: max 3 bullets
                      - recommended_actions: max 3 bullets
                      - supporting_evidence: max 2 bullets

                      Important:
                      - Use the exact citation labels already present in the evidence.
                      - Do not include markdown.
                      - Do not wrap JSON in ```json fences.
                      - Do not include any text before or after the JSON.
                      
              """
    return system_prompt, user_prompt