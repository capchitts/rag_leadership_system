import os

structure = {
    "rag_leadership_system": {
        "data/reports": ["sample_report.pdf"],

        "ingestion": [
            "pdf_loader.py",
            "text_extractor.py",
            "table_extractor.py",
            "image_extractor.py"
        ],

        "processing": [
            "hierarchical_chunker.py",
            "metadata_extractor.py",
            "document_cleaner.py"
        ],

        "embedding": [
            "embedder.py"
        ],

        "indexing": [
            "vector_index.py",
            "bm25_index.py"
        ],

        "retrieval": [
            "hybrid_retriever.py",
            "reranker.py",
            "query_expander.py"
        ],

        "prompts": [
            "executive_report_prompt.py",
            "query_expansion_prompt.py",
            "faithfulness_prompt.py",
            "groundedness_prompt.py"
        ],

        "generation": [
            "context_builder.py",
            "context_packer.py",
            "insight_generator.py"
        ],

        "llm": [
            "llm_client.py"
        ],

        "evaluation": [
            "retrieval_metrics.py",
            "faithfulness.py",
            "groundedness.py",
            "evaluator.py"
        ],

        "pipeline": [
            "rag_pipeline.py"
        ],

        "tests": [
            "test_ingestion.py",
            "test_chunking.py",
            "test_retrieval.py",
            "test_generation.py",
            "test_evaluation.py"
        ],

        "": [
            "main.py",
            "demo_run.py",
            "requirements.txt",
            "README.md"
        ]
    }
}

def create_project(base, structure):

    for folder, files in structure.items():

        root = os.path.join(base, folder)
        os.makedirs(root, exist_ok=True)

        for subfolder, subfiles in files.items():

            path = os.path.join(root, subfolder)
            os.makedirs(path, exist_ok=True)

            for file in subfiles:
                file_path = os.path.join(path, file)

                with open(file_path, "w") as f:
                    f.write("")

if __name__ == "__main__":
    create_project(".", structure)
    print("Project structure created successfully!")