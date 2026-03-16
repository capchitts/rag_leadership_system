from pathlib import Path

from config import settings
from pipeline.ingestion_pipeline import run_ingestion_pipeline


def main():
    project_root = Path(__file__).resolve().parent
    data_path = str((project_root / settings.DATA_DIR).resolve())

    print(f"Running ingestion for: {data_path}")
    tracer = run_ingestion_pipeline(data_path)
    print("Ingestion complete.")
    print(tracer.to_dict())


if __name__ == "__main__":
    main()