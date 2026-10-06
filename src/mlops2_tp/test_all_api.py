import json
import os
from pathlib import Path

import requests

from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = "http://localhost:8800/v1/predict"
API_KEY = os.getenv("API_KEY")

TEST_CASES_DIR = Path(__file__).parent / "test_cases"


def load_test_case(filename: str) -> dict:
    file_path = TEST_CASES_DIR / filename

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def test_fastapi(data: dict) -> dict:
    headers = {
        "X-API-KEY": API_KEY,
    }

    response = requests.post(
        FASTAPI_URL,
        json=data,
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def main():
    test_cases = {
        "potable": "potable.json",
        "non_potable": "non_potable.json",
    }

    for case_name, filename in test_cases.items():

        print(f"\n=== Caso: {case_name} ===")

        data = load_test_case(filename)

        fastapi_result = test_fastapi(data)

        print(
            f"FastAPI -> "
            f"{fastapi_result['label']} "
            f"(prediction={fastapi_result['prediction']}, "
            f"probability={fastapi_result['probability']:.4f}, "
            f"model_version={fastapi_result['model_version']})"
        )


if __name__ == "__main__":
    main()