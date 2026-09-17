from typing import Any

import requests


class APIClientError(Exception):
    pass


def _error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return str(detail) or f"HTTP {response.status_code}"


def get_health(api_base_url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=5)
    except requests.RequestException as exc:
        raise APIClientError("Could not connect to the FastAPI server.") from exc
    if not response.ok:
        raise APIClientError(_error_detail(response))
    return response.json()


def ask_question(api_base_url: str, question: str) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{api_base_url}/query",
            json={"question": question},
            timeout=180,
        )
    except requests.RequestException as exc:
        raise APIClientError("Could not connect to the FastAPI server.") from exc

    if not response.ok:
        raise APIClientError(_error_detail(response))

    payload = response.json()
    return {
        "answer": payload["answer"],
        "sources": payload.get("sources", []),
        "passages": payload.get("passages", []),
    }
