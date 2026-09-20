from fastapi.testclient import TestClient

from vllm_cloudru.main import create_app


def test_health_reports_ready():
    client = TestClient(create_app())
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
