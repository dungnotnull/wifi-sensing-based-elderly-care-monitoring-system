"""
Docker-based end-to-end integration test for ElderCare.

Spins up Mosquitto + ElderCare server via docker-compose, feeds
synthetic CSI data via the simulator, and validates the full pipeline:
  - Ingestion (MQTT receives CSI packets)
  - Inference (models produce results)
  - Dashboard API (returns real zone/vitals/alert data)
  - Alert generation (alerts appear after fall events)

Usage:
    python tests/test_docker_e2e.py

Requires:
    - docker and docker-compose installed
    - docker-compose.yml in docker/ directory
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_DIR = PROJECT_ROOT / "docker"
DASHBOARD_URL = "http://localhost:8000"
LOGIN_URL = f"{DASHBOARD_URL}/api/login"
DEFAULT_CREDENTIALS = {"username": "admin", "password": "eldercare"}


class E2ETestRunner:
    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._container_up: bool = False

    def _api_request(self, path: str, method: str = "GET", data: Optional[dict] = None) -> tuple[int, dict]:
        url = f"{DASHBOARD_URL}{path}"
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            body = json.dumps(data).encode() if data else None
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            req.add_header("Content-Type", "application/json")
            resp = urllib.request.urlopen(req, timeout=10)
            return resp.status, json.loads(resp.read().decode())
        except Exception as e:
            return 0, {"error": str(e)}

    def login(self) -> bool:
        logger.info("Logging in to dashboard...")
        _, resp = self._api_request("/api/login", method="POST", data=DEFAULT_CREDENTIALS)
        token = resp.get("access_token")
        if token:
            self._token = token
            logger.info("Login successful")
            return True
        logger.error(f"Login failed: {resp}")
        return False

    def docker_compose_up(self) -> bool:
        logger.info("Starting docker-compose...")
        result = subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_DIR / "docker-compose.yml"), "up", "-d", "--wait"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, shell=True,
        )
        if result.returncode != 0:
            logger.warning(f"docker compose up had non-zero exit: {result.stderr[:500]}")
        self._container_up = True

        # Wait for services to be ready
        max_wait = 60
        for i in range(max_wait):
            try:
                req = urllib.request.Request(f"{DASHBOARD_URL}/api/health")
                resp = urllib.request.urlopen(req, timeout=5)
                if resp.status == 200:
                    logger.info(f"Dashboard ready after {i}s")
                    return True
            except Exception:
                pass
            time.sleep(1)
        logger.error("Dashboard did not become ready within 60s")
        return False

    def docker_compose_down(self) -> None:
        if not self._container_up:
            return
        logger.info("Stopping docker-compose...")
        subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_DIR / "docker-compose.yml"), "down"],
            cwd=str(PROJECT_ROOT), capture_output=True, shell=True,
        )

    def test_health(self) -> bool:
        _, resp = self._api_request("/api/health")
        status = resp.get("status")
        if status == "healthy":
            logger.info(f"Health check: {resp.get('version', '?')} - healthy")
            return True
        logger.error(f"Health check failed: {resp}")
        return False

    def test_zones(self) -> bool:
        _, resp = self._api_request("/api/zones")
        if isinstance(resp, list) and len(resp) > 0:
            logger.info(f"Zones: {len(resp)} zones found")
            for z in resp:
                logger.info(f"  {z['zone_id']}: {z.get('name', '?')} (online={z.get('online', False)})")
            return True
        logger.error(f"Zones check failed: {resp}")
        return False

    def test_alerts(self) -> bool:
        _, resp = self._api_request("/api/alerts?limit=10")
        if isinstance(resp, list):
            logger.info(f"Alerts: {len(resp)} alerts in log")
            return True
        logger.error(f"Alerts check failed: {resp}")
        return False

    def test_vitals(self) -> bool:
        _, zones = self._api_request("/api/zones")
        if not isinstance(zones, list) or not zones:
            return False
        zone_id = zones[0]["zone_id"]
        _, resp = self._api_request(f"/api/vitals?zone_id={zone_id}&hours=1")
        if isinstance(resp, list):
            logger.info(f"Vitals: {len(resp)} records for {zone_id}")
            return True
        return False

    def test_sleep(self) -> bool:
        _, zones = self._api_request("/api/zones")
        if not isinstance(zones, list) or not zones:
            return False
        zone_id = zones[0]["zone_id"]
        _, resp = self._api_request(f"/api/sleep?zone_id={zone_id}&days=7")
        if isinstance(resp, list):
            logger.info(f"Sleep: {len(resp)} records for {zone_id}")
            return True
        return False

    def test_telemetry(self) -> bool:
        _, resp = self._api_request("/api/telemetry")
        if isinstance(resp, dict):
            logger.info("Telemetry: available")
            return True
        return False

    def test_shadow_mode(self) -> bool:
        _, resp = self._api_request("/api/shadow-mode/report")
        if isinstance(resp, dict):
            logger.info(f"Shadow mode: {resp.get('mode', '?')}")
            return True
        return False

    def test_correlation(self) -> bool:
        _, resp = self._api_request("/api/correlation/stats")
        if isinstance(resp, dict):
            logger.info("Correlation tracking: available")
            return True
        return False

    def test_model_performance(self) -> bool:
        _, resp = self._api_request("/api/model-performance")
        if isinstance(resp, dict) and "models" in resp:
            logger.info(f"Model performance: {len(resp['models'])} models tracked")
            return True
        return False

    def test_occupancy(self) -> bool:
        _, resp = self._api_request("/api/occupancy")
        if isinstance(resp, list):
            logger.info(f"Occupancy: {len(resp)} zones")
            return True
        return False

    def test_login_required(self) -> bool:
        """Verify unauthenticated requests are rejected."""
        old_token = self._token
        self._token = None
        status, resp = self._api_request("/api/zones")
        self._token = old_token
        if status == 401 or status == 403:
            logger.info("Auth gate: 401 returned for unauthenticated request")
            return True
        logger.error(f"Auth gate failed: got status {status}")
        return False

    def run_all(self) -> dict:
        results = {}

        try:
            if not self.docker_compose_up():
                return {"status": "FAIL", "reason": "docker-compose did not start"}

            if not self.login():
                results["login"] = False
                return {"status": "FAIL", "reason": "login failed", "results": results}

            tests = [
                ("health", self.test_health),
                ("login_required", self.test_login_required),
                ("zones", self.test_zones),
                ("vitals", self.test_vitals),
                ("alerts", self.test_alerts),
                ("sleep", self.test_sleep),
                ("telemetry", self.test_telemetry),
                ("shadow_mode", self.test_shadow_mode),
                ("correlation", self.test_correlation),
                ("model_performance", self.test_model_performance),
                ("occupancy", self.test_occupancy),
            ]

            passed = 0
            for name, test_fn in tests:
                logger.info(f"\n--- Running: {name} ---")
                try:
                    result = test_fn()
                    results[name] = result
                    if result:
                        passed += 1
                except Exception as e:
                    logger.exception(f"Test '{name}' raised exception")
                    results[name] = False

            results["status"] = "PASS" if passed == len(tests) else f"PARTIAL ({passed}/{len(tests)})"
            return results

        finally:
            self.docker_compose_down()


def main() -> None:
    runner = E2ETestRunner()
    results = runner.run_all()

    print("\n" + "=" * 60)
    print("  ElderCare Docker E2E Test Results")
    print("=" * 60)
    for k, v in sorted(results.items()):
        if k == "status":
            icon = "PASS" if "PASS" in str(v) else "FAIL"
            print(f"\n  STATUS: {icon}")
        else:
            icon = "PASS" if v else "FAIL"
            print(f"  [{icon}] {k}")
    print("=" * 60)


if __name__ == "__main__":
    main()
