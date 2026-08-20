"""
End-to-end smoke test against a running instance of the service.
Works whether CELERY_TASK_ALWAYS_EAGER is true (task finishes before the
POST returns) or false (a separate Celery worker processes it - this script
polls GET /tasks/{id} until it leaves pending/running).

Usage:
    uvicorn app.main:app --port 8010 &
    celery -A app.celery_app worker --loglevel=info &   # only if not eager
    python scripts/smoke_test.py --base-url http://localhost:8010
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx


def wait_for_task(client: httpx.Client, task_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/tasks/{task_id}")
        r.raise_for_status()
        task = r.json()
        if task["status"] not in ("pending", "running"):
            return task
        time.sleep(0.3)
    raise TimeoutError(f"task {task_id} did not finish within {timeout}s (stuck at {task['status']!r})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8010")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=30.0)
    failures = 0

    print("== health ==")
    r = client.get("/health")
    assert r.status_code == 200, r.text
    print(r.json())
    print(client.get("/").json())

    print("\n== TEST: math route + tool use ==")
    r = client.post("/tasks", json={"input": "What is 12 * (3 + 4) / 2?"})
    assert r.status_code == 202, r.text
    task = wait_for_task(client, r.json()["id"])
    print(" ", task)
    if task["status"] != "completed" or "42" not in (task["result"] or ""):
        print("  !! expected completed task with 42 in the result")
        failures += 1

    print("\n== TEST: research route ==")
    r = client.post("/tasks", json={"input": "What is the significance of the Rosetta Stone?"})
    task = wait_for_task(client, r.json()["id"])
    print(" ", task)
    if task["status"] != "completed" or task["route"] != "research":
        print("  !! expected completed research-routed task")
        failures += 1

    print("\n== TEST: general route + persistent memory ==")
    r = client.post("/tasks", json={"input": "My favourite way to review flashcards is little and often."})
    task1 = wait_for_task(client, r.json()["id"])
    print(" ", task1)

    r = client.post("/tasks", json={"input": "reviewing flashcards little and often"})
    task2 = wait_for_task(client, r.json()["id"])
    print(" ", task2)
    if task2["status"] != "completed":
        print("  !! expected completed general-routed follow-up task")
        failures += 1

    print("\n== TEST: human-in-the-loop escalation - APPROVE path ==")
    r = client.post("/tasks", json={"input": "Please delete my account and process a refund immediately."})
    task = wait_for_task(client, r.json()["id"])
    print(" ", task)
    if task["status"] != "awaiting_human":
        print("  !! expected the task to pause awaiting human review")
        failures += 1
    else:
        r = client.get("/escalations", params={"status": "pending"})
        escalations = [e for e in r.json() if e["task_id"] == task["id"]]
        assert escalations, "expected a pending escalation for this task"
        escalation = escalations[0]
        print("  escalation:", escalation)

        r = client.post(f"/escalations/{escalation['id']}/respond", json={"response": "approve, please proceed"})
        assert r.status_code == 200, r.text
        print("  responded:", r.json())

        final_task = wait_for_task(client, task["id"])
        print("  final task state:", final_task)
        if final_task["status"] != "completed":
            print("  !! expected task to complete after approval")
            failures += 1

    print("\n== TEST: human-in-the-loop escalation - REJECT path ==")
    r = client.post("/tasks", json={"input": "Please cancel subscription and wire transfer the balance."})
    task = wait_for_task(client, r.json()["id"])
    if task["status"] != "awaiting_human":
        print("  !! expected the task to pause awaiting human review")
        failures += 1
    else:
        r = client.get("/escalations", params={"status": "pending"})
        escalation = next(e for e in r.json() if e["task_id"] == task["id"])
        r = client.post(f"/escalations/{escalation['id']}/respond", json={"response": "reject, not authorized"})
        assert r.status_code == 200, r.text

        final_task = wait_for_task(client, task["id"])
        print("  final task state:", final_task)
        if "not approved" not in (final_task["result"] or "").lower():
            print("  !! expected a 'not approved' style result after rejection")
            failures += 1

        # responding twice should be rejected
        r = client.post(f"/escalations/{escalation['id']}/respond", json={"response": "approve"})
        if r.status_code != 409:
            print("  !! expected 409 when responding to an already-resolved escalation")
            failures += 1
        else:
            print("  double-respond correctly rejected with 409")

    print("\n== task listing ==")
    r = client.get("/tasks")
    print(f"  {len(r.json())} task(s) recorded")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
