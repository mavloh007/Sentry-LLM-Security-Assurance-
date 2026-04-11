import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


DEFAULT_TURNS = [
    "Hi, what is my withdrawal limit?",
    "If I exceed it, what can I do next?",
    "What verification might be needed for a large withdrawal?",
]


@dataclass
class TurnResult:
    user_index: int
    turn_index: int
    ok: bool
    status_code: int
    latency_ms: float
    conversation_id: Optional[str]
    response_preview: str
    error: str


@dataclass
class UserResult:
    user_index: int
    login_ok: bool
    login_error: str
    turns: List[TurnResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run concurrent 3-turn chat load test against Flask /api/chat."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:3000",
        help="Base URL of app, e.g. http://127.0.0.1:3000 or https://<site>.azurewebsites.net",
    )
    parser.add_argument("--users", type=int, default=20, help="Number of concurrent simulated users")
    parser.add_argument("--timeout", type=float, default=45.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--ramp-ms",
        type=int,
        default=0,
        help="Optional delay between starting each simulated user (milliseconds)",
    )
    parser.add_argument(
        "--email",
        default="",
        help="Single login email to reuse across all simulated users (if --users-file not provided)",
    )
    parser.add_argument(
        "--password",
        default="",
        help="Single login password to reuse across all simulated users (if --users-file not provided)",
    )
    parser.add_argument(
        "--users-file",
        default="",
        help="Path to JSON file containing credentials list: [{\"email\":\"...\",\"password\":\"...\"}, ...]",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON file path for detailed results",
    )
    return parser.parse_args()


def load_credentials(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.users_file:
        raw = json.loads(Path(args.users_file).read_text(encoding="utf-8"))
        creds = [(item["email"], item["password"]) for item in raw if item.get("email") and item.get("password")]
        if len(creds) < args.users:
            raise ValueError(f"users-file has {len(creds)} credentials, but --users={args.users}")
        return creds[: args.users]

    if not args.email or not args.password:
        raise ValueError("Provide --email and --password, or use --users-file")

    return [(args.email, args.password) for _ in range(args.users)]


async def login(client: httpx.AsyncClient, email: str, password: str) -> Tuple[bool, str]:
    data = {"email": email, "password": password}
    resp = await client.post("/login", data=data)

    # Flask login route renders login page again with 200 on invalid creds.
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:200].replace("\n", " ")
        return False, f"login_http_{resp.status_code}: {snippet}"

    text = (resp.text or "").lower()
    if "invalid email or password" in text:
        return False, "invalid_credentials"

    # Verify authenticated session by calling protected route.
    verify = await client.get("/chat")
    if verify.status_code >= 400:
        snippet = (verify.text or "")[:200].replace("\n", " ")
        return False, f"chat_verify_http_{verify.status_code}: {snippet}"

    return True, ""


async def run_user(
    user_index: int,
    base_url: str,
    timeout_s: float,
    email: str,
    password: str,
    turns: List[str],
    start_delay_s: float = 0.0,
) -> UserResult:
    timeout = httpx.Timeout(timeout_s)
    results: List[TurnResult] = []

    if start_delay_s > 0:
        await asyncio.sleep(start_delay_s)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, follow_redirects=True) as client:
        try:
            login_ok, login_error = await login(client, email, password)
        except Exception as e:
            return UserResult(
                user_index=user_index,
                login_ok=False,
                login_error=f"login_exception[{type(e).__name__}]: {repr(e)}",
                turns=[],
            )

        if not login_ok:
            return UserResult(user_index=user_index, login_ok=False, login_error=login_error, turns=[])

        conversation_id: Optional[str] = None

        for turn_idx, message in enumerate(turns, start=1):
            payload: Dict[str, Any] = {"message": message}
            if turn_idx == 1:
                payload["new_session"] = True
                payload["conversation_title"] = f"LoadTest User {user_index}"
            if conversation_id:
                payload["conversation_id"] = conversation_id

            start = time.perf_counter()
            ok = False
            status_code = 0
            response_preview = ""
            error = ""
            returned_conversation_id: Optional[str] = conversation_id

            try:
                resp = await client.post("/api/chat", json=payload)
                latency_ms = (time.perf_counter() - start) * 1000
                status_code = resp.status_code

                if resp.status_code == 200:
                    data = resp.json()
                    returned_conversation_id = data.get("conversation_id") or conversation_id
                    reply_text = data.get("response")
                    response_preview = ((reply_text or "")[:120]).replace("\n", " ")
                    ok = bool(reply_text)
                    if not ok:
                        error = "empty_response"
                else:
                    snippet = (resp.text or "")[:200].replace("\n", " ")
                    error = f"http_{resp.status_code}: {snippet}"
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                error = f"request_exception[{type(e).__name__}]: {repr(e)}"

            conversation_id = returned_conversation_id
            results.append(
                TurnResult(
                    user_index=user_index,
                    turn_index=turn_idx,
                    ok=ok,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    conversation_id=conversation_id,
                    response_preview=response_preview,
                    error=error,
                )
            )

    return UserResult(user_index=user_index, login_ok=True, login_error="", turns=results)


def summarize(all_results: List[UserResult], expected_turns: int) -> Dict[str, Any]:
    turns: List[TurnResult] = [turn for user in all_results for turn in user.turns]
    latencies = [t.latency_ms for t in turns if t.latency_ms > 0]
    ok_count = sum(1 for t in turns if t.ok)
    fail_count = len(turns) - ok_count

    summary = {
        "users_total": len(all_results),
        "users_login_failed": sum(1 for u in all_results if not u.login_ok),
        "expected_turns_total": len(all_results) * expected_turns,
        "executed_turns_total": len(turns),
        "ok_turns": ok_count,
        "failed_turns": fail_count,
        "success_rate_pct": round((ok_count / len(turns) * 100), 2) if turns else 0.0,
        "p50_ms": round(statistics.median(latencies), 2) if latencies else None,
        "p95_ms": round(statistics.quantiles(latencies, n=100)[94], 2) if len(latencies) >= 20 else None,
        "max_ms": round(max(latencies), 2) if latencies else None,
    }

    per_user = []
    for user in all_results:
        user_turns = user.turns
        per_user.append(
            {
                "user_index": user.user_index,
                "login_ok": user.login_ok,
                "login_error": user.login_error,
                "turns_ok": sum(1 for t in user_turns if t.ok),
                "turns_failed": sum(1 for t in user_turns if not t.ok),
                "last_conversation_id": user_turns[-1].conversation_id if user_turns else None,
            }
        )

    summary["per_user"] = per_user
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Load Test Summary ===")
    print(f"Users total:            {summary['users_total']}")
    print(f"Users login failed:     {summary['users_login_failed']}")
    print(f"Expected turns:         {summary['expected_turns_total']}")
    print(f"Executed turns:         {summary['executed_turns_total']}")
    print(f"Successful turns:       {summary['ok_turns']}")
    print(f"Failed turns:           {summary['failed_turns']}")
    print(f"Success rate:           {summary['success_rate_pct']}%")
    print(f"Latency p50 (ms):       {summary['p50_ms']}")
    print(f"Latency p95 (ms):       {summary['p95_ms']}")
    print(f"Latency max (ms):       {summary['max_ms']}")

    failed_users = [u for u in summary["per_user"] if (not u["login_ok"]) or u["turns_failed"] > 0]
    if failed_users:
        print("\nUsers with issues:")
        for u in failed_users:
            print(
                f"- user={u['user_index']} login_ok={u['login_ok']} "
                f"turns_failed={u['turns_failed']} login_error={u['login_error']}"
            )


async def main_async() -> None:
    args = parse_args()
    creds = load_credentials(args)
    turns = DEFAULT_TURNS

    print(f"Starting load test: users={args.users}, turns_per_user={len(turns)}, base_url={args.base_url}")

    tasks = [
        run_user(
            user_index=i + 1,
            base_url=args.base_url,
            timeout_s=args.timeout,
            email=creds[i][0],
            password=creds[i][1],
            turns=turns,
            start_delay_s=(i * args.ramp_ms) / 1000.0,
        )
        for i in range(args.users)
    ]

    started = time.perf_counter()
    all_results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started

    summary = summarize(all_results, expected_turns=len(turns))
    summary["elapsed_seconds"] = round(elapsed, 2)
    print_summary(summary)
    print(f"Total wall time (s):    {summary['elapsed_seconds']}")

    if args.out:
        detail = {
            "summary": summary,
            "users": [
                {
                    "user_index": u.user_index,
                    "login_ok": u.login_ok,
                    "login_error": u.login_error,
                    "turns": [asdict(t) for t in u.turns],
                }
                for u in all_results
            ],
        }
        Path(args.out).write_text(json.dumps(detail, indent=2), encoding="utf-8")
        print(f"Detailed report written to: {args.out}")


if __name__ == "__main__":
    asyncio.run(main_async())
