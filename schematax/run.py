"""Runs every task x condition against one OpenAI-compatible server and appends results to a journal.

The journal is one JSON line per request, fsynced after each batch, so an interrupted run resumes
where it stopped. Batches are followed by a pause of the same length (--duty) to keep the GPU
from running flat out.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

from .tasks import TASKS


def load_done(path: Path) -> set[tuple]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["task"], r["cond"], r["id"]))
    return done


async def ask(client, model, prompt, schema, max_tokens):
    kw = {}
    if schema is not None:
        kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kw)
    return r.choices[0].message.content or "", r.usage.completion_tokens, time.perf_counter() - t0, r.choices[0].finish_reason


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="model label used in the journal file name")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="local")
    p.add_argument("--tasks", default="gsm8k,agnews,conll")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--duty", type=float, default=0.5, help="fraction of time the GPU is busy")
    args = p.parse_args()

    out = Path("results") / f"{args.label}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = load_done(out)
    client = AsyncOpenAI(base_url=args.base_url, api_key="local", timeout=600)

    jobs = []
    for name in args.tasks.split(","):
        load, build, score, max_tokens = TASKS[name]
        for doc in load(args.n):
            for cond, (prompt, schema) in build(doc).items():
                if (name, cond, doc["id"]) not in done:
                    jobs.append((name, cond, doc, prompt, schema, max_tokens, score))
    total = len(jobs) + len(done)
    print(f"{args.label}: {len(done)} done, {len(jobs)} to go", flush=True)

    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(jobs), args.batch):
            chunk = jobs[start:start + args.batch]
            t0 = time.perf_counter()
            res = await asyncio.gather(*(ask(client, args.model, j[3], j[4], j[5]) for j in chunk))
            busy = time.perf_counter() - t0
            for (name, cond, doc, _, schema, _, score), (text, ntok, secs, finish) in zip(chunk, res):
                ok, s = score(doc, cond, text)
                f.write(json.dumps({"task": name, "cond": cond, "id": doc["id"], "ok": ok, "score": s,
                                    "tokens": ntok, "secs": round(secs, 3), "finish": finish,
                                    "text": text}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n = len(done) + start + len(chunk)
            print(f"{args.label}: {n}/{total} ({busy:.1f}s batch)", flush=True)
            if args.duty < 1:
                await asyncio.sleep(busy * (1 - args.duty) / args.duty)
    print(f"{args.label}: finished", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
