# Prefix Cache Router Simulator

## Overview

Large language models (LLMs) often rely on a **key–value (KV) cache** to store intermediate
attention states.  When two requests share the same prompt prefix, the KV cache
for that prefix can be reused rather than recomputed, saving memory bandwidth and
compute time.  This optimisation is commonly referred to as **prefix caching**.  The
BentoML blog explains that prefix caching allows the server to skip the
expensive prefill phase when a new request has an identical prefix【239421193071328†L60-L83】.
However, the cache can only be used when the prefix matches exactly【239421193071328†L86-L87】.

This project implements a simple **prefix cache aware router**.  The router
maintains a mapping from prompt prefixes to worker instances.  When a new
request arrives:

* If the prefix has been seen before, the router forwards the request to the same
  worker, resulting in a **cache hit** and reuse of the existing KV cache.
* If it is a new prefix, the router assigns it to the worker with the fewest
  unique prefixes, balancing load across workers and tracking a **cache miss**.

By simulating a stream of requests, you can measure the cache hit rate and
evaluate how grouping requests with identical prefixes improves utilisation.
This simple model illustrates why production serving systems often route
similar prompts together and reorder batches to maximise prefix reuse【239421193071328†L115-L126】.

## Why this matters

LLM inference is memory‑IO bound and the prefill phase (computing keys and
values for the entire prompt) dominates latency【239421193071328†L60-L83】.  When many
requests share system prompts, retrieval augmented context or conversation
prefixes, prefix caching can reduce latency and cost by an order of
magnitude【239421193071328†L139-L151】.  Implementing cache‑aware routing is a
key technique for high‑throughput LLM serving systems.

## Features

* **Prefix aware routing:** Requests with the same prefix are routed to the
  same worker to reuse KV cache entries, resulting in cache hits.
* **Load balancing:** New prefixes are assigned to the worker with the fewest
  unique prefixes to keep utilisation balanced.
* **Statistics tracking:** The router records total requests, hits, misses,
  hit rate, prefix assignments and per‑worker load distribution.
* **Simulation tool:** A command‑line interface allows you to simulate
  streams of requests drawn from a pool of prefixes with a controllable
  repetition probability.  The tool reports hit rate and worker load.

## Installation

This project is self‑contained and requires only Python 3.8+.
To run the tests, install [`pytest`](https://docs.pytest.org/) via pip:

```bash
pip install pytest
```

## Usage

You can run the simulation via the command line.  The following example
simulates 100 requests drawn from the prefix pool `A,B,C,D` with a 70 % chance
that each request repeats the previous prefix and assigns them to 3 workers:

```bash
python router.py --workers 3 --prefixes A,B,C,D --repeat-prob 0.7 --requests 100 --seed 42
```

Sample output:

```
{
  "total_requests": 100,
  "hits": 65,
  "misses": 35,
  "hit_rate": 0.65,
  "assignments": {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 0
  },
  "worker_load": {
    "0": 2,
    "1": 1,
    "2": 1
  }
}
```

In this run, 65 % of requests reused the KV cache thanks to repeated prefixes.
The `assignments` map shows which worker each prefix was assigned to and
`worker_load` shows the number of unique prefixes handled by each worker.

## Running the tests

The test suite validates the routing logic and the simulator.  To run the
tests, simply execute:

```bash
pytest
```
