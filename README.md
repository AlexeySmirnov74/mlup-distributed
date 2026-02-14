<img width="1696" height="608" alt="mlup-distributed" src="https://github.com/user-attachments/assets/2a60dd7d-88d9-4da2-84d8-7e7560457660" />

# 🚀 MLup Distributed (Redis Queue Edition)

> An enhanced, production-ready version of MLup
> featuring a distributed queue, Redis storage, fault tolerance, and horizontal scaling.

----------

# 📌 What is it?

This is a modified version of the base **https://mlup.org** project, where:
* The `worker_and_queue` architecture has been completely redesigned.
* Queues and job statuses are moved from process memory to Redis.
* Horizontal scaling of workers is implemented.
* Metrics (Prometheus + Grafana) are integrated.
* Leader election is implemented.
* Automatic stale job requeueing is active.
* High availability and fault tolerance are provided.

----------

# 🧨 The Problem with Base MLup

In the standard MLup implementation, the flow is:
```
POST /predict → receive predict_id
GET /predict/{predict_id} → receive result
```

However, in the base version:
* Statuses and results are stored **within the process memory**.
* The queue is local to the uvicorn worker.
* The protocol is effectively blocking (waiting up to `ttl_client_wait`).

----------

## ❗ What this means

When running `uvicorn workers=4`, you get:
* **Process A** (memory A)
* **Process B** (memory B)
* **Process C** (memory C)
* **Process D** (memory D)

Each process maintains its own separate queue, set of `predict_id`s, and statuses.

----------

## 💥 The Conflict

* **POST** → Directed to Process A → stored in memory A
* **GET** → Directed to Process C → checking memory C

Process C has no knowledge of the `predict_id` created by Process A, resulting in 404/408 errors or lost IDs.

### 🚨 Instability without Sticky Sessions
Relying on sticky sessions is a bottleneck because it breaks horizontal scaling, doesn't work well in Kubernetes, and offers no protection if a process crashes.

----------

## 🔥 The Core Issue

If the process that accepted the POST request crashes, restarts, or is redeployed, the `predict_id` disappears instantly because it lived only in that specific process's memory.

----------

# 💡 The Solution: MLup Distributed

The queue and status storage are moved to Redis. Now:
* `predict_id` is global.
* Any API process can retrieve the result.
* Any worker can process the task.
* A process crash does not destroy the queue.
* Execution guarantees are implemented.

----------

# 🏗 Architecture

<img width="1696" height="608" alt="structure" src="https://github.com/user-attachments/assets/cbd8a67c-8fed-4c5d-9784-4ed8accbe036" />

* **CLIENT** → Sends request to FastAPI.
* **FastAPI (API)** → Receives request and pushes to Redis.
* **Redis Queue** → Stores the tasks globally.
* **Workers (1-N)** → Distributed Docker containers or machines processing tasks.

----------

# 🧠 Key Features

* **Leader Election**: One worker becomes the leader via Redis lock to handle dead worker cleanup and stale job requeueing.
* **Reliable Queue**: Uses `BRPOPLPUSH` and inflight tracking to ensure tasks are only ACKed upon successful completion.
* **Metrics**: Prometheus + Grafana integration for monitoring queue length, active workers, and request duration.

----------

# 📦 Installation & Run

```
# Install
pip install -r requirements.txt

# Run via Docker
docker compose up --build

# Scale workers
docker compose up --scale worker=4
```

# 🌍 Horizontal Scaling

You can run workers:

* on other machines
* in other containers
* in Kubernetes
* in Docker Swarm
* The main thing is access to Redis.

# 🛡 What is now guaranteed

* No loss of predict_id
* No loss of task when a worker crashes
* No problem with uvicorn workers
* No need for sticky sessions
* Horizontal scaling
* Fault-tolerance
* Production-grade architecture
