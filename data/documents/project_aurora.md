# Project Aurora: Real-Time Fraud Detection Pipeline (Sanitized Sample)

> **Notice**: Fictional project portfolio document for the Personal AI Knowledge Assistant.

## Project Overview
Project Aurora is a distributed, low-latency transaction scoring service built to identify fraudulent payments in real time across digital banking channels.

## Implementation & Deployment Status
- **Status: DEPLOYED (Production on AWS EKS since August 2024)**.
- Currently serving live production traffic for 12 enterprise banking partners.
- Peak throughput: 15,000 transactions per second.
- Latency budget: p99 latency under 18 milliseconds.

## System Architecture & Technologies
- **Ingestion**: Apache Kafka message queues streaming JSON transaction events.
- **In-Memory Cache**: Redis cluster used for feature lookups (sliding 24-hour transaction frequency, velocity checks).
- **Inference Engine**: Custom PyTorch deep learning classifier loaded onto optimized ONNX Runtime instances.
- **Serving Layer**: Asynchronous FastAPI microservices running in Docker containers orchestrated by AWS EKS.
- **Observability**: Prometheus metrics and Grafana dashboards tracking drift, latency, and error rates.

## Documented Results & Impact
- Reduced payment fraud chargebacks by 34% within the first four months of full deployment.
- Maintained 99.99% service availability across two AWS availability zones.
- Designed automated fallback rules that route payments safely if inference latency exceeds 35 milliseconds.
