# TensorFlow & Keras Best Practices Auditor

This repository contains **skill.md**, a structured auditing framework designed for **LLM coding agents** to evaluate machine learning repositories. This skill enables an agent to systematically verify if a project adheres to industry-standard engineering and research practices within the **TensorFlow and Keras ecosystem**.

## Purpose

The Auditor skill guides a coding agent through a multi-stage review process to ensure that a repository is not just functional, but **scalable, reproducible, and production-ready**. It forces the agent to look beyond the model architecture and evaluate the **entire infrastructure lifecycle**, from data orchestration to numerical stability.

## Core Auditing Categories

1.  **Research Lifecycle:** Verifies the establishment of **simple baselines and heuristics** before introducing model complexity.
2.  **Keras API Standards:** Ensures proper use of the **Functional API** for non-linear topologies and **Model Subclassing** for dynamic control flows.
3.  **Data Pipeline Efficiency:** Checks for asynchronous transformations like **prefetching** and **interleaving** with `tf.data.AUTOTUNE` to prevent GPU data starvation.
4.  **Performance Engineering:** Audits the use of `@tf.function` for graph compilation and **XLA optimization**.
5.  **Numerical Stability:** Inspects for **gradient clipping** and numerical guards to prevent catastrophic **NaN or Infinity** values.
6.  **Distributed Training:** Validates the implementation of `tf.distribute.Strategy` for multi-GPU or multi-node scaling.
7.  **Scientific Reproducibility:** Confirms a **four-level seeding hierarchy** and the use of deterministic operations.
8.  **Deployment & MLOps:** Checks for standardized formats like **.keras** and the use of tracking tools like **Weights & Biases or MLflow**.

## Sources and Methodology

The auditing criteria in this repository were synthesized from a comprehensive corpus of TensorFlow documentation and expert research methodologies:

*   **Google's "Rules of Machine Learning":** Provided the foundational philosophy of prioritizing **infrastructure trust** and data validation (Phase I) over algorithmic complexity.
*   **Keras 3 Multi-Backend Documentation:** Informed the standards for **backend-agnostic development**, allowing models to run seamlessly on JAX, PyTorch, or TensorFlow.
*   **TensorFlow Performance Guides:** Supplied the technical benchmarks for optimizing **`tf.data` pipelines** and identifying bottlenecks using the **TensorFlow Profiler**.
*   **Numerical Debugging Protocols:** Derived from expert guides on using **TensorBoard Debugger V2** and `tf.debugging` utilities to trace the root causes of numerical instability.
*   **MLOps & Experiment Tracking:** Integrated best practices from the **TensorFlow Model Garden** and industry-leading tools for logging hyperparameters and model lineage.
*   **Reproducibility Studies:** Established the rigorous requirements for **deterministic execution** and global seeding across Python, NumPy, and TensorFlow.
