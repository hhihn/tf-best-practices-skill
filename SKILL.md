---
name: tf-best-practices
description: Review tensorflow code for common mistakes, anti-patterns, and rule violations
disable-model-invocation: true
argument-hint: "[file or directory, defaults to current project]"
allowed-tools: Read, Glob, Grep, Agent
---

*Version 1.1.0*
This `skill.md` provides a comprehensive checklist of best practices within the **TensorFlow and Keras ecosystem**. It is designed for use by a coding agent to evaluate whether a repository adheres to established engineering and research standards.

# TensorFlow & Keras Best Practices Checklist

## 1. Research Lifecycle & Infrastructure
* **Establish Baselines First:** Does the project define simple heuristics or human-curated ranking systems as a baseline before implementing complex ML models?
* **Metric Instrumentation:** Are granular system behaviors (e.g., granular user interactions) tracked via structured instrumentation rather than fragile logging like string grepping?
* **Data/Code Separation:** Is the repository code-only, with large datasets and image files hosted externally in cloud storage (e.g., S3, GCS)?
* **Validation Strategy:** Does the project use a "Phase I" approach, prioritizing infrastructure trust and data validation before model complexity?

## 2. Coding Standards & Keras API Usage
* **High-Level API Preference:** Does the code prioritize the use of high-level Keras APIs (`model.fit()`, `model.evaluate()`, `model.predict()`) for standard workflows?
* **Model Architecture:**
    *   Is the **Functional API** used for models with non-linear topologies (multiple inputs/outputs)?
    *   Is **Model Subclassing** reserved for dynamic control flows (e.g., recursive networks)?
    *   Does it use a hybrid approach: subclassed layers for complex modules assembled via the Functional API?
* **Custom Layer Implementation:**
    *   Do custom layers defer weight creation to the `build(self, input_shape)` method instead of `__init__`?
    *   Are internal activations/regularization penalties added via `self.add_loss()`?
* **Keras 3 Multi-Backend Compatibility:**
    *   Does the code use `keras.ops` instead of `tf.*` for mathematical operations to remain backend-agnostic?
    *   Are RNG-using layers (like custom dropout) using the stateless `keras.random` ops with a `SeedGenerator`?

## 3. Data Pipeline Optimization (`tf.data`)
* **Asynchrony:** Is `tf.data.Dataset.prefetch(tf.data.AUTOTUNE)` applied at the end of the pipeline to decouple the producer and consumer?
* **Parallelism:**
    *   Does `map` use `num_parallel_calls=tf.data.AUTOTUNE`?
    *   Does `interleave` use `num_parallel_calls=tf.data.AUTOTUNE` to parallelize data extraction?
* **Caching:** Is `dataset.cache()` used after expensive preprocessing steps if the data fits in memory?
* **Vectorization:** are `batch()` transformations applied *before* `map()` operations to vectorize user-defined functions where possible?
* **Cloud Storage Efficiency:** If using remote storage (S3/GCS), is data sharded into 100 MB – 1 GB files and accessed sequentially?

## 4. Performance & Custom Training
* **Graph Compilation:** Are custom training steps or performance-critical functions wrapped in the `@tf.function` decorator?
* **Eager Debugging:** Is there a mechanism to easily disable `@tf.function` for line-by-line inspection (Eager mode) during debugging?
* **XLA Optimization:** Is `jit_compile=True` utilized (default in Keras 3 on GPU) for accelerated linear algebra compilation?

## 5. Numerical Stability & Debugging
* **Numerical Guards:** Are `tf.debugging` utilities (like `check_numerics`) or the **TensorBoard Debugger V2** used to catch NaNs and Infs?
* **Exploding Gradient Prevention:** Does the optimizer or custom training loop implement gradient clipping (via `clipnorm` or `clipvalue`)?
* **Stable Operations:** Are logarithms or exponentials stabilized with small epsilon values (e.g., `log(x + epsilon)`) to prevent `log(0)` errors?
* **Normalization:** Is input data normalized (mean 0, std dev 1) to ensure stable gradient flow?

## 6. Distributed Training & Scalability
* **Strategy Usage:** Does the code use `tf.distribute.Strategy` (e.g., `MirroredStrategy` for single-node multi-GPU or `MultiWorkerMirroredStrategy` for multi-node)?
* **Scope Discipline:** Are the model and optimizer created strictly within the `strategy.scope()`?
* **Advanced Parallelism:** For massive models exceeding single-device memory, is **DTensor** (SPMD approach) utilized?

## 7. Reproducibility & Tracking
* **The Seeding Hierarchy:** Are seeds set across all four levels?
    1. Python `random.seed()`
    2. OS `os.environ['PYTHONHASHSEED']`
    3. NumPy `np.random.seed()`
    4. TensorFlow Global `tf.random.set_seed()`
* **Deterministic Ops:** Is the environment configured for determinism via `os.environ['TF_DETERMINISTIC_OPS'] = '1'`?
* **Experiment Tracking:** Is an automated tool like **Weights & Biases (W&B)** or **MLflow** used to log hyperparameters, metrics, git commits, and model weights?

## 8. Deployment & MLOps
* **Standardized Formats:** are models saved in the `.keras` (Keras 3) or `SavedModel` (TF 2) format for production interoperability?
* **Lineage & Retraining:** Does the project use an MLOps framework like **TFX (TensorFlow Extended)** to automate model lineage and retraining?
* **Edge Deployment:** For mobile or embedded targets, are models optimized and converted for **LiteRT** (formerly TF lite)?

