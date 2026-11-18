# Request Envelope: Provenance and Drift Detection Example

This example demonstrates how to use **RequestEnvelope** to track the complete runtime context of an AI agent, including:
- **System instructions**: Guiding the model's role and policies.
- **Tool schemas**: Function definitions and parameter JSON schemas (`lookup_order`, `process_refund`).
- **Retrieval context**: RAG knowledge base document IDs and chunk hashes.
- **Hyperparameters**: Sampling parameters like `temperature`, `top_p`, and `response_format`.
- **Uncaptured dimensions**: Declared dimensions not directly captured in the envelope (such as client session state or device type).

## Comparable vs Confounded Runs

When monitoring an agent for behavioral drift:
- **Comparable runs**: The prompt template, rendered variables, and all request envelope components (`system`, `tools`, `retrieval_context`, `hyperparameters`) remain identical to the accepted baseline. In this state, repeated test failures indicate potential provider behavioral drift (`observed_model_drift`).
- **Confounded runs**: If an engineer modifies a tool parameter schema, updates a RAG chunk, or tweaks the temperature, PromptDrift detects this change and attributes failures to `prompt_changed` with specific evidence naming the modified component (e.g., `request_envelope.tools differs from the accepted baseline.`), preventing false attribution of model drift.

## Running this Example

You can run this example offline using the deterministic `mock` provider:

```bash
# 1. Run the test suite
promptdrift run -c examples/request-envelope/promptdrift.yaml

# 2. Accept the initial baseline
promptdrift baseline --accept -c examples/request-envelope/promptdrift.yaml

# 3. Monitor for regressions and drift
promptdrift monitor -c examples/request-envelope/promptdrift.yaml --samples 2
```
