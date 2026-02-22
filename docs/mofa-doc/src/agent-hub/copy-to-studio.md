# Copy to Studio

Use this flow when converting hub catalog items into MoFA Studio dataflows.

## Snippet Expectations

A copy-ready snippet should include:

- node identifier
- required configuration fields with safe defaults
- minimal input/output wiring hints

## Safety Guardrails

Before running copied snippets:

- verify provider/model identifiers exist in your environment
- replace placeholder secrets with local secure configuration
- review version compatibility

## Suggested Snippet Shape

```yaml
name: example-flow
nodes:
  - id: example-node
    type: hub/example-node
    config:
      timeout_ms: 30000
      retries: 1
connections: []
```

Keep snippets minimal and explicit. Avoid implicit defaults that can change behavior across versions.
