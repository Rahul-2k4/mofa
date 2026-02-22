# Agent Hub Integration

This section explains how hub entries connect to MoFA documentation and developer workflows.

## Integration Targets

Hub metadata should map cleanly to:

- docs pages in `docs/mofa-doc/src/`
- runnable examples in `examples/`
- Studio-oriented snippet guidance

## Documentation Mapping

When adding a new node or agent type:

1. add/update the upstream registry entry
2. add or update the related docs page
3. add at least one reference example (if behavior is non-trivial)

## Update Workflow

Use this path for stale or incorrect entries:

1. open issue/PR in `mofa-org/mofa-node-hub` for registry fixes
2. open docs update PR in `mofa-org/mofa` if docs mapping also changes
3. link the two PRs for traceability

## Validation Checklist

- entry metadata parses correctly
- linked docs pages exist
- example/snippet references are not broken
- compatibility notes are present when required
