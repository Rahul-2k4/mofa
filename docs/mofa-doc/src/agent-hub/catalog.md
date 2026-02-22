# Agent Hub Catalog

This page defines the catalog model and UX behavior used by the centralized docs.

## Catalog Fields

At minimum, each catalog entry should expose:

- `name`
- `description`
- `category` (or equivalent grouping)
- `version`
- `source` (repository or package link)

Recommended fields:

- `tags`
- `inputs` and `outputs` summary
- compatibility constraints (runtime/provider/version)
- sample usage snippet

## Search and Filtering

Catalog experiences should support:

- full-text search by `name`, `description`, and `tags`
- filtering by category
- filtering by compatibility labels when available

## Caching Guidance

Use client-side cache for registry fetches with a short TTL (for example 5-10 minutes) to reduce repeated requests and improve page responsiveness.

## Failure Behavior

If the hub source is unavailable:

- show last known cached snapshot when available
- otherwise show an explicit unavailable state
- provide direct link to the upstream registry URL for debugging
