---
title: API Rate Limits
category: api
updated: 2025-06-18
---

# API Rate Limits

The API limits how many requests your account can make per minute. The limit depends on your plan.

## Limits per plan

| Plan | Requests per minute |
|---|---|
| Starter | 60 |
| Growth | 300 |
| Scale | 1,200 |

The limit applies to your whole account, not to individual keys. Test-mode requests and live-mode requests count together. Each call to a list endpoint counts as one request, regardless of how many objects it returns.

Webhook deliveries sent by Acme Billing to your server do not count toward the limit.

## Response headers

Every API response includes these headers:

| Header | Meaning |
|---|---|
| `X-RateLimit-Limit` | The number of requests allowed per minute for your plan |
| `X-RateLimit-Remaining` | Requests left in the current minute |
| `X-RateLimit-Reset` | Unix timestamp when the counter resets |

## When you exceed the limit

The API responds with HTTP `429` and the error code `rate_limited`. The response includes a `Retry-After` header with the number of seconds to wait before trying again.

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Too many requests. Retry after 12 seconds."
  }
}
```

## Handling limits well

- Read `X-RateLimit-Remaining` and slow down when it gets low.
- On a `429`, wait for the number of seconds in `Retry-After` and retry. Use exponential backoff if you keep receiving `429`.
- Prefer list endpoints with `limit=100` over many single lookups.
- Use webhooks instead of polling for status changes.

## Need more?

Higher limits come with a higher plan. See the Pricing and Plans page for the full comparison.
