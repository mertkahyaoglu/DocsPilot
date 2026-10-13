---
title: Webhooks
category: api
updated: 2025-07-22
---

# Webhooks

Webhooks let Acme Billing notify your server when something happens, such as an invoice being paid. They are available on every plan.

## Setting up an endpoint

### Creating the endpoint

Go to Settings → Developers → Webhooks and select Add endpoint. Enter an HTTPS URL and choose the events you want. Each account can have up to 10 endpoints. When the endpoint is created you are shown its signing secret, which starts with `whsec_`.

### Responding

Your server must respond with any `2xx` status code within 10 seconds. Any other status, a redirect, or a timeout counts as a failed delivery. Do your heavy processing after you have returned the response.

## Events

| Event | Sent when |
|---|---|
| `invoice.created` | An invoice is created |
| `invoice.sent` | An invoice is sent to the customer |
| `invoice.paid` | An invoice is fully paid |
| `invoice.overdue` | An invoice passes its due date unpaid |
| `payment.succeeded` | A payment is captured |
| `payment.failed` | A payment attempt fails |
| `subscription.created` | A subscription starts |
| `subscription.past_due` | A subscription's first charge fails |
| `subscription.canceled` | A subscription is canceled, manually or by dunning |

## Payload

```json
{
  "id": "evt_1Pq7Xz",
  "type": "invoice.paid",
  "created": 1752000000,
  "data": {
    "object": {
      "id": "inv_3Zt9Rw",
      "status": "paid",
      "amount_due": 0
    }
  }
}
```

## Verifying signatures

### The header

Every request carries an `Acme-Signature` header:

```
Acme-Signature: t=1752000000,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
```

### How to verify

1. Split the header and read `t` (timestamp) and `v1` (signature).
2. Build the signed string: the timestamp, a period, and the raw request body (`{t}.{body}`).
3. Compute an HMAC-SHA256 of that string using your endpoint's signing secret.
4. Compare the result with `v1` using a constant-time comparison.
5. Reject the request if `t` is more than 5 minutes away from your server's clock.

Always use the raw body. Re-serializing the JSON will change the bytes and break the signature.

## Retries

If a delivery fails, Acme Billing retries it on this schedule, measured from the previous attempt:

| Retry | Delay |
|---|---|
| 1 | 1 minute |
| 2 | 5 minutes |
| 3 | 30 minutes |
| 4 | 2 hours |
| 5 | 6 hours |
| 6 | 24 hours |

After the sixth retry fails, the event is marked failed and is not retried automatically. Endpoints that keep failing can be disabled automatically; see the webhook troubleshooting notes for details.

## Delivery guarantees

Events are delivered at least once, so your handler must be idempotent: use the event `id` to ignore duplicates. Events are not guaranteed to arrive in order.

## Delivery log

Settings → Developers → Webhooks → Deliveries shows every attempt with its status code and response time. Entries are kept for 14 days, and you can resend any event from there.

## Testing

Use the Send test event button on the endpoint page, or create an endpoint with a test-mode key to receive events for sandbox data.
