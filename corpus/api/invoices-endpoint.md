---
title: Invoices API Reference
category: api
updated: 2025-07-22
---

# Invoices API Reference

All endpoints are relative to `https://api.acmebilling.example/v2` and require a bearer API key. Request and response bodies are JSON.

## Create an invoice

`POST /v2/invoices`

```bash
curl https://api.acmebilling.example/v2/invoices \
  -H "Authorization: Bearer ak_test_51Hq0example" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: 7c1e4a52-3f0b-4c9e-9a77-0d3f1d5a8b21" \
  -d '{
    "customer_id": "cus_8Kq2Lm",
    "currency": "usd",
    "due_in_days": 30,
    "line_items": [
      { "description": "Website design", "quantity": 1, "unit_amount": 250000 }
    ]
  }'
```

Amounts are integers in the smallest currency unit, so `250000` is $2,500.00.

Response `201 Created`:

```json
{
  "id": "inv_3Zt9Rw",
  "object": "invoice",
  "status": "draft",
  "customer_id": "cus_8Kq2Lm",
  "currency": "usd",
  "amount_due": 250000,
  "due_date": "2025-08-21",
  "hosted_url": "https://pay.acmebilling.example/i/inv_3Zt9Rw"
}
```

## Other endpoints

| Method and path | Purpose |
|---|---|
| `GET /v2/invoices/{id}` | Retrieve one invoice |
| `GET /v2/invoices` | List invoices |
| `POST /v2/invoices/{id}/send` | Send a draft invoice by email |
| `POST /v2/invoices/{id}/void` | Void an unpaid invoice |

## Pagination

List endpoints accept `limit` (default 20, maximum 100) and `starting_after` (an object id). The response contains `has_more`, which is `true` when another page exists.

## Idempotency

Add an `Idempotency-Key` header to `POST` requests so you can safely retry after a network error. A key is remembered for 24 hours. Reusing a key with a different body returns `409` with the code `idempotency_conflict`.

## Errors

Errors use this shape:

```json
{
  "error": {
    "code": "validation_failed",
    "message": "line_items[0].unit_amount must be a positive integer"
  }
}
```

| HTTP status | Code | Meaning |
|---|---|---|
| 400 | invalid_request | Malformed JSON or missing parameter |
| 401 | invalid_api_key | Missing or wrong key |
| 403 | forbidden | Key lacks permission |
| 404 | not_found | Object does not exist |
| 409 | idempotency_conflict | Idempotency key reused with different body |
| 422 | validation_failed | Parameter value rejected |
| 429 | rate_limited | Too many requests |
| 500 | internal_error | Problem on our side; retry |
