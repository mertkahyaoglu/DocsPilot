---
title: API Authentication and Keys
category: api
updated: 2025-06-18
---

# API Authentication and Keys

The Acme Billing REST API lets you create invoices, manage customers and subscriptions, and receive events. The base URL for all requests is `https://api.acmebilling.example/v2`.

## API keys

Create keys at Settings → Developers → API keys. A key is shown only once when it is created, so copy it to a secure place immediately. You can have up to 10 active keys per account. API access is available on every plan.

There are two kinds of key, identified by their prefix:

| Prefix | Mode | Behavior |
|---|---|---|
| `ak_test_` | Test | Works on sandbox data. No real money moves and no emails are sent to customers. |
| `ak_live_` | Live | Works on your real data. |

## Sending the key

Send the key as a bearer token in the `Authorization` header:

```bash
curl https://api.acmebilling.example/v2/customers \
  -H "Authorization: Bearer ak_test_51Hq0example"
```

A request without a valid key returns `401` with the error code `invalid_api_key`.

## Rolling a key without downtime

To replace a key, open it at Settings → Developers → API keys and choose Roll key. A new key is created, and the old key keeps working for 24 hours so you can deploy the new one. After 24 hours the old key stops working and requests that use it return `401`.

## Revoking a key

Choose Revoke on the key. Revocation is immediate and cannot be undone. Do this straight away if you think a key has leaked.

## Permissions

Keys inherit the permissions of the Admin role. Only Owners and Admins can create, roll, or revoke keys.

## Keep keys secret

Never put a live key in browser code, a public repository, or a support message. Acme Billing staff will never ask you to email or share your API key with anyone. If someone asks for it, treat it as a phishing attempt and roll the key.
