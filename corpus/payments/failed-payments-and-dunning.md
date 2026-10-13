---
title: Failed Payments and Dunning
category: payments
updated: 2025-07-01
---

# Failed Payments and Dunning

Dunning is the process of retrying a failed charge and notifying the customer. It applies to subscriptions and to invoices with automatic collection turned on.

## Retry schedule

When a charge fails, Acme Billing retries it automatically on days 1, 3, 5, and 7 after the first failure. Retries run at 09:00 UTC.

On Growth and Scale you can change the schedule at Settings → Payments → Retry rules. Starter uses the fixed schedule above.

## Subscription status during dunning

| Moment | Status |
|---|---|
| First charge fails | past_due |
| All 4 retries fail | unpaid |
| Day 21 after the first failure | canceled |

A subscription is canceled automatically on day 21 after the first failed charge if it has not been paid by then. If the customer pays at any point, the status returns to active.

## Events

Each failed attempt sends a `payment.failed` webhook event. The first failure also sends `subscription.past_due`, and automatic cancellation sends `subscription.canceled`. See the Webhooks page for the payload format.

## Customer emails

The customer is emailed after each failed attempt with a secure link to update their payment method. The link is valid for 7 days after each email.

## Decline codes

| Code | Meaning | Suggested action |
|---|---|---|
| card_declined | The bank refused the charge. | Customer should contact the bank or try another card. |
| insufficient_funds | Not enough balance. | A later retry may succeed. |
| expired_card | Card expiry date has passed. | Customer must update the card. |
| authentication_required | 3-D Secure verification is needed. | Customer must approve the payment from the email link. |
| processing_error | Temporary problem. | Retry; no customer action needed. |

## Stopping dunning

Open the subscription or invoice and choose Stop retries. The status stays as it was, and you can record an offline payment by choosing Mark as paid.
