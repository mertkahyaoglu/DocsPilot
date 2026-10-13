---
title: Refunding Your Customers' Payments
category: payments
updated: 2025-05-30
---

# Refunding Your Customers' Payments

This page is about refunds you issue to your own customers for payments they made through Acme Billing. For refunds of your Acme Billing subscription fees, see the Refund Policy.

## Issuing a refund

1. Go to Payments and open the payment.
2. Select Refund.
3. Choose Full or Partial and enter the amount.
4. Add an internal note, then select Confirm refund.

Only Owners, Admins, and Accountants can issue refunds.

## Time limit

Payments can be refunded up to 90 days after the original payment date. After that, you need to pay the customer outside Acme Billing and record it with a credit note.

## Partial refunds

You can issue several partial refunds against one payment as long as the total does not exceed the amount paid.

## Fees

Processing fees on the original payment are not returned when you refund. The dispute fee does not apply to refunds, so refunding a customer is cheaper than losing a dispute.

## When the customer receives the money

Refunds appear on the customer's statement within 5 to 10 business days, depending on their bank. ACH refunds can take up to 10 business days.

## Effect on the invoice

A full refund returns the invoice to the Overdue or Sent status, depending on the due date, so it can be paid again. If you do not want that, issue a credit note to close the invoice instead. A partial refund reduces the amount shown as paid.

## Refund through the API

Send `POST /v2/payments/{id}/refund` with an optional `amount` in the smallest currency unit. Omitting `amount` refunds the full payment.
