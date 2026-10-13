---
title: Creating and Sending Invoices
category: invoices
updated: 2025-07-15
---

# Creating and Sending Invoices

This page explains everything about creating an invoice in Acme Billing, from the first draft to voiding a mistake.

## Creating a new invoice

### Starting a draft

Go to Invoices → New invoice. Choose an existing customer or select Add customer to create one on the spot. The invoice starts in the Draft status and nothing is sent until you choose to send it.

### Line items

Each invoice can contain up to 100 line items. A line item has a description, quantity, unit price, and optional tax rates. Frequently used items can be saved at Settings → Invoices → Products so you can pick them from a list.

### Invoice numbers

Invoice numbers are generated automatically using a prefix and a counter, for example INV-0001. Change the prefix or the starting number at Settings → Invoices → Numbering. Numbers are assigned when the invoice is first sent, not when the draft is created.

### Payment terms

Choose one of: Due on receipt, Net 7, Net 15, Net 30, Net 45, or Net 60. The default is Net 30. You can change the account default at Settings → Invoices → Defaults.

## Invoice statuses

An invoice moves through these statuses:

| Status | Meaning |
|---|---|
| Draft | Created but not sent. Can be edited freely and does not count toward your invoice limit. |
| Sent | Delivered to the customer by email. |
| Viewed | The customer opened the hosted invoice page. |
| Paid | Full payment has been received. |
| Overdue | The due date has passed and the invoice is not paid. |
| Void | Cancelled. No payment can be taken. |

An invoice becomes Overdue at 00:00 in your account time zone on the day after its due date.

## Sending

### By email

Select Send on a draft. The customer receives an email from noreply@mail.acmebilling.example containing a PDF and a link to the hosted invoice page. You can add up to 5 additional recipients in the CC field.

### Sharing a link

Every sent invoice has a hosted page that stays valid indefinitely. Copy the link from the invoice's Share menu to send it through your own channel.

## Editing after sending

An invoice that has been sent but not paid can be edited and re-sent. The customer is notified of the change. A Paid invoice cannot be edited. To correct one, issue a credit note: open the invoice and choose Issue credit note. Credit notes can be for the full or a partial amount.

## Voiding an invoice

Open the invoice and choose Void, or call `POST /v2/invoices/{id}/void` through the API. Only unpaid invoices can be voided. A voided invoice number is never reused, so there will be a gap in your numbering sequence. Voided invoices still count toward your monthly invoice limit.

## Branding

Add your logo and choose an accent colour at Settings → Invoices → Branding. The logo should be a PNG or JPG file up to 2 MB. Branding applies to the PDF, the email, and the hosted page.

## Duplicating an invoice

Open any invoice and choose Duplicate to create a new draft with the same customer and line items. The new draft gets a fresh number when you send it.

## Bulk actions

On the invoice list, tick several invoices to send, remind, or download them as a single ZIP of PDFs. You can select up to 100 invoices at a time.
