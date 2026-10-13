---
title: Payment Reminders and Late Fees
category: invoices
updated: 2025-03-03
---

# Payment Reminders and Late Fees

Acme Billing can nudge customers to pay on time and, on higher plans, add a late fee when they do not.

## Default reminder schedule

Reminders are available on every plan and are on by default. For each unpaid invoice, the customer receives an email:

- 3 days before the due date
- On the due date
- 7 days after the due date
- 14 days after the due date

Reminders stop as soon as the invoice is marked Paid or Void.

## Customizing reminders

Go to Settings → Invoices → Reminders. You can switch individual reminders on or off, change the number of days, and edit the email text. Placeholders such as `{{customer_name}}` and `{{amount_due}}` are available in the template.

## Sending a manual reminder

Open an unpaid invoice and choose Send reminder. Manual reminders are limited to one every 24 hours per invoice.

## Late fees

Automatic late fees are available on Growth and Scale only. To set one up, go to Settings → Invoices → Late fees.

You can choose either:

- a flat amount, or
- a percentage of the invoice total, up to a maximum of 5%.

The fee is added once, 7 days after the due date by default. You can change this delay to any value between 1 and 30 days. The fee appears on the invoice as a separate line item, and the customer is notified by email.

## Waiving a fee

Open the invoice and choose Remove late fee. The fee line is removed and the invoice total is updated. A waived fee is not applied again.

## Reminders and customer communication

Reminder emails are sent from noreply@mail.acmebilling.example. If a customer replies, the message goes to the email address in your Business profile, not to Acme Billing.
