---
title: Troubleshooting Invoices That Were Not Delivered
category: troubleshooting
updated: 2025-04-14
---

# Troubleshooting Invoices That Were Not Delivered

Use this procedure when a customer says they did not receive an invoice email.

## Procedure

1. **Check the status.** Open the invoice. If the status is Draft, it has not been sent. Select Send. If it shows Sent or Viewed, continue.
2. **Check the email address.** Open the customer record and look at the email field for typos. Correct it and resend if needed.
3. **Open the activity log.** On the invoice, open the Activity tab. Look for the delivery event. A `bounced` entry shows the reason, such as a full mailbox or an address that does not exist.
4. **Ask the customer to check spam.** The invoice is sent from noreply@mail.acmebilling.example. Ask the customer to look in their spam or promotions folder and to allow that sender.
5. **Resend the invoice.** Choose Send reminder or Resend on the invoice. A resend creates a new delivery event.
6. **Share the link.** If emails still do not arrive, copy the hosted invoice link from the Share menu and send it through another channel.
7. **Open a ticket.** If the activity log shows `delivered` but the customer still cannot find the email, contact support with the invoice number.

## Common bounce reasons

| Reason | What it means |
|---|---|
| mailbox_full | The customer's inbox is full. Try again later. |
| no_such_user | The address does not exist. Fix the email address. |
| blocked | The customer's mail server rejected the message. Ask them to allow the sender. |

## Hard bounces

After a hard bounce, the address is marked undeliverable and Acme Billing stops sending to it. Updating the customer's email address clears the flag.

## Delivery history

Email delivery logs are kept for 90 days. Entries older than that are no longer visible in the Activity tab.
