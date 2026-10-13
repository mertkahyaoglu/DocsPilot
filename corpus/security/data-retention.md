---
title: Data Retention and Encryption
category: security
updated: 2025-06-30
---

# Data Retention and Encryption

This page lists how long Acme Billing keeps each type of data and how it is protected.

## Retention periods

| Data | Retention |
|---|---|
| Financial records (invoice amounts, dates, tax details, payments) | 7 years |
| Customer personal data (names, emails, addresses) | While your account is active; deleted 90 days after cancellation |
| Audit log | 400 days |
| API request logs | 30 days |
| Webhook delivery logs | 14 days |
| Email delivery logs | 90 days |
| Backups | 35 days |
| Export download links | 24 hours |

## Financial records

Tax law in most countries requires invoicing records to be kept for several years. Acme Billing retains invoice and payment financial records for 7 years. After your account is canceled and the 90-day read-only period ends, these records are anonymised: customer names, emails, and addresses are removed, and only amounts, dates, and tax figures remain.

## Deletion requests

### Requesting deletion

To have your personal data removed without waiting for the 90 days, go to Settings → Data → Request deletion. The Owner must confirm by email.

### Timing

Deletion requests are completed within 30 days. Backups that still contain the data are overwritten within the 35-day backup window.

## Exporting your data

Go to Settings → Data → Export to download invoices, customers, and payments in CSV or JSON. The download link is emailed to the Owner and expires after 24 hours.

## Encryption

Data is encrypted at rest with AES-256. All traffic to the app and the API uses TLS 1.2 or higher, and older protocol versions are refused.

## Data location

When you create your account you choose a data region: United States or European Union. Your data is stored in that region, and the region cannot be changed after sign-up. Backups stay in the same region.

## Access by Acme staff

Support staff can see your account data only while handling a ticket you opened, and each access is written to an internal log.

## Payment details

Full card numbers are never stored on Acme Billing servers. Card data is handled by our payment processing partner, and we keep only the last four digits, expiry date, and card brand.
