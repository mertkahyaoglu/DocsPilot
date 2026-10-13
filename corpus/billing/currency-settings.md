---
title: Currency Settings
category: billing
updated: 2025-01-27
---

# Currency Settings

Acme Billing supports 24 currencies, including USD, EUR, GBP, CAD, AUD, JPY, CHF, and SEK.

## Setting your default currency

Go to Settings → Billing → Currency and choose a currency from the list. New invoices, reports, and payment links use this currency unless you override it.

Changing the default currency does not alter existing invoices. Invoices that have already been created keep the currency they were issued in.

## Per-invoice currency

On Growth and Scale you can choose a different currency for each invoice or customer. On Starter, all invoices are issued in the account's default currency.

To set a currency for one customer, open the customer record and change Currency under Billing details. To set it for a single invoice, change Currency on the invoice form before sending.

## Exchange rates

Acme Billing does not convert your customers' payments. A customer pays in the currency of the invoice, and your payout is made in that currency to a bank account that can hold it.

For reports that combine several currencies, amounts are converted to your default currency using rates refreshed daily at 00:00 UTC. These converted figures are for reporting only.

## Rounding

Amounts are stored in the currency's smallest unit. For currencies with no decimals, such as JPY, invoice totals are rounded to whole units.

## Zero-decimal currencies in the API

When you create invoices through the API, amounts are always given in the smallest currency unit. For USD that is cents, so $25.00 is 2500. For JPY it is whole yen, so ¥2500 is 2500.
