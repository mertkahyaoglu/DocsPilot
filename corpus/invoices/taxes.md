---
title: Tax Rates and Tax IDs
category: invoices
updated: 2024-11-19
---

# Tax Rates and Tax IDs

Acme Billing can add sales tax, VAT, or GST to your invoices. You decide which rates apply; Acme Billing does not file or remit taxes on your behalf.

## Creating a tax rate

Go to Settings → Invoices → Tax rates and select New tax rate. Enter a name (for example "VAT 20%"), the percentage, and whether it is inclusive or exclusive of the price.

## Applying tax rates

On the invoice form, select a tax rate for each line item. You can apply up to 3 tax rates to a single line item, which is useful for regions with state and county taxes. Tax is calculated per line and rounded to two decimal places, rounding half up.

## Tax-inclusive pricing

If a rate is marked inclusive, the unit price you enter already contains the tax. The invoice PDF shows how much of the total is tax. Mixed invoices that contain both inclusive and exclusive rates are supported.

## Your Tax ID

Add your own Tax ID at Settings → Business profile → Tax ID. It is printed on every invoice. You can also store a Tax ID on each customer record so it appears in their billing details.

## Reverse charge

For business customers in another country where reverse charge applies, add a zero-percent tax rate named "Reverse charge" and enable the "Show reverse charge note" option. The invoice then displays the legal notice required in many VAT jurisdictions.

## Tax reports

Go to Reports → Taxes to see the tax collected per rate and per month. You can export the table as CSV. The report is based on the invoice date, not the payment date.

## Changing a rate

Editing a tax rate affects only future invoices. Sent invoices keep the rate they were issued with. If a rate changes legally, create a new rate and archive the old one rather than editing it.
