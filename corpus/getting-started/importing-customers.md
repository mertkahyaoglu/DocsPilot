---
title: Importing Customers from CSV
category: getting-started
updated: 2025-02-10
---

# Importing Customers from CSV

If you already keep your customer list in a spreadsheet, you can bring it into Acme Billing in a few minutes.

## Before you start

Export your list as a CSV file in UTF-8 encoding. Each file can contain up to 5,000 rows and can be at most 5 MB. If your list is bigger, split it into several files and import them one after another.

## Columns

The first row must be a header row. Two columns are required:

- `name`
- `email`

These columns are optional:

- `phone`
- `tax_id`
- `currency` (a three-letter code such as USD, EUR, or GBP)
- `address_line1`
- `city`
- `postal_code`
- `country` (two-letter ISO code, for example US or DE)

Column names are case-insensitive. Unknown columns are ignored.

## Running the import

1. Go to Customers → Import.
2. Select your CSV file and choose Upload.
3. Review the column mapping and fix anything marked in red.
4. Select Start import.

You will see a results summary when the import finishes. Rows that failed validation are listed with a reason and can be downloaded as a new CSV so you can correct and re-upload only those rows.

## Duplicates

Customers are matched by email address. If a row has the same email as an existing customer, the existing customer is updated with the values from the file instead of a duplicate being created. Blank cells never overwrite existing values.

## Currency note

If a row has a `currency` value and your plan supports multi-currency invoicing, that currency becomes the customer's default. On a plan without multi-currency invoicing, the value is ignored and the account default currency is used.
