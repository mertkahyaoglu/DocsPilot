---
title: Account Security and Sign-in
category: security
updated: 2025-05-05
---

# Account Security and Sign-in

## Signing in

You can sign in with your email address and password, or with Google login if you created your account that way. An account created with Google login does not have an Acme Billing password.

## Password rules

Passwords must be at least 12 characters long. After 5 failed sign-in attempts, the account is locked for 15 minutes. You will see error code AUTH-1004 while the lock is active, and the lock ends on its own.

## Two-factor authentication

### Turning it on

Go to Settings → Security → Two-factor authentication and choose Set up. Scan the QR code with any TOTP authenticator and enter the 6-digit code to confirm.

### Recovery codes

When you finish setup you receive 10 one-time recovery codes. Store them safely. Each code can be used once if you lose access to your authenticator.

### Requiring it for everyone

The Owner can require two-factor authentication for all team members at Settings → Security → Require two-factor. Members without it set up are asked to do so at their next sign-in.

## Sessions

A session ends after 30 minutes of inactivity, and after 12 hours at the longest, even if you stay active. Settings → Security → Sessions lists your active sessions and lets you sign out of all of them.

## Audit log

Settings → Security → Audit log records sign-ins, plan changes, API key changes, bank account changes, and team changes. Entries are kept for 400 days and can be exported as CSV by Owners and Admins.

## Notifications

You receive an email whenever a sign-in happens from a new device, when your password changes, and when a payout bank account is edited.

## Reporting a problem

If you suspect your account has been accessed by someone else, change your password, sign out of all sessions, and roll your API keys. Then open a support ticket via Help → Contact support.
