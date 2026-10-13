---
title: Troubleshooting Login Problems
category: troubleshooting
updated: 2025-05-05
---

# Troubleshooting Login Problems

## I forgot my password

On the sign-in page choose Forgot password and enter your email. The reset link is valid for 60 minutes. If it expires, request a new one. Check your spam folder if nothing arrives after a few minutes.

If you signed up with Google login, you do not have an Acme Billing password. Use the Continue with Google button instead.

## My account is locked

After 5 failed password attempts the account is locked for 15 minutes and shows error code AUTH-1004. Wait for the lock to end, then sign in again. Resetting your password also clears the lock.

## Error codes

| Code | Meaning |
|---|---|
| AUTH-1001 | Wrong email or password |
| AUTH-1004 | Account temporarily locked |
| AUTH-1007 | Two-factor code invalid or expired |

## My two-factor code is not accepted

Codes change every 30 seconds. Make sure the clock on the device running your authenticator is set automatically. If the code is still rejected, use one of your 10 recovery codes instead.

## I lost my authenticator and recovery codes

- **Team member:** ask the account Owner to reset your two-factor setup at Settings → Team → open the member's menu → Reset two-factor.
- **Owner:** open a support ticket. We will verify your identity using your business details, which takes up to 2 business days.

## I am signed out too often

Sessions end after 30 minutes of inactivity and after 12 hours at most. This is expected and cannot be changed.

## The verification email did not arrive

The verification link for new accounts is valid for 24 hours. On the sign-in page choose Resend verification email.
