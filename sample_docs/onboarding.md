# New Engineer Onboarding

Welcome to the platform team. This doc covers the basics you need for your
first week.

## VPN access

You need VPN access before you can reach any internal service. File a
request in the access-requests channel with your manager as approver.
Access is usually granted within one business day. Once approved, install
the VPN client from the internal software portal and connect using your
company SSO credentials.

## Local environment setup

Clone the monorepo, run `make bootstrap`, and it will install the toolchain
pinned in `.tool-versions`. If bootstrap fails on macOS, it is almost always
a stale Homebrew cache. Run `brew update` first.

## On-call

You are not on the on-call rotation during your first month. After that,
you will be added as a secondary responder, shadowing a primary for two
full rotations before taking a primary shift yourself.
