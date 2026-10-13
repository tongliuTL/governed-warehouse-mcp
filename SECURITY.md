# Security policy

## Reporting a vulnerability

Please report security issues privately. Use GitHub's **Report a vulnerability** button on the
repository's Security tab. If that is not available, open an issue that asks for a private
contact, without including details.

Issues of particular interest:
- a query that reads a model above the caller's tier;
- a query that reads a PII column without `pii:read`;
- a query that writes to the warehouse, or reaches files, the network or settings;
- a token the verifier accepts when it should not;
- an error or search result that reveals a model the caller cannot see.

Each control is listed in [docs/threat-model.md](docs/threat-model.md). A report that names the
control it gets around, and includes a failing test, is the easiest to act on.

## Scope and deployment notes

- **Development credentials.** `gwm keys` and `gwm token` produce development credentials. In
  production, verify tokens against your identity provider's public key and issue short-lived
  tokens.
- **Network exposure.** The HTTP transport binds to `127.0.0.1` by default and has DNS-rebinding
  protection for that host. To expose it on a network, put it behind TLS and your usual gateway.
- **What is out of scope.** Row-level security, inference through aggregation, and network-level
  denial of service. The threat model explains each.
- **The data.** All data in this repository is synthetic.
