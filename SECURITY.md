# Security Policy

s2pro-native includes a hand-written HTTP parser and service, a scheduler,
native CUDA kernels, a safetensors loader, and an FP8 GEMM integration. A
security issue in any of these layers can affect host availability, GPU
memory, or request privacy.

## Supported versions

Security fixes are developed against the latest `main` commit.

| Version | Security support |
| --- | --- |
| `main` | Supported for current development and coordinated fixes. |
| Older commits and branches | Not supported. |

The project has no published release line yet; until one exists, deploy from
a pinned `main` commit you have reviewed.

## Report a vulnerability privately

Do not open a public issue, pull request, or discussion for a suspected
vulnerability. Use one of these private channels:

1. Submit a private vulnerability report through
   [GitHub Security Advisories](https://github.com/luka-loehr/s2pro-native/security/advisories/new).
2. Email [luka@lukaloehr.com](mailto:luka@lukaloehr.com) with the subject
   `SECURITY s2pro-native`.

Include the commit hash, a reproduction path, and the impact you believe the
issue has. You will receive an acknowledgement within seven days.

## Deployment boundary

The server binds loopback by default, does not terminate TLS, and provides no
identity provider beyond an optional static bearer token. Public deployments
must place it behind an authenticated, rate-limited proxy with request and
response timeouts, and must treat prompt text as sensitive payload throughout
the surrounding stack. The service itself does not write prompt text or
generated audio to its logs.
