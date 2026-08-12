# Security Exceptions

Security gates may ignore an advisory only when the exact identifier, dependency
conflict, exposure reasoning, owner, deadline, and removal condition are recorded
here. Broad severity suppression is not an exception process.

## PYSEC-2026-3552 — cryptography

- **Status:** temporary accepted dependency conflict
- **Introduced:** 2026-08-12
- **Review no later than:** 2026-09-12
- **Owner:** replace with the instantiated project's security/service owner
- **Scope:** Python dependency audits in quality and tag-release workflows
- **Suppression:** exact `--ignore-vuln PYSEC-2026-3552` only

The audit reports that this advisory is fixed in cryptography 50. MLflow 3.x,
which provides the experiment/package/serving model format used by this template,
currently constrains cryptography below 50. The template raises cryptography to
the newest currently compatible major range (`>=49,<50`) and does not call
cryptography APIs directly. This reduces exposure but does not declare the
advisory harmless.

Remove the exception as soon as a tested MLflow release accepts cryptography 50,
or replace MLflow if the conflict remains past the review deadline. The owner
must rerun the complete quality gate, `pip-audit` without the exception, package
validation, and serving parity before closing it. An instantiated project must
reassess exposure for its own TLS, certificate, signing, artifact-store, and
untrusted-input paths rather than inheriting this acceptance automatically.
