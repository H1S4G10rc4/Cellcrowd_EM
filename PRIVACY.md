# Publication boundary

This repository contains reviewed source code and mathematical documentation only.
It deliberately excludes all participant inputs and results, including records
that use aliases, numeric identifiers or hashed names.

Do not commit:

- Participant names, account names, email addresses, IDs or lookup tables.
- Raw annotations, microscopy images or identifying filenames.
- Fitted annotator tables, saved predictions or source-data manifests.
- Notebooks, notebook outputs, logs, generated figures or compiled documents.
- Archives, local backups, font binaries, credentials or machine-specific paths.

The model accepts caller-supplied annotator identifiers and preserves them in local
outputs. It is not an anonymization system. Keep private inputs outside the clone
where practical, and treat derived results as private too. A hash of a name can
still be linked to that person.

## Safeguards

`PUBLIC_FILES.txt` is an explicit publication allowlist. `.gitignore` ignores
everything else by default. The publication checker rejects unlisted files,
data-like paths, non-text files, links, unusually large files and common credential
patterns in the actual staged or committed contents. It also rejects personal
home-directory paths and email addresses. The license attribution is intentional
and is not participant data.

Install the local pre-commit and pre-push hooks with `make install-hooks`. CI runs
the checker and tests on pushes and pull requests. Before a release, separately
compare the staged content with private identifiers using a local audit. Never
upload that identifier list or print matched values into public logs. Automated
checks cannot reliably recognize every personal name or indirect identifier.

## Historical exposure

Replacing a branch with a clean history does not remove previously downloaded
clones, cached commit pages, or all unreachable objects held by the hosting
service. Removing previously published sensitive data can require assistance
from GitHub Support. Do not attach the private backup or identifier list to a
public issue. See GitHub's guidance on
[removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
