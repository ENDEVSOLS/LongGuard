# EnDevSols Long Suite — PyPI Publishing Standards & Checklist

This document defines the unified release and publishing standards across all repositories in the **EnDevSols Long Suite** (`LongGuard`, `LongCache`, `LongParser`, `LongProbe`, etc.).

Following a uniform workflow name, environment structure, and setup protocol ensures zero friction, predictable CI/CD pipelines, and foolproof PyPI Trusted Publishing configuration.

---

## 1. Unified Workflow Architecture

To eliminate configuration drift and avoid claim mismatches on PyPI, **all Long Suite repositories use the identical workflow file path and naming convention**:

* **Standard File Path:** `.github/workflows/publish.yml`
* **Workflow Name:** `Publish to PyPI`

### Why `publish.yml`?
PyPI Trusted Publishing requires repository owners to declare the exact workflow filename generating the OpenID Connect (OIDC) identity token. By standardizing on `publish.yml` across all tools, maintainers and administrators always register the same filename on PyPI regardless of which tool is being configured.

---

## 2. GitHub Environments Standard

Every Long Suite repository must configure two environments under **Settings → Environments**:

1. **`pypi`**:
   - Used for production package publication to [PyPI](https://pypi.org/).
   - Recommended: Configure **Required reviewers** (e.g., repository maintainers / admin) to prevent accidental releases.
2. **`testpypi`**:
   - Used for pre-release rehearsals and verification on [TestPyPI](https://test.pypi.org/).
   - Open access for release testing.

---

## 3. PyPI Trusted Publishing Configuration Matrix

When registering a GitHub Actions publisher on PyPI or TestPyPI, use the standardized claims:

| Field | TestPyPI (`test.pypi.org`) | Production PyPI (`pypi.org`) |
|---|---|---|
| **Owner** | `ENDEVSOLS` | `ENDEVSOLS` |
| **Repository name** | `<RepositoryName>` (e.g. `LongGuard`) | `<RepositoryName>` (e.g. `LongGuard`) |
| **Workflow name** | `publish.yml` | `publish.yml` |
| **Environment name** | `testpypi` | `pypi` |

> [!IMPORTANT]
> The **Workflow name** field on PyPI must only contain the file name (`publish.yml`), **never** the full directory path (do not enter `.github/workflows/publish.yml`).

---

## 4. Setup Checklist for New Long Suite Repositories

Use this checklist whenever creating a new repository or onboarding a project into the Long Suite:

### Phase A: Repository Setup
- [ ] Add `.github/workflows/publish.yml` using the suite-standard release template.
- [ ] Verify that permissions in `publish.yml` include:
  ```yaml
  permissions:
    id-token: write
    contents: write
  ```
- [ ] In GitHub repo **Settings → Environments**, create:
  - [ ] Environment `pypi`
  - [ ] Environment `testpypi`

### Phase B: PyPI Registration (Complete both at the same time)
- [ ] Log in to **TestPyPI** (`https://test.pypi.org/manage/account/publishing/`):
  - [ ] Add pending or project publisher for `ENDEVSOLS/<RepositoryName>`, workflow `publish.yml`, environment `testpypi`.
- [ ] Log in to **Production PyPI** (`https://pypi.org/manage/account/publishing/`):
  - [ ] Add pending or project publisher for `ENDEVSOLS/<RepositoryName>`, workflow `publish.yml`, environment `pypi`.

### Phase C: Rehearsal & Verification
- [ ] Create a release candidate tag (e.g. `v0.1.0rc1`).
- [ ] Dispatch `Publish to PyPI` with `testpypi: true` and verify successful publication on TestPyPI.
- [ ] Dispatch `Publish to PyPI` with `publish: true` for the final tagged stable release.
