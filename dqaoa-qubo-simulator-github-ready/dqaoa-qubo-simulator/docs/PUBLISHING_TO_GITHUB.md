# Publishing this repository to GitHub

Recommended repository name: `dqaoa-qubo-simulator`.

## Before the first push

1. Confirm the authorship and MIT licensing language with the authors and LSU/RAISE LAB.
2. Confirm that no confidential data, credentials, licensed CPLEX files, unpublished reviewer correspondence, or third-party material is present.
3. Run the tests and the quick start.
4. Replace the repository URL in `pyproject.toml` and `CITATION.cff` if a different name is chosen.

## Command-line route

Create an empty public repository under the `LSU-RAISE-LAB` organization. Do not initialize it with a README, license, or `.gitignore`, because those files already exist locally.

Then run from the project directory:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial public release of the DQAOA-QUBO simulator"
git remote add origin https://github.com/LSU-RAISE-LAB/dqaoa-qubo-simulator.git
git push -u origin main
```

If GitHub CLI is installed and authenticated:

```bash
gh repo create LSU-RAISE-LAB/dqaoa-qubo-simulator \
  --public \
  --source=. \
  --remote=origin \
  --push
```

## After the push

- Add repository description and topics.
- Enable Issues and Discussions if desired.
- Add branch protection for `main` after the first release.
- Create a `v1.0.0` release and tag.
- Connect the GitHub repository to Zenodo if a permanent software DOI is desired.
- Update `CITATION.cff` and the README when the associated paper receives its final citation and DOI.
