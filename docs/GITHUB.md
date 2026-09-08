# Preparing a GitHub repository

The code is organized for upload from this folder. A local Git repository is initialized on `main`; no remote, commit or push was created during assembly.

Before public release, the authors should choose the code license and add any permanent data/checkpoint link that actually exists. This repository intentionally contains neither an invented license grant nor a placeholder public data DOI. The citation file contains the author/title information supplied with the accepted manuscript.

To inspect what will be tracked:

```bash
git status --short
python scripts/check_repo.py
```

The `.gitignore` excludes environments, generated `outputs/`, raw and processed datasets, model checkpoint formats, local credentials and cache files. The two small plotting NPZ files are explicitly allowed so that figure reproduction remains self-contained. The CSV/JSON plot evidence should be committed along with the code.

After creating the empty repository in your own GitHub account, add its actual URL as the `origin` remote, make the initial commit and push `main`. No account or destination is assumed in this folder.

If distributing a ZIP instead, use the companion `regshift-deeponet.zip`; it excludes local Git metadata and generated outputs.
