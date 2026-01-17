# Development Setup Guide

This guide covers setting up and working with the Sharpfin dependency as a git submodule.

## Initial Setup

### For New Developers

When cloning this repository for the first time:

```bash
# Clone with submodules included
git clone --recursive https://github.com/tarcon/ComfyUI-Sharpfin.git
cd ComfyUI-Sharpfin

# Install dependencies
pip install -r requirements.txt
```

### If You Already Cloned Without `--recursive`

```bash
# Initialize and fetch the submodule
git submodule update --init --recursive
```

## Working with the Sharpfin Submodule

### Checking Submodule Status

```bash
# See which commit the submodule is currently on
git submodule status

# View submodule details
git config --file .gitmodules --list
```

### Updating to Latest Sharpfin Version

```bash
# Navigate to submodule directory
cd lib

# Fetch latest changes
git fetch origin

# Checkout the latest main branch
git checkout origin/main

# Return to parent repo
cd ..

# Commit the submodule update
git add lib
git commit -m "Update Sharpfin to latest version"
git push
```

**Alternative one-liner:**
```bash
git submodule update --remote --merge
git add lib
git commit -m "Update Sharpfin submodule"
```

### Pinning to a Specific Version

For stability, you may want to pin to a specific commit or tag:

```bash
cd lib
git checkout <commit-hash-or-tag>
cd ..
git add lib
git commit -m "Pin Sharpfin to version X.Y.Z"
```

## Common Issues and Solutions

### Issue: Submodule directory is empty

**Solution:**
```bash
git submodule update --init --recursive
```

### Issue: Changes made inside submodule directory

**Important:** Don't make changes directly in `lib/`. Instead:

1. Fork the Sharpfin repository
2. Make changes in your fork
3. Submit a pull request to the original repo
4. Update submodule to your fork temporarily if needed:

```bash
# Edit .gitmodules to point to your fork
vim .gitmodules

# Update submodule URL
git submodule sync
git submodule update --remote
```

### Issue: Merge conflicts with submodule

```bash
# Accept their version
git checkout --theirs lib
git add lib

# Or accept your version
git checkout --ours lib
git add lib
```

## Pull Request Guidelines

When submitting a PR that updates the Sharpfin submodule:

1. **Document the reason** for the update in the PR description
2. **List any breaking changes** from the Sharpfin update
3. **Test thoroughly** to ensure compatibility
4. **Link to the Sharpfin commits** you're updating to

Example PR description:
```
Updates Sharpfin submodule to include the new sparse GPU implementation.

Changes:
- Sharpfin updated from commit abc123 to def456
- Adds ~7x performance improvement for GPU resizing
- No breaking changes to our implementation

Sharpfin changes: https://github.com/drhead/Sharpfin/compare/abc123...def456
```

## Testing Submodule Changes Locally

Before committing a submodule update:

```bash
# Update submodule
cd lib
git pull origin main
cd ..

# Run your test suite
pytest tests/

# Test with ComfyUI
python -m comfy.main --cpu  # or your test command

# If tests pass, commit the update
git add lib
git commit -m "Update Sharpfin: [reason]"
```

## CI/CD Considerations

Ensure your CI/CD pipeline handles submodules:

**GitHub Actions example:**
```yaml
- name: Checkout code
  uses: actions/checkout@v4
  with:
    submodules: 'recursive'
```

**GitLab CI example:**
```yaml
variables:
  GIT_SUBMODULE_STRATEGY: recursive
```

## Removing the Submodule (if needed)

If you ever need to remove the submodule:

```bash
# Remove submodule entry from .git/config
git submodule deinit -f lib

# Remove submodule directory from .git/modules
rm -rf .git/modules/lib

# Remove submodule from working tree
git rm -f lib

# Commit the removal
git commit -m "Remove Sharpfin submodule"
```

## Resources

- [Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Sharpfin Repository](https://github.com/drhead/Sharpfin)
- [Submodule Cheat Sheet](https://gist.github.com/gitaarik/8735255)