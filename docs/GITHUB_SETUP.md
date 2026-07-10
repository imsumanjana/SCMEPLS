# GitHub Setup and History View

Run these commands from the project root.

```powershell
git init
git branch -M main
git remote add origin https://github.com/imsumanjana/SCMEPLS.git
git add .
git commit -m "Update plot typography controls"
git push -u origin main
```

If the remote already exists:

```powershell
git remote -v
git remote set-url origin https://github.com/imsumanjana/SCMEPLS.git
git push -u origin main
```

If VS Code does not show history or graph:

1. Confirm the folder opened in VS Code is the repository root.
2. Check that at least one commit exists: `git log --oneline --graph --all`.
3. Install the **Git Graph** extension or use VS Code **Source Control Graph** if available.
4. Fetch remote refs: `git fetch --all --prune`.
5. Confirm branch tracking: `git branch -vv`.
6. If needed, set upstream: `git branch --set-upstream-to=origin/main main`.

The graph/history appears only after commits exist. Adding a remote alone does not create graph history.
