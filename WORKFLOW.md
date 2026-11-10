# Daily Workflow Guide

This document captures how to work safely across both branches of this repo
and how to recover if things go wrong.

---

## Branch Strategy

| Branch | Owner | Purpose | Push? |
|--------|-------|---------|-------|
| `main`            | Team       | Synced with `origin/main`; production-ready | ✅ |
| `roy-local-work`  | Roy (you)  | Your WIP, notebook edits, the camera-ready PDF | ❌ (local-only) |

---

## Daily Routine (recommended)

1. **Start every session here:**
   ```bash
   cd "C:\College-Work\Projects\Explainable AI for Enhanced Network Intrusion Detection in Resource-Constrained Environments"
   git checkout roy-local-work          # Work on your private branch
   git fetch origin                    # See what the team has done
   git log --oneline origin/main -10   # Review their changes
   ```

2. **Edit notebooks freely.** Nothing leaves your machine unless you explicitly push.

3. **When you're ready to share:**
   ```bash
   git checkout main
   git merge roy-local-work            # bring your changes into main
   git push origin main                # share with the team
   ```

4. **Pull team changes:**
   ```bash
   git checkout roy-local-work
   git rebase main                     # put your WIP on top of the team's latest
   ```

---

## If Git Goes Wrong

### "I have unstaged changes and need to do something"
```bash
git stash push -u -m "checkpoint: what I was doing"
# ... do risky thing ...
git stash pop
```

### "I'm on the wrong branch"
```bash
git stash push -u -m "moving-to-correct-branch"
git checkout correct-branch
git stash pop
```

### "I need to see what files changed locally but I don't want to lose them"
```bash
git status --short
git diff > my-changes.patch
# later:
git apply my-changes.patch
```

### "Something is broken and I want a clean slate without losing my work"
```bash
git checkout -b backup-of-current      # save current state under a throwaway branch name
git checkout main
git reset --hard origin/main          # align perfectly with remote
git checkout -b roy-local-work         # recreate your private branch
# your `backup-of-current` branch still has everything if you need it
```

---

## Recovery Locations

- **Stash**: `git stash list` (lifesaver for half-finished work)
- **Reflog**: `git reflog` (every commit HEAD has pointed at for ~30 days)
- **Branches**: `git branch` (any branch can be checked out, even if it looks dead)
- **Backup branch**: `backup-of-current` (only exists if you ran the recovery above)
