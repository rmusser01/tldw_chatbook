# Git Feature Branch Splitting Guide

This guide documents the process of splitting a large pull request (PR) into smaller, more manageable PRs.

## Why Split PRs?

- **Easier reviews**: Smaller PRs are easier for reviewers to understand and review thoroughly
- **Faster merges**: Smaller changes are less likely to have conflicts and can be merged faster
- **Better git history**: Each PR represents a single logical change
- **Reduced risk**: If one feature has issues, it doesn't block other features

## Prerequisites

Before splitting a PR, ensure you have:
- A backup of your current branch
- A clear understanding of which changes belong together
- Knowledge of dependencies between features

## Method 1: Cherry-Pick Approach (Recommended)

This method is best when you have clean, well-organized commits.

### Step 1: Create a Backup
```bash
git checkout your-feature-branch
git branch backup-your-feature-branch
```

### Step 2: Identify Commits
```bash
# View commit history with one-line format
git log --oneline main..your-feature-branch

# Or view with more detail
git log --stat main..your-feature-branch
```

### Step 3: Create New Feature Branches
```bash
# Create branches from your base branch (main/dev)
git checkout main
git checkout -b feature-1-descriptive-name
git checkout -b feature-2-descriptive-name
git checkout -b feature-3-descriptive-name
```

### Step 4: Cherry-Pick Commits
```bash
# Switch to first feature branch
git checkout feature-1-descriptive-name

# Cherry-pick specific commits
git cherry-pick abc1234
git cherry-pick def5678

# Or cherry-pick a range
git cherry-pick abc1234..def5678
```

### Step 5: Clean Up Commits (Optional)
```bash
# Squash related commits together
git rebase -i main

# In the editor, mark commits to squash with 's'
# Keep the first commit as 'pick', change others to 'squash'
```

## Method 2: Patch File Approach

Best for extracting specific file changes regardless of commit structure.

### Step 1: Generate Patches
```bash
git checkout your-feature-branch

# Create patch for specific files
git format-patch main --stdout -- path/to/file1.py path/to/file2.py > feature1.patch

# Or create patch for specific commits
git format-patch -1 abc1234 --stdout > specific-commit.patch
```

### Step 2: Apply Patches
```bash
# Create new branch
git checkout -b feature-1-descriptive-name main

# Apply the patch
git apply feature1.patch

# Stage and commit
git add .
git commit -m "Feature 1: Clear description of changes"
```

## Method 3: File Checkout Approach

Best when you want to extract the final state of specific files.

### Step 1: Create New Branch
```bash
git checkout -b feature-1-descriptive-name main
```

### Step 2: Checkout Specific Files
```bash
# Get specific files from your original branch
git checkout your-feature-branch -- path/to/file1.py
git checkout your-feature-branch -- path/to/file2.py
git checkout your-feature-branch -- path/to/directory/
```

### Step 3: Commit Changes
```bash
git add .
git commit -m "Feature 1: Description of changes"
```

## Method 4: Interactive Rebase Split

Best for splitting commits that contain multiple logical changes.

### Step 1: Start Interactive Rebase
```bash
git checkout your-feature-branch
git rebase -i main
```

### Step 2: Mark Commits to Edit
```
# Change 'pick' to 'edit' for commits you want to split
edit abc1234 Large commit with multiple features
pick def5678 Another commit
```

### Step 3: Split the Commit
```bash
# When rebase stops at the commit
git reset HEAD~

# Now files are unstaged, add them selectively
git add path/to/feature1/files
git commit -m "Feature 1: Description"

git add path/to/feature2/files
git commit -m "Feature 2: Description"

# Continue rebase
git rebase --continue
```

## Best Practices

### 1. Logical Grouping
Group changes by:
- **Feature**: All files related to a specific feature
- **Layer**: Separate UI, business logic, and database changes
- **Type**: Bug fixes separate from features
- **Dependencies**: Keep dependent changes together

### 2. Commit Organization
```bash
# Example grouping for a web application
PR 1: Authentication Feature
- auth/login.py
- auth/logout.py
- templates/login.html
- tests/test_auth.py

PR 2: User Profile Feature
- models/user_profile.py
- views/profile.py
- templates/profile.html
- tests/test_profile.py

PR 3: UI Improvements
- static/css/styles.css
- templates/base.html
- static/js/ui.js

PR 4: Bug Fixes
- Various small fixes across multiple files
```

### 3. Dependency Management
- Merge PRs in dependency order
- Note dependencies in PR descriptions
- Use draft PRs for dependent features

### 4. PR Description Template
```markdown
## Description
Brief description of what this PR does

## Related PRs
- Depends on: #123
- Related to: #124, #125

## Changes
- Added feature X
- Modified component Y
- Fixed bug Z

## Testing
- [ ] Unit tests pass
- [ ] Manual testing completed
- [ ] No regressions identified
```

## Handling the Original PR

### 1. Close with Explanation
```markdown
This PR has been split into smaller PRs for easier review:

- #101: Feature A - Authentication improvements
- #102: Feature B - Profile page updates  
- #103: Bug fixes - Various small fixes
- #104: UI updates - CSS and layout improvements

Closing this PR in favor of the smaller ones listed above.
```

### 2. Update Branch Protection
If the original PR was set up with branch protection or CI/CD:
- Transfer any required reviews
- Ensure new PRs pass all checks
- Update any documentation references

## Common Scenarios

### Scenario 1: Mixed Features and Fixes
```bash
# Original branch has features and bug fixes mixed
git checkout -b bugfixes-only main
git checkout original-branch -- $(git diff --name-only main original-branch | grep -E "(fix|patch)")
git commit -m "Bug fixes: Various fixes extracted from feature branch"

git checkout -b new-feature-only main
git checkout original-branch -- $(git diff --name-only main original-branch | grep -v -E "(fix|patch)")
git commit -m "Feature: New feature without bug fixes"
```

### Scenario 2: Frontend and Backend Changes
```bash
# Split frontend and backend
git checkout -b frontend-changes main
git checkout original-branch -- static/ templates/ *.css *.js *.html
git commit -m "Frontend: UI updates and improvements"

git checkout -b backend-changes main
git checkout original-branch -- *.py models/ views/ api/
git commit -m "Backend: API and business logic updates"
```

### Scenario 3: Database Migrations
```bash
# Always separate database migrations
git checkout -b db-migrations main
git checkout original-branch -- migrations/ alembic/ *.sql
git commit -m "Database: Schema updates and migrations"
```

## Troubleshooting

### Merge Conflicts During Cherry-Pick
```bash
# If cherry-pick has conflicts
git cherry-pick --abort  # To cancel
# Or
git add .               # After resolving conflicts
git cherry-pick --continue
```

### Finding Which Commits Touch Which Files
```bash
# List commits that touched specific files
git log --oneline main..your-feature-branch -- path/to/file.py

# Show which files were changed in each commit
git log --stat --oneline main..your-feature-branch
```

### Undoing Mistakes
```bash
# If you mess up, you can always reset to your backup
git checkout -B feature-1-descriptive-name backup-your-feature-branch
```

## Final Checklist

Before submitting the new PRs:

- [ ] All changes from original PR are included in new PRs
- [ ] Each PR has a clear, single purpose
- [ ] Dependencies are documented
- [ ] Commit messages are clear and descriptive
- [ ] Tests pass for each PR independently
- [ ] Original PR is closed with links to new PRs
- [ ] Reviewers are notified of the change

## Example Workflow

```bash
# 1. Backup original
git checkout feature-everything
git branch backup-feature-everything

# 2. Create API changes PR
git checkout -b feature-api-endpoints main
git cherry-pick abc1234 def5678  # API-related commits
git push origin feature-api-endpoints

# 3. Create UI changes PR  
git checkout -b feature-ui-updates main
git cherry-pick ghi9012 jkl3456  # UI-related commits
git push origin feature-ui-updates

# 4. Create test updates PR
git checkout -b feature-test-coverage main
git cherry-pick mno7890          # Test-related commits
git push origin feature-test-coverage

# 5. Close original PR with explanation and links
```

Remember: The goal is to make life easier for reviewers while maintaining a clean, understandable git history.