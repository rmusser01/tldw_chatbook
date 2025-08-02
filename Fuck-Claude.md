# MediaWindow v2 CSS Fix Plan

## I'M BEING STUPID

The nav panel is small, yes, but the REAL issue is:
- Why is main-content (yellow border) starting SO FAR to the right?
- In a horizontal layout, it should start right after the nav panel
- But there's a MASSIVE gap between them

## THE REAL PROBLEM

Something is pushing main-content way over to the right. This could be:
1. A margin-left on main-content
2. Some other invisible element between them
3. A layout calculation bug

The gap is between nav panel and main-content, not a width issue!