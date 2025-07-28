# Textual Layouts and CSS Guide

## Overview
Textual is a Python framework for creating Terminal User Interfaces (TUIs). Unlike web CSS, Textual has its own CSS dialect with specific properties and behaviors designed for terminal environments.

## Layout Types in Textual

### 1. Vertical Layout (default)
```css
.container {
    layout: vertical;
}
```
- Children stack top to bottom
- Children expand to full width by default
- Height is determined by content unless specified

### 2. Horizontal Layout
```css
.container {
    layout: horizontal;
}
```
- Children arrange left to right
- Children expand to full height by default
- Width is determined by content unless specified

### 3. Grid Layout
```css
.container {
    layout: grid;
    grid-size: 3 2;        /* 3 columns, 2 rows */
    grid-columns: 1fr 2fr 1fr;  /* Column widths */
    grid-rows: auto 1fr;        /* Row heights */
    grid-gutter: 1 2;          /* Row gap, column gap */
}
```
- Children placed in cells automatically, left-to-right, top-to-bottom
- NO `grid-column` or `grid-row` properties for positioning
- Use `column-span` and `row-span` for spanning cells
- Empty cells remain empty

### 4. Dock Layout
```css
.widget {
    dock: top;    /* or bottom, left, right */
}
```
- Removes widget from normal flow
- Docked widgets take priority over layout
- Multiple widgets can be docked to same edge (stack in order)
- Remaining space goes to non-docked widgets

## Size Units

### Fractional Units (fr)
- `1fr` = one fraction of available space
- `2fr` = twice as much space as `1fr`
- Only works in grid-columns/grid-rows or width/height

### Percentage (%)
- Relative to parent container
- `width: 50%` = half of parent width

### Cell Units
- `width: 20` = 20 character cells
- `height: 10` = 10 lines

### Auto
- Size based on content
- Minimum size needed to fit content

## Common Patterns

### Sidebar Layout
```css
/* Using Horizontal */
.main-container {
    layout: horizontal;
}
.sidebar {
    dock: left;
    width: 25%;
    min-width: 20;
}
.content {
    width: 1fr;
}

/* Using Grid */
.main-container {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 25% 1fr;
}
/* Children automatically placed in cells */
```

### Collapsible Sidebar
```css
.sidebar {
    width: 25%;
}
.sidebar.collapsed {
    display: none;
}
/* OR */
.container.sidebar-collapsed {
    grid-columns: 0 1fr;
}
```

### Header/Footer/Content
```css
.screen {
    layout: vertical;
}
.header {
    dock: top;
    height: 3;
}
.footer {
    dock: bottom;
    height: 3;
}
.content {
    height: 1fr;
}
```

## Important Differences from Web CSS

### No Positioning Properties
- No `position: absolute/relative/fixed`
- No `grid-column` or `grid-row` for placement
- No `float`
- No `flex` or `flexbox`

### Layout Behavior
- Widgets expand to fill available space by default
- No margin collapse
- No z-index (except special cases)
- Overflow is handled differently

### Sizing
- No `box-sizing` property
- Padding is inside widget bounds
- Border takes additional space

## Best Practices

### 1. Choose the Right Layout
- **Vertical**: For stacked components (forms, lists)
- **Horizontal**: For side-by-side components
- **Grid**: For complex layouts with alignment needs
- **Dock**: For fixed UI elements (headers, sidebars)

### 2. Size Appropriately
- Use `fr` units for flexible sizing
- Set `min-width`/`max-width` for constraints
- Use `height: 1fr` to fill vertical space

### 3. Handle Overflow
- Use `ScrollableContainer` for scrolling content
- Set `overflow-x` and `overflow-y` as needed
- Consider viewport constraints

### 4. Responsive Design
- Use reactive attributes for dynamic layouts
- Test with different terminal sizes
- Consider minimum terminal size (80×24)

## Common Pitfalls

### 1. Grid Confusion
```css
/* WRONG - These properties don't exist */
.item {
    grid-column: 1;
    grid-row: 2;
}

/* RIGHT - Use widget order in compose() */
/* First widget goes in first cell, etc. */
```

### 2. Width/Height Issues
```css
/* May not work as expected */
.widget {
    width: 100%;  /* Often unnecessary */
}

/* Better */
.widget {
    width: 1fr;   /* In grid or with specific parent */
}
```

### 3. Dock + Layout Conflicts
```css
/* Confusing behavior */
.container {
    layout: grid;
}
.child {
    dock: left;  /* Breaks out of grid */
}
```

### 4. Missing Constraints
```css
/* May grow indefinitely */
.sidebar {
    width: 25%;
    /* Missing: min-width, max-width */
}
```

## Debugging Layout Issues

### 1. Visual Debugging
- Add borders: `border: solid red;`
- Add backgrounds: `background: $error;`
- Use the Textual devtools: `textual run --dev app.py`

### 2. Check Widget Tree
- Ensure widgets are in correct order
- Verify parent-child relationships
- Check for accidentally nested containers

### 3. Simplify
- Start with basic layout
- Add complexity gradually
- Test each change

### 4. Common Fixes
- Remove unnecessary width: 100%
- Check for conflicting layout properties
- Ensure proper container nesting
- Verify dock doesn't break layout