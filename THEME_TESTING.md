# Theme Testing Guide

## Quick Start Testing

### Step 1: Verify Theme Button Visibility
1. Start the Flask server: `python app.py`
2. Open `http://localhost:5000` in browser
3. Look at the **top-right corner**
4. You should see a **moon icon (🌙)** on a white button
5. Button should have light grey border

### Step 2: Test Light Mode (Default)
Verify these colors are visible:
- ✅ White cards with dark text
- ✅ Navy blue buttons
- ✅ Light grey borders
- ✅ White upload area
- ✅ Dark text that's easy to read

### Step 3: Switch to Dark Mode
1. Click the **moon icon (🌙)** in top-right
2. Verify smooth color transition (0.3s)
3. Icon should change to **sun (☀️)**
4. Background should become very dark

### Step 4: Verify Dark Mode Colors
Check these elements:
- ✅ Dark slate cards with white text
- ✅ Blue buttons
- ✅ Dark borders
- ✅ Dark upload area
- ✅ White text that's easy to read
- ✅ All text remains readable

### Step 5: Test Persistence
1. Refresh the page (F5 or Ctrl+R)
2. Dark mode should still be active
3. Icon should still be sun (☀️)

### Step 6: Test Browser Restart
1. Close the browser completely
2. Reopen browser
3. Navigate to `http://localhost:5000`
4. Dark mode should load automatically
5. Theme preference was saved!

---

## Detailed Component Testing

### Upload Section

**Light Mode**
```
□ Box has light grey dashed border
□ Background is light grey
□ Text is dark grey
□ Hover: Box becomes white
□ Hover: Border becomes blue
□ Icon (upload arrow) is navy blue
□ Browse button is navy blue
□ Button hover: Becomes medium blue
```

**Dark Mode**
```
□ Box has dark grey dashed border
□ Background is deep black
□ Text is off-white
□ Hover: Box becomes dark slate
□ Hover: Border becomes bright blue
□ Icon is bright blue
□ Browse button is bright blue
□ Button hover: Becomes lighter blue
```

### Personality Details Section

**Light Mode**
```
□ Main box has white background
□ Left border is navy blue
□ All text is dark grey
□ Sub-boxes have light grey background
□ Characteristic items are readable
□ Strength items have green left border
□ Weakness items have yellow left border
```

**Dark Mode**
```
□ Main box has dark slate background
□ Left border is bright blue
□ All text is off-white
□ Sub-boxes have deep black background
□ Characteristic items are readable
□ Strength items have green left border
□ Weakness items have yellow left border
```

---

## Color Verification Checklist

### Light Mode Colors

Use browser DevTools Color Picker (Ctrl+Shift+C):

- [ ] Navy Blue (#1e3a8a) - Headers, primary buttons
- [ ] Blue (#3b82f6) - Secondary elements
- [ ] Light Grey (#f3f4f6) - Backgrounds
- [ ] White (#ffffff) - Cards, surfaces
- [ ] Dark Grey (#1f2937) - Main text
- [ ] Medium Grey (#6b7280) - Secondary text
- [ ] Light Grey Border (#d1d5db) - Borders

### Dark Mode Colors

Use browser DevTools Color Picker:

- [ ] Blue (#3b82f6) - Primary buttons, icons
- [ ] Light Blue (#60a5fa) - Hovers, secondary elements
- [ ] Deep Black (#0f172a) - Main background
- [ ] Dark Slate (#1e293b) - Cards, surfaces
- [ ] Off-white (#f1f5f9) - Main text
- [ ] Light Slate (#cbd5e1) - Secondary text
- [ ] Dark Grey (#334155) - Borders

---

## Browser Compatibility Testing

### Test on These Browsers

- [ ] Chrome (Latest)
- [ ] Firefox (Latest)
- [ ] Safari (Latest)
- [ ] Edge (Latest)
- [ ] Chrome Mobile (Android)
- [ ] Safari Mobile (iOS)

### Expected Result
All browsers should:
- Display light mode on first load
- Switch to dark mode when button clicked
- Persist theme choice
- Show smooth transitions
- Display correct colors

---

## Accessibility Testing

### Keyboard Navigation
```
1. Tab key: Navigate to theme toggle button
   Expected: Button receives focus outline
   
2. Space or Enter: Activate theme toggle
   Expected: Theme switches, icon changes
   
3. Tab again: Move to next element
   Expected: Focus outline visible on next element
```

### Color Contrast Testing
Use: WebAIM Contrast Checker or browser DevTools

**Light Mode**
- [ ] Dark grey text on white: Ratio should be 6.5:1 or higher
- [ ] Navy blue button on white: Ratio should be 6.8:1 or higher
- [ ] Light grey border visible on white

**Dark Mode**
- [ ] Off-white text on dark slate: Ratio should be 8.5:1 or higher
- [ ] Blue button on dark slate: Ratio should be 8.2:1 or higher
- [ ] Dark grey border visible on deep black

### Screen Reader Testing
Using NVDA (Windows) or VoiceOver (Mac):
- [ ] Theme toggle button is announced
- [ ] Button purpose is clear ("Toggle dark/light mode")
- [ ] Theme change is perceivable
- [ ] All text is readable by screen reader

---

## Responsive Design Testing

### Desktop (1920×1080)
- [ ] Theme button visible top-right
- [ ] Button size appropriate (50×50px)
- [ ] All elements adapt correctly
- [ ] No overlapping elements

### Tablet (768×1024)
- [ ] Theme button still accessible
- [ ] Cards stack properly
- [ ] Text remains readable
- [ ] Colors display correctly

### Mobile (375×667)
- [ ] Theme button is touchable (min 44×44px)
- [ ] Button doesn't overlap content
- [ ] Cards adapt to narrow screen
- [ ] All text visible and readable
- [ ] Colors work on small screen

---

## Performance Testing

### LocalStorage Check
```javascript
// In Browser Console
localStorage.getItem('theme')
// Should return: 'light' or 'dark'

localStorage.setItem('theme', 'dark')
localStorage.setItem('theme', 'light')
// Should work without errors
```

### Theme Switch Performance
- [ ] Theme changes instantly (< 100ms)
- [ ] No lag when clicking button
- [ ] Transitions are smooth
- [ ] No page flicker
- [ ] No console errors

### Network Load
- [ ] No extra network requests
- [ ] Theme is client-side only
- [ ] LocalStorage doesn't affect performance

---

## Edge Case Testing

### Testing Scenarios

**1. Multiple Tabs**
- [ ] Open app in two browser tabs
- [ ] Switch to dark mode in tab 1
- [ ] Refresh tab 2
- [ ] Dark mode should load in tab 2 (shared localStorage)

**2. Private Browsing**
- [ ] Open app in private/incognito mode
- [ ] Switch to dark mode
- [ ] Close private window
- [ ] Open new private window
- [ ] Light mode should load (no persistent storage)

**3. LocalStorage Disabled**
- [ ] Disable localStorage in DevTools
- [ ] Try to switch theme
- [ ] Should still work for current session
- [ ] Won't persist after refresh

**4. Browser Storage Full**
- [ ] Fill up localStorage with other data
- [ ] Try to set theme
- [ ] Should handle gracefully
- [ ] Fallback to light mode

---

## Visual Regression Testing

### Screenshot Comparison

Take screenshots in both modes:

**Light Mode Screenshots**
- [ ] Full page light mode
- [ ] Upload area (default state)
- [ ] Upload area (hover state)
- [ ] Results section
- [ ] Personality details
- [ ] Mobile view

**Dark Mode Screenshots**
- [ ] Full page dark mode
- [ ] Upload area (default state)
- [ ] Upload area (hover state)
- [ ] Results section
- [ ] Personality details
- [ ] Mobile view

Compare with previous versions to catch regressions.

---

## Testing Checklist Summary

```
FUNCTIONALITY
☐ Light mode loads by default
☐ Dark mode toggle works
☐ Theme persists on refresh
☐ Theme persists after browser restart
☐ Icon changes between moon/sun

COLORS
☐ All light mode colors correct
☐ All dark mode colors correct
☐ Transitions are smooth
☐ All text readable in both modes
☐ All borders visible

ACCESSIBILITY
☐ Keyboard navigation works
☐ Focus indicators visible
☐ Color contrast meets WCAG AA
☐ No color-only information
☐ Screen reader compatible

RESPONSIVE
☐ Desktop: All elements display
☐ Tablet: Adapts properly
☐ Mobile: Button is touchable
☐ No overlapping elements
☐ Text remains readable

PERFORMANCE
☐ No console errors
☐ Theme switch is instant
☐ No network requests
☐ LocalStorage working
☐ No lag or flicker

CROSS-BROWSER
☐ Chrome/Edge ✓
☐ Firefox ✓
☐ Safari ✓
☐ Mobile Chrome ✓
☐ Mobile Safari ✓
```

---

## Reporting Issues

If you find any issues during testing:

1. **Note the issue**: What's wrong?
2. **Browser/Device**: Where did it happen?
3. **Steps to reproduce**: How to make it happen?
4. **Expected vs. Actual**: What should happen?
5. **Screenshot**: Visual evidence

Example:
```
Issue: Dark mode button is hidden on mobile
Browser: Chrome Mobile (Samsung Galaxy S21)
Steps: Open app → Rotate to portrait
Expected: Button visible in top-right
Actual: Button hidden behind header
Screenshot: [attach image]
```

---

## Quick Reference Commands

### Keyboard Shortcuts
```
Ctrl+Shift+C    - Open color picker (DevTools)
F12             - Open DevTools
Ctrl+J          - Open Console
Ctrl+Shift+K    - Open Console (Firefox)
Ctrl+R          - Refresh page
Ctrl+Shift+R    - Hard refresh (clear cache)
```

### Browser Console Commands
```javascript
// Check current theme
document.documentElement.getAttribute('data-theme')

// Switch theme manually
document.documentElement.setAttribute('data-theme', 'dark')
document.documentElement.setAttribute('data-theme', 'light')

// Check localStorage
localStorage.getItem('theme')
localStorage.setItem('theme', 'dark')
localStorage.clear() // WARNING: Clears all storage

// Check CSS variables
getComputedStyle(document.body).getPropertyValue('--primary-color')
```

---

Ready to test! Happy clicking! 🎉
