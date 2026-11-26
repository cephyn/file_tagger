# Random Tag Color Design

**Status:** Approved
**Date:** 2026-11-26
**Implementation scope:** Tag create and edit behavior

## Goal

Add an obvious **Assign random color** action to both tag creation and tag
editing. Randomization should update the dialog preview and require the normal
OK action before any database change is saved.

## Confirmed Product Decisions

- A single reusable tag dialog will handle both create and edit operations.
- Create dialogs begin with an empty name and a random initial preview color.
- Edit dialogs begin with the existing name and existing preview color.
- **Assign random color** updates the preview only. It does not save or close
  the dialog.
- **Choose color** continues to provide manual color selection.
- **OK** validates and saves the name and current preview color.
- **Cancel** makes no database change.
- Random colors use vivid HSV values: full hue range, 128–255 saturation, and
  180–255 brightness.
- Random assignment avoids the exact color of another tag. Editing an existing
  tag may keep its own current color.

## Considered Approaches

1. **Shared custom tag dialog** — recommended. It combines name, preview,
   random assignment, manual selection, and save/cancel controls in one
   consistent interface for create and edit.
2. **Native picker with an extra pre-choice prompt** — less code, but adds a
   blocking choice and cannot place an explicit random action inside the native
   Qt color dialog.
3. **Randomize from the main tag toolbar** — easy to discover for edits, but
   immediate-save behavior would differ between create and edit and could
   surprise users.

## Approved Architecture

Introduce a small `TagEditDialog` in `file_tag_manager.py` or a focused UI
module. The dialog owns only editing state and exposes the chosen name and
color to `FileTagManager`.

The dialog contains:

- A `QLineEdit` for the tag name.
- A colored preview widget or framed `QLabel`.
- A **Choose color** button that opens `QColorDialog` with the current preview
  as its initial color.
- An **Assign random color** button that calls a shared random-color helper.
- Standard OK and Cancel buttons.

`FileTagManager.add_tag()` opens the dialog with no existing tag. After OK, it:

1. Validates that the name is nonblank.
2. Checks that the name is unique.
3. Creates the tag with the dialog's preview color.
4. Commits through the existing modern/legacy synchronization path.
5. Refreshes the tag lists.

`FileTagManager.edit_tag()` opens the dialog with the selected tag's name and
color. After OK, it:

1. Validates that the name is nonblank.
2. Checks duplicate names only if the name changed.
3. Updates the selected tag's name and color.
4. Commits through the existing modern/legacy synchronization path.
5. Refreshes affected lists and file-tag associations.

## Random Color Behavior

Use `QColor.fromHsv(hue, saturation, value).name()` with the existing
application's vivid HSV ranges. Generate colors from a cryptographically
unnecessary source suitable for UI selection.

The dialog asks `FileTagManager` or a shared helper for the set of existing
tag colors. It retries a small bounded number of times to avoid an exact
duplicate. If all retries collide, the final generated valid color is
accepted; tag creation will not be blocked because of color uniqueness.

## Error Handling

- Whitespace-only names are invalid and keep the dialog open.
- Existing-name conflicts keep the dialog open and show the existing warning.
- Cancelling the color picker leaves the current preview unchanged.
- Cancelling the tag dialog leaves the database unchanged.
- Invalid `QColor` values are ignored and retain the prior preview.

## Testing

Focused Qt tests will cover:

- Creating a tag through the shared dialog.
- Editing a tag name and color through the shared dialog.
- Initial random preview selection for creation.
- Initial existing-color preview selection for editing.
- Random preview updates without persisting until OK.
- Manual color selection updating the preview.
- Nonblank name and duplicate-name validation.
- Cancel behavior without database changes.
- Existing tag-color collision avoidance.
- Color and name synchronization with the modern persistence path.

The implementation is complete only after the focused tests pass and the
relevant existing UI suite remains green.
