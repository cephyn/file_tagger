# Recursive Folder Scanning Design

**Status:** Approved
**Date:** 2026-11-26
**Implementation scope:** `Scan Directory` behavior

## Goal

When the user runs **Scan Directory**, File Tagger will find untagged files in
the selected directory and every visible nested subdirectory. The existing
untagged-file dialog and tag-application behavior will remain unchanged.

## Confirmed Product Decisions

- Recursive traversal starts at the directory currently shown in the file
  explorer and includes all descendant directories.
- Dot-prefixed hidden directories and files are skipped.
- Hidden Windows entries are also skipped where the platform exposes the hidden
  file attribute.
- Symbolic links are not traversed or scanned.
- Cancelling the scan stops discovery and processing without opening a partial
  result dialog.
- Filesystem enumeration errors are reported through the existing **Scan Error**
  dialog rather than silently ignored.

## Considered Approaches

1. **`os.walk`** — recommended. It provides recursive traversal, supports
   pruning directories before descent, avoids following symbolic directory
   links, and integrates directly with the existing progress loop.
2. **`Path.rglob('*')`** — concise, but hidden-path pruning and cancellation
   require additional filtering and provide less explicit control over traversal
   failures.
3. **Custom `os.scandir` traversal** — maximum control over cancellation and
   permissions, but unnecessary complexity for this feature.

## Approved Architecture

Add a testable file-enumeration helper in a utility module or the file-tagging
module. The helper returns an iterator of visible, regular files below one root
directory.

The existing `ScanThread` will replace its top-level `os.scandir` block with
calls to this helper. The helper will:

- Use `os.walk(..., followlinks=False)`.
- Prune dot-prefixed directories and Windows hidden directories before descent.
- Skip dot-prefixed files and Windows hidden files.
- Skip symbolic-file links.
- Stop yielding files when a caller-provided cancellation predicate becomes true.
- Raise directory-enumeration errors so the worker emits the existing
  `scan_error` signal.

The worker will continue to:

1. Discover the complete file list.
2. Compute the total for progress reporting.
3. Query each file against the database.
4. Generate suggestions only for files that are absent or currently untagged.
5. Emit the existing `scan_finished` mapping or `scan_error` signal.

The dialog will receive the same `dict[path, suggestions]` shape it receives
today. Paths from nested subfolders remain distinguishable by their full paths,
even though the list will continue to display file names and tooltips.

## Error and Cancellation Behavior

- `os.walk` will receive an error callback that raises. The thread catches the
  exception and sends it to the existing scan-error connection.
- Cancellation stops the enumeration iterator before additional work is started
  and also remains checked by the file-processing loop.
- A cancellation must not emit `scan_finished`, preventing a partial results
  dialog.
- Directory symlinks are not followed. File symlinks are omitted.

## Testing

Focused automated tests will cover:

- Discovery of root-level and nested files.
- Pruning dot-prefixed hidden directories.
- Skipping dot-prefixed hidden files.
- Skipping symbolic-file and symbolic-directory links.
- Prompt cancellation during enumeration.
- Propagation of filesystem enumeration errors.

The implementation is complete only after the focused tests fail before the
change and pass after the change, and the relevant existing UI suite remains
green.
