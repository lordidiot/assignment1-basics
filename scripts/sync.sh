#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="/Users/idiot/Projects/assignment1-basics"
NSCC_DEST="nscc:/home/users/nus/e1115691/idiot/assignment1-basics/"
CS336_DEST="cs336:/root/assignment1-basics/"

# Option 1: Replace the placeholder with your NSCC password.
# Option 2: Leave it unchanged and set NSCC_PASSWORD in your environment.
NSCC_PASSWORD="${NSCC_PASSWORD:-REPLACE_WITH_YOUR_PASSWORD}"

# Defaults
WATCH=true
EXCLUDE_DATA=true
USE_GITIGNORE=true
DRY_RUN=false
TARGET=""

usage() {
    cat <<EOF
Usage:
  $(basename "$0") TARGET [options]

Targets:
  nscc
  cs336

Options:
  --watch          Initial sync, then watch for changes (default)
  --no-watch       Sync once and exit
  --once           Alias for --no-watch

  --exclude-data   Exclude the top-level data/ directory (default)
  --include-data   Include the top-level data/ directory

  --gitignore      Respect Git ignore rules (default)
  --no-gitignore   Include files ignored by Git

  --dry-run        Preview changes without transferring anything
  -h, --help       Show this help

Examples:
  $(basename "$0") nscc
  $(basename "$0") nscc --no-watch
  $(basename "$0") cs336 --include-data
  $(basename "$0") cs336 --no-gitignore
  $(basename "$0") nscc --dry-run --no-watch
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 ||
        die "Required command not found: $1"
}

# watchman-make invokes this same script without command-line arguments.
# Pass the selected settings through environment variables for that invocation.
if [[ "${ASSIGNMENT_SYNC_WATCH_CHILD:-0}" == "1" ]]; then
    TARGET="${ASSIGNMENT_SYNC_TARGET:?Missing ASSIGNMENT_SYNC_TARGET}"
    WATCH=false
    EXCLUDE_DATA="${ASSIGNMENT_SYNC_EXCLUDE_DATA:-true}"
    USE_GITIGNORE="${ASSIGNMENT_SYNC_USE_GITIGNORE:-true}"
    DRY_RUN="${ASSIGNMENT_SYNC_DRY_RUN:-false}"
else
    while (($# > 0)); do
        case "$1" in
            nscc|cs336)
                [[ -z "$TARGET" ]] ||
                    die "Only one target may be specified."

                TARGET="$1"
                ;;

            --watch)
                WATCH=true
                ;;

            --no-watch|--once)
                WATCH=false
                ;;

            --exclude-data)
                EXCLUDE_DATA=true
                ;;

            --include-data)
                EXCLUDE_DATA=false
                ;;

            --gitignore)
                USE_GITIGNORE=true
                ;;

            --no-gitignore)
                USE_GITIGNORE=false
                ;;

            --dry-run)
                DRY_RUN=true
                ;;

            -h|--help)
                usage
                exit 0
                ;;

            *)
                die "Unknown argument: $1"
                ;;
        esac

        shift
    done
fi

[[ -n "$TARGET" ]] || {
    usage
    exit 1
}

[[ -d "$SOURCE_DIR" ]] ||
    die "Source directory does not exist: $SOURCE_DIR"

require_command rsync

case "$TARGET" in
    nscc)
        REMOTE_DEST="$NSCC_DEST"

        require_command sshpass

        [[ "$NSCC_PASSWORD" != "REPLACE_WITH_YOUR_PASSWORD" ]] ||
            die "Set NSCC_PASSWORD or replace REPLACE_WITH_YOUR_PASSWORD in the script."

        # sshpass -e reads the password from SSHPASS.
        export SSHPASS="$NSCC_PASSWORD"
        RSYNC_COMMAND=(sshpass -e rsync)
        ;;

    cs336)
        REMOTE_DEST="$CS336_DEST"
        RSYNC_COMMAND=(rsync)
        ;;

    *)
        die "Unsupported target: $TARGET"
        ;;
esac

SSH_COMMAND="ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

RSYNC_OPTIONS=(
    --archive
    --recursive
    --compress
    --human-readable
    --partial
    --itemize-changes
    --exclude='/.git/'
)

if [[ "$DRY_RUN" == true ]]; then
    RSYNC_OPTIONS+=(--dry-run)
fi

# Produce a NUL-delimited list containing:
#   1. Tracked files
#   2. Untracked files that are not ignored by Git
#
# Tracked files remain included even when a later .gitignore rule matches them,
# which is normal Git behavior.
git_file_list() {
    local path

    cd "$SOURCE_DIR"

    git ls-files \
        --cached \
        --others \
        --exclude-standard \
        -z |
    while IFS= read -r -d '' path; do
        # A tracked file may have been deleted locally but remain in Git's index.
        # Skip it rather than asking rsync to remove anything remotely.
        [[ -e "$path" || -L "$path" ]] || continue

        # Avoid recursively copying a checked-out Git submodule as a normal
        # directory. Ordinary files and symbolic links are still included.
        [[ ! -d "$path" || -L "$path" ]] || continue

        # Exclude only SOURCE_DIR/data, not nested directories named data.
        if [[ "$EXCLUDE_DATA" == true ]] &&
           { [[ "$path" == "data" ]] || [[ "$path" == data/* ]]; }; then
            continue
        fi

        printf '%s\0' "$path"
    done
}

sync_once() {
    printf '\nSyncing:\n'
    printf '  Source:      %s/\n' "$SOURCE_DIR"
    printf '  Destination: %s\n\n' "$REMOTE_DEST"

    if [[ "$USE_GITIGNORE" == true ]]; then
        require_command git

        git -C "$SOURCE_DIR" rev-parse \
            --is-inside-work-tree >/dev/null 2>&1 ||
            die "$SOURCE_DIR is not inside a Git working tree. Use --no-gitignore to bypass Git filtering."

        git_file_list |
            "${RSYNC_COMMAND[@]}" \
                "${RSYNC_OPTIONS[@]}" \
                --from0 \
                --files-from=- \
                -e "$SSH_COMMAND" \
                "$SOURCE_DIR/" \
                "$REMOTE_DEST"
    else
        local options=("${RSYNC_OPTIONS[@]}")

        if [[ "$EXCLUDE_DATA" == true ]]; then
            options+=(--exclude='/data/')
        fi

        "${RSYNC_COMMAND[@]}" \
            "${options[@]}" \
            -e "$SSH_COMMAND" \
            "$SOURCE_DIR/" \
            "$REMOTE_DEST"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        printf '\nDry run complete: %s\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')"
    else
        printf '\nSync complete: %s\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

# Always perform an initial sync.
sync_once

# Stop after the initial sync when Watchman is disabled.
if [[ "$WATCH" == false ]]; then
    exit 0
fi

require_command watchman-make

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

# watchman-make accepts a script path and invokes it when matching files
# change. Export the current options so the invoked script performs a
# one-shot sync using the same settings.
export ASSIGNMENT_SYNC_WATCH_CHILD=1
export ASSIGNMENT_SYNC_TARGET="$TARGET"
export ASSIGNMENT_SYNC_EXCLUDE_DATA="$EXCLUDE_DATA"
export ASSIGNMENT_SYNC_USE_GITIGNORE="$USE_GITIGNORE"
export ASSIGNMENT_SYNC_DRY_RUN="$DRY_RUN"
export NSCC_PASSWORD

printf '\nWatching %s for changes.\n' "$SOURCE_DIR"
printf 'Press Ctrl-C to stop.\n\n'

cd "$SOURCE_DIR"

exec watchman-make \
    -p '**/*' \
    --run "$SCRIPT_PATH"