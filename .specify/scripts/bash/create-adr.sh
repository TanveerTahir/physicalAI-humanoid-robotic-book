#!/bin/bash

# Script to create a new Architecture Decision Record (ADR)
# Usage: ./create-adr.sh "Title of the ADR"

set -e  # Exit on any error

TITLE="$1"
if [ -z "$TITLE" ]; then
    echo "Usage: $0 \"<ADR Title>\""
    exit 1
fi

# Get the next ADR number by checking existing files
NEXT_NUMBER=1
ADR_DIR="history/adr"
if [ -d "$ADR_DIR" ]; then
    MAX_NUM=$(ls "$ADR_DIR"/ADR-*.md 2>/dev/null | sed 's|.*/ADR-\([0-9]*\)\.md|\1|' | sort -n | tail -1)
    if [ ! -z "$MAX_NUM" ]; then
        NEXT_NUMBER=$((MAX_NUM + 1))
    fi
fi

# Create the ADR directory if it doesn't exist
mkdir -p "$ADR_DIR"

# Generate the ADR filename
FILENAME="$ADR_DIR/ADR-$NEXT_NUMBER.md"

# Create the ADR template
cat > "$FILENAME" << EOF
# ADR-$NEXT_NUMBER: $TITLE

Date: $(date +%Y-%m-%d)

## Status

Proposed

## Context

[Describe the context and problem statement. What are the forces at play? What requirements need to be met?]

## Decision

[What is the change that we're proposing and doing?]

## Consequences

[What becomes easier or more difficult to do because of this change? What are the trade-offs?]

## Alternatives

[What other alternatives were considered? Why were they rejected?]

## References

[Any links to relevant resources, discussions, or documentation]
EOF

# Output JSON response with the path and ID
echo "{\"adr_path\": \"$FILENAME\", \"adr_id\": $NEXT_NUMBER}"