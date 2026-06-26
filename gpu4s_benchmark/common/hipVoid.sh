#!/bin/bash

# Check if a target file was passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_file.cpp>"
    exit 1
fi

TARGET_FILE="$1"

# Use sed to find the exact function names and prepend 'void '
# -i modifies the file in place
# -E enables extended regular expressions
sed -i -E 's/\b(hipSetDevice|hipGetDeviceProperties|hipEventCreate|hipEventRecord|hipEventSynchronize|hipEventElapsedTime)\b/(void)\1/g' "$TARGET_FILE"

echo "Successfully updated HIP functions in $TARGET_FILE"