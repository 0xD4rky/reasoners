#!/bin/bash
set -e

echo "Building Qwen Tokenizer..."

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Build complete!"
echo ""
echo "Running tests..."
./test_tokenizer

echo ""
echo "To use in Python, add the following to your PYTHONPATH:"
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)"

