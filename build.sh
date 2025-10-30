#!/bin/bash
set -e

echo "building reasoners"

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "build complete!"
echo "to use in python, add the following to your PYTHONPATH:"
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)"
