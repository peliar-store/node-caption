#!/bin/bash
# Patch ffi-napi's bundled node-addon-api to fix napi_add_finalizer
# signature incompatibility with Electron 28+.
# Run after npm install if electron-rebuild fails.

FILE="node_modules/ffi-napi/node_modules/node-addon-api/napi-inl.h"

if [ ! -f "$FILE" ]; then
  echo "File not found: $FILE (already patched or ffi-napi not installed)"
  exit 0
fi

sed -i 's/status = napi_add_finalizer(env, obj, data, finalizer, hint, nullptr);/status = napi_add_finalizer(env, obj, data, reinterpret_cast<node_api_basic_finalize>(finalizer), hint, nullptr);/' "$FILE"

echo "Patched $FILE"
