#!/usr/bin/env bash
set -euo pipefail

# Converts an FBX file to glTF/GLB using the external FBX2glTF tool.
# Requirements:
# - FBX2glTF in PATH (https://github.com/facebookincubator/FBX2glTF)
#
# Usage:
#   tools/convert_fbx2gltf.sh input.fbx output_dir [--glb]
#
# Examples:
#   tools/convert_fbx2gltf.sh assets/models/environment/SunTemple/SunTemple.fbx assets/models/environment/SunTemple/converted --glb

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 input.fbx output_dir [--glb]" >&2
  exit 2
fi

in="$1"
out_dir="$2"
mode="gltf"
if [[ ${3:-} == "--glb" ]]; then
  mode="glb"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

fbx2gltf_bin="${FBX2GLTF_BIN:-}"
if [[ -z "$fbx2gltf_bin" ]]; then
  if command -v FBX2glTF >/dev/null 2>&1; then
    fbx2gltf_bin="FBX2glTF"
  elif [[ -x "$script_dir/FBX2glTF" ]]; then
    fbx2gltf_bin="$script_dir/FBX2glTF"
  elif [[ -x "$script_dir/FBX2glTF-linux-x64" ]]; then
    fbx2gltf_bin="$script_dir/FBX2glTF-linux-x64"
  fi
fi

if [[ -z "$fbx2gltf_bin" ]]; then
  echo "FBX2glTF not found." >&2
  echo "Put it in PATH (name: FBX2glTF) or place the binary in tools/ (e.g. tools/FBX2glTF-linux-x64)." >&2
  echo "Alternatively set FBX2GLTF_BIN=/absolute/path/to/FBX2glTF" >&2
  echo "Repo: https://github.com/facebookincubator/FBX2glTF" >&2
  exit 1
fi

mkdir -p "$out_dir"
base="$(basename "$in")"
name="${base%.*}"

if [[ "$mode" == "glb" ]]; then
  out_file="$out_dir/$name.glb"
  "$fbx2gltf_bin" --binary --input "$in" --output "$out_file"
  echo "Wrote: $out_file"
else
  out_file="$out_dir/$name.gltf"
  "$fbx2gltf_bin" --input "$in" --output "$out_file"
  echo "Wrote: $out_file"
  echo "Note: textures may be copied alongside depending on FBX2glTF build/options."
fi
