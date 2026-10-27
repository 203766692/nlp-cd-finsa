#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/CroFinSA}"
TRAIN_URL="${TRAIN_URL:-}"
VALID_URL="${VALID_URL:-}"
TEST_URL="${TEST_URL:-}"

usage() {
  cat <<'EOF'
用法：
  bash scripts/download_data.sh [选项]

选项：
  --train-url URL   训练集下载地址
  --valid-url URL   验证集下载地址
  --test-url URL    测试集下载地址
  --data-dir DIR    数据保存目录，默认 data/CroFinSA
  -h, --help        显示帮助

也可通过 TRAIN_URL、VALID_URL、TEST_URL 和 DATA_DIR 环境变量传入。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-url)
      TRAIN_URL="${2:-}"
      shift 2
      ;;
    --valid-url)
      VALID_URL="${2:-}"
      shift 2
      ;;
    --test-url)
      TEST_URL="${2:-}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$TRAIN_URL" && -z "$VALID_URL" && -z "$TEST_URL" ]]; then
  echo "未提供下载地址。请传入教师或数据平台发布的 URL。" >&2
  usage >&2
  exit 1
fi

mkdir -p "$DATA_DIR"

download_file() {
  local url="$1"
  local target="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 "$url" -o "$target"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$target" "$url"
  else
    echo "未找到 curl 或 wget，无法下载数据。" >&2
    exit 1
  fi
}

if [[ -n "$TRAIN_URL" ]]; then
  download_file "$TRAIN_URL" "$DATA_DIR/train.json"
fi
if [[ -n "$VALID_URL" ]]; then
  download_file "$VALID_URL" "$DATA_DIR/valid.json"
fi
if [[ -n "$TEST_URL" ]]; then
  download_file "$TEST_URL" "$DATA_DIR/test.json"
fi

echo "数据已保存到：$DATA_DIR"
