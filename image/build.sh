#!/usr/bin/env bash
# image/build.sh [--tag TAG] [--distribute]
#
# Builds the serving image on this machine (run it on NODE1), checks that the build-time edits are in the image, and
# optionally copies the image to the seven other nodes listed in ../cluster.env so that every node runs the identical image ID.
#
#   --tag TAG      image tag (default dsv41-flash-sm121:n1, the tag config/recipe.env expects)
#   --distribute   docker save | ssh NODEn docker load, for NODE2..NODE8, then compare the image IDs
#
# Needs: docker; for --distribute, ../cluster.env (NODE1..NODE8, SSH_USER, SSH_OPTS) and docker on every node.
# The base image is pulled by digest (see the Dockerfile). Nothing else is downloaded at build time.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=dsv41-flash-sm121:n1
DISTRIBUTE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) [ $# -ge 2 ] || { echo "--tag needs a value (see --help)" >&2; exit 2; }; TAG=$2; shift ;;
    --distribute) DISTRIBUTE=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

echo "== building $TAG"
docker build -t "$TAG" -f "$HERE/Dockerfile" "$HERE"

# The bake script is idempotent: run again in a throw-away container, it reports every edited file as "already applied"
# when the build put the edits in. "applied now" would mean the image was built without them.
echo "== checking the build-time edits (no GPU needed)"
if ! out=$(docker run --rm --entrypoint python3 "$TAG" -I /opt/dsv41-bake/dsv41_bake.py 2>&1); then
  printf '%s\n' "$out" >&2
  echo "the bake check failed in $TAG" >&2
  exit 1
fi
printf '%s\n' "$out"
case "$out" in
  *"applied now"*) echo "$TAG was built without the build-time edits (the check had to apply them)" >&2; exit 1 ;;
esac
case "$out" in
  *"already applied"*) ;;
  *) echo "the bake check printed no 'already applied' line; is $TAG the image built from this Dockerfile?" >&2; exit 1 ;;
esac

ID=$(docker image inspect "$TAG" --format '{{.Id}}')
echo "== built $TAG = $ID"
[ "$DISTRIBUTE" = 1 ] || exit 0

ENV_FILE=${CLUSTER_ENV:-$HERE/../cluster.env}
[ -r "$ENV_FILE" ] || { echo "no $ENV_FILE (copy cluster.env.example to cluster.env and fill it in)" >&2; exit 2; }
# shellcheck disable=SC1090
. "$ENV_FILE"
read -r -a SSH_ARGS <<< "${SSH_OPTS:-}"
target() { if [ -n "${SSH_USER:-}" ]; then echo "$SSH_USER@$1"; else echo "$1"; fi; }
fail=0
for i in 2 3 4 5 6 7 8; do
  var=NODE$i; node=${!var:-}
  [ -n "$node" ] && [ "$node" != CHANGE_ME ] || { echo "NODE$i is not set in $ENV_FILE" >&2; exit 2; }
  echo "== copying the image to $node (NODE$i)"
  docker save "$TAG" | ssh "${SSH_ARGS[@]}" "$(target "$node")" docker load
  got=$(ssh "${SSH_ARGS[@]}" "$(target "$node")" docker image inspect "$TAG" --format '{{.Id}}')
  if [ "$got" = "$ID" ]; then echo "   $node: $got"; else echo "   $node: $got != $ID" >&2; fail=1; fi
done
[ "$fail" = 0 ] && echo "== all eight nodes have $TAG = $ID" || { echo "image IDs differ: launch/up.sh will refuse" >&2; exit 1; }
