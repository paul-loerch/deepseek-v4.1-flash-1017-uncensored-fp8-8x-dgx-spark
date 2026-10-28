#!/usr/bin/env bash
# check-public.sh [DIR] [PATTERN_FILE...] — pre-publication scan for data that must not be published.
# Generic patterns are built in (private/CGNAT IPv4, e-mail addresses, MAC addresses, key material, API tokens, home
# directories, macOS metadata files). Extra fixed-string or regex patterns (one per line, '#' comments) can be passed
# as files kept OUTSIDE the repository. Exits 1 on any hit and prints file:line:match. Documentation IPs (RFC 5737:
# 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) and the RFC 2606 example domains are allowed.
set -u
D=${1:-.}; shift || true
hits=0
report() { echo "[$1] $2"; hits=$((hits + 1)); }
scan() { # label, extended regex
  local out
  out=$(grep -rnIE --exclude-dir=.git -e "$2" "$D" 2>/dev/null \
        | grep -vE '(192\.0\.2\.|198\.51\.100\.|203\.0\.113\.|@example\.(com|org|net)|noreply@)' || true)
  [ -n "$out" ] && while IFS= read -r l; do report "$1" "$l"; done <<< "$out"
}
scan private-ipv4 '(^|[^0-9.])(10\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])|192\.168|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7]))\.[0-9]{1,3}\.[0-9]{1,3}([^0-9.]|$)'
scan email '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
scan mac-address '([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}'
scan key-material 'BEGIN [A-Z ]*PRIVATE KEY|ssh-(ed25519|rsa) AAAA|ecdsa-sha2-nistp[0-9]+ AAAA'
scan api-token '(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}|xox[abpr]-[A-Za-z0-9-]+)'
scan home-path '(/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|C:\\Users\\)'
scan secret-assignment '(^|[^A-Za-z0-9_])(password|passwd|secret|api_key|apikey|access_token|auth_token|token)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"'$<{[:space:]]{6,}["'"'"']'
for f in "$@"; do
  [ -f "$f" ] || continue
  while IFS= read -r p; do
    case "$p" in ''|'#'*) continue;; esac
    out=$(grep -rnIiF --exclude-dir=.git -e "$p" "$D" 2>/dev/null || true)
    [ -n "$out" ] && while IFS= read -r l; do report "listed" "$l"; done <<< "$out"
  done < "$f"
done
junk=$(find "$D" -path "$D/.git" -prune -o \( -name '.DS_Store' -o -name '._*' -o -name '*.rif' -o -name '.env' -o -name '__pycache__' -o -name '*.pyc' \) -print 2>/dev/null)
[ -n "$junk" ] && while IFS= read -r l; do report "junk-file" "$l"; done <<< "$junk"
if [ "$hits" -gt 0 ]; then echo "check-public: $hits finding(s)"; exit 1; fi
echo "check-public: clean"
