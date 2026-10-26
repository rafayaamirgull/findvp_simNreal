#!/usr/bin/env bash
# steve_env.sh — connect THIS dev-PC container to Steve's onboard-PC ROS 2 graph.
#
#   source /home/ws/steve_env.sh          # set up network + env in this shell
#   source /home/ws/steve_env.sh --check  # just report status, change nothing
#
# Nothing on the onboard robot PC is touched. All changes are local to this
# container and are NOT persistent (live `ip` address + shell env) — re-source
# after a container restart or a cable replug.
#
# Background (found 2026-09-10): the onboard PC (mmo-700-uni-bonn-2204) publishes
# ROS 2 on the 192.168.60.0/24 subnet — DDS discovery multicast is sourced from
# 192.168.60.90 (its enp2s0 NIC). It is NOT reachable for ROS on 192.168.1.0/24
# even though that NIC answers ARP for 192.168.1.10 (a dead-end address).
# Onboard ROS settings: ROS_DOMAIN_ID=74, RMW=rmw_fastrtps_cpp, LOCALHOST_ONLY=0.

# ---- config -----------------------------------------------------------------
STEVE_IFACE="${STEVE_IFACE:-}"              # auto-detect the wired NIC when unambiguous
STEVE_MYIP="${STEVE_MYIP:-192.168.60.50}"   # address to claim (must be free)
STEVE_CIDR=24
STEVE_ONBOARD="${STEVE_ONBOARD:-192.168.60.90}"
STEVE_DOMAIN_ID=74
STEVE_RMW=rmw_fastrtps_cpp
# ---------------------------------------------------------------------------

if [ -z "$STEVE_IFACE" ] && command -v ip >/dev/null 2>&1; then
  # Prefer an interface which already carries Steve's subnet.  Otherwise use
  # one unambiguous, connected, non-wireless Ethernet interface.  Do not use
  # `ip route get` here: before the Steve address is assigned it commonly
  # returns the laptop's Wi-Fi/default-route interface.
  STEVE_IFACE=$(ip -o -4 addr show 2>/dev/null | awk '
    $4 ~ /^192\.168\.60\./ { print $2; exit }
  ')
  if [ -z "$STEVE_IFACE" ]; then
    _steve_candidates=()
    for _steve_net_path in /sys/class/net/*; do
      [ -e "$_steve_net_path" ] || continue
      _steve_name=${_steve_net_path##*/}
      case "$_steve_name" in lo|docker*|br-*|veth*|virbr*|tun*|tap*) continue ;; esac
      [ -d "$_steve_net_path/wireless" ] && continue
      [ "$(cat "$_steve_net_path/type" 2>/dev/null)" = "1" ] || continue
      [ "$(cat "$_steve_net_path/carrier" 2>/dev/null)" = "1" ] || continue
      _steve_candidates+=("$_steve_name")
    done
    if [ "${#_steve_candidates[@]}" -eq 1 ]; then
      STEVE_IFACE=${_steve_candidates[0]}
    elif [ "${#_steve_candidates[@]}" -gt 1 ]; then
      echo "Multiple connected Ethernet interfaces found: ${_steve_candidates[*]}" >&2
      echo "Run 'ip -br link', then export STEVE_IFACE=<Steve cable interface>." >&2
      return 1 2>/dev/null || exit 1
    fi
    unset _steve_candidates _steve_net_path _steve_name
  fi
fi

_steve_status() {
  local ip carrier
  carrier=$(cat "/sys/class/net/$STEVE_IFACE/carrier" 2>/dev/null || echo "?")
  ip=$(ip -4 -br addr show "$STEVE_IFACE" 2>/dev/null | awk '{$1=$2="";print}' | xargs)
  echo "iface     : $STEVE_IFACE  (carrier=$carrier)  addr: ${ip:-<none>}"
  if ping -c1 -W1 "$STEVE_ONBOARD" >/dev/null 2>&1; then
    echo "onboard   : $STEVE_ONBOARD reachable"
  else
    echo "onboard   : $STEVE_ONBOARD NOT reachable  (cable? wrong iface? set STEVE_IFACE=)"
  fi
  echo "ROS env   : DOMAIN_ID=${ROS_DOMAIN_ID:-unset}  LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-unset}  RMW=${RMW_IMPLEMENTATION:-unset}"
}

if [ "${1:-}" = "--check" ]; then
  _steve_status
  return 0 2>/dev/null || exit 0
fi

# --- 1. network: put STEVE_IFACE on the onboard subnet (idempotent) ---------
if [ -z "$STEVE_IFACE" ] || [ ! -d "/sys/class/net/$STEVE_IFACE" ]; then
  echo "Steve Ethernet interface ${STEVE_IFACE:-<unset>} is not present on this laptop." >&2
  echo "Reconnect the USB Ethernet adapter; then check: ip -br link" >&2
  return 1 2>/dev/null || exit 1
fi
if [ "$(cat "/sys/class/net/$STEVE_IFACE/carrier" 2>/dev/null)" != "1" ]; then
  if ! ip -br link show "$STEVE_IFACE" | grep -q '<[^>]*UP'; then
    sudo ip link set "$STEVE_IFACE" up || {
      echo "Could not enable $STEVE_IFACE." >&2
      return 1 2>/dev/null || exit 1
    }
  fi
  sleep 1
  if [ "$(cat "/sys/class/net/$STEVE_IFACE/carrier" 2>/dev/null)" != "1" ]; then
    echo "$STEVE_IFACE exists but has no Ethernet carrier; check its cable and Steve's power." >&2
    return 1 2>/dev/null || exit 1
  fi
fi
if [ -n "$STEVE_IFACE" ] && ! ip -4 addr show "$STEVE_IFACE" 2>/dev/null | grep -q 'inet 192\.168\.60\.'; then
  sudo ip addr add "${STEVE_MYIP}/${STEVE_CIDR}" dev "$STEVE_IFACE" 2>/dev/null \
    && echo "added ${STEVE_MYIP}/${STEVE_CIDR} to $STEVE_IFACE" \
    || {
      echo "Could not assign ${STEVE_MYIP}/${STEVE_CIDR} to $STEVE_IFACE; check sudo and existing addresses." >&2
      return 1 2>/dev/null || exit 1
    }
fi

# --- 2. ROS 2 discovery must match the onboard PC --------------------------
export ROS_DOMAIN_ID=$STEVE_DOMAIN_ID
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=$STEVE_RMW

# --- 3. source ROS (only if not already on PATH) --------------------------
[ -z "${ROS_DISTRO:-}" ] && [ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ -f /home/ws/install/setup.bash ] && source /home/ws/install/setup.bash

# --- 4. reset the daemon so it picks up the new domain/env ----------------
timeout 5 ros2 daemon stop >/dev/null 2>&1 || true

echo "----------------------------------------------------------------"
_steve_status
echo "----------------------------------------------------------------"
echo "checks (run separately): ros2 topic list"
echo "                         timeout 10 ros2 topic echo /joint_states sensor_msgs/msg/JointState --once"
