#!/usr/bin/env bash
# Allow the laptop to set the robot clock over ssh (stepmap sync-clock): only /bin/date, no password.
# The Jetson has no RTC battery; without internet it starts at 1970. Run once ON THE ROBOT, by the user:
#   sudo bash ~/stepmap_live/stepmap/robot/setup_clock_sync.sh
# Undo: sudo rm /etc/sudoers.d/stepmap-date
set -e
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
USER_NAME=${SUDO_USER:-unitree}
F=/etc/sudoers.d/stepmap-date
TMP=$(mktemp)
echo "$USER_NAME ALL=(root) NOPASSWD: /bin/date" > "$TMP"
visudo -cf "$TMP"                       # refuse a broken sudoers file
install -m 0440 -o root -g root "$TMP" "$F"
rm -f "$TMP"
echo "ok: $USER_NAME may run 'sudo -n /bin/date' ($F)"
# NTP would fight the laptop time when internet comes back; it is harmless (both are correct then).
