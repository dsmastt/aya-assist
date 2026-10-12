"""Command classification policy for the assistant's shell executor.

Layered analyzer: normalization -> quote-aware segmentation -> per-segment
rules -> path policy -> merged verdict. Pure string analysis; never executes.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Action(Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    DENY = "deny"


_SEVERITY = {Action.ALLOW: 0, Action.APPROVE: 1, Action.DENY: 2}

READ_ONLY = "read-only"
SAFE_WRITE = "safe-write"
NETWORK = "network"
APP_LAUNCH = "app-launch"
WORKSPACE = "workspace"
SYSTEM_CONFIG = "system-config"
DANGEROUS = "dangerous"
FORBIDDEN = "forbidden"
SHELL = "shell"


@dataclass
class Verdict:
    action: Action = Action.ALLOW
    category: str = SHELL
    reasons: list = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        return f"{self.action.value} [{self.category}] {self.summary}"


def _merge(a: Verdict, b: Verdict) -> Verdict:
    """Worst verdict wins; on ties the more specific category wins."""
    if _SEVERITY[b.action] > _SEVERITY[a.action]:
        a, b = b, a
    elif _SEVERITY[b.action] == _SEVERITY[a.action]:
        if a.category == SHELL and b.category != SHELL:
            a, b = b, a
    reasons = list(dict.fromkeys([*a.reasons, *b.reasons]))[:5]
    return Verdict(a.action, a.category, reasons, a.summary or b.summary)


DEFAULT_WRITE_ROOTS = ["~", "/tmp", "/var/tmp", "/mnt"]
DEFAULT_APPROVE_PATHS = ["~/.config", "~/.local/share"]
DEFAULT_FORBIDDEN_PATHS = [
    "/etc", "/usr", "/boot", "/sys", "/proc", "/run", "/var/lib",
    "/var/log", "/opt", "~/.ssh", "~/.gnupg",
    "~/.config/hypr", "~/.config/caelestia",
]
DEFAULT_LAUNCHERS = {
    "codium", "code", "firefox", "zen", "zen-bin", "foot", "kitty",
    "thunar", "dolphin", "mpv", "vlc", "btop", "htop", "nvtop",
    "discord", "spotify", "steam", "gimp", "krita", "blender",
    "qbittorrent", "xdg-open", "ani-cli", "notify-send", "yt-dlp",
    "playerctl", "fzf", "rg", "git", "python", "python3", "uv", "uvx",
    "pip", "cargo", "go", "node", "npm", "java", "feh", "imv", "micro",
    "vim", "nvim", "docker",
}

_ELEVATION = {"sudo", "su", "pkexec", "doas", "sudoedit", "runuser"}
_ELEVATION_SAFE_ARGV0 = {
    "echo", "printf", "grep", "rg", "cat", "which", "whereis", "type",
    "man", "info", "help", "ls", "find",
}
_FS_TOOLS = {
    "mkfs", "mkfs.ext2", "mkfs.ext3", "mkfs.ext4", "mkfs.btrfs",
    "mkfs.fat", "mkfs.vfat", "mkfs.ntfs", "mkfs.xfs", "fdisk", "sfdisk",
    "cfdisk", "parted", "gdisk", "sgdisk", "wipefs",
}
_SECURE_ERASE = {"shred", "wipe", "srm"}
_KILL_TOOLS = {"pkill", "killall", "kill", "xkill"}
_POWER = {"shutdown", "reboot", "poweroff", "halt", "telinit", "init"}
_PKG = {"pacman", "yay", "paru", "apt", "apt-get", "dnf", "zypper",
        "flatpak", "snap", "nix", "emerge"}
_WRAPPERS = {"env", "command", "nohup", "setsid", "time", "nice",
             "ionice", "stdbuf", "exec"}
_SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh", "csh"}
_READ_ONLY = {
    "ls", "cat", "head", "tail", "less", "more", "grep", "rg", "egrep",
    "fgrep", "locate", "which", "whereis", "type", "pwd", "echo",
    "printf", "test", "[", "true", "false", "date", "cal", "who",
    "whoami", "id", "uname", "uptime", "free", "df", "du", "ps", "pgrep",
    "top", "htop", "btop", "nvtop", "lsblk", "lspci", "lsusb", "lsmod",
    "ip", "ss", "host", "dig", "nslookup", "file", "stat", "wc", "sort",
    "uniq", "cut", "tr", "diff", "cmp", "md5sum", "sha1sum", "sha256sum",
    "sha512sum", "journalctl", "dmesg", "nvidia-smi", "sensors",
    "notify-send", "caelestia", "hyprctl", "playerctl", "man", "info",
    "help", "bat", "eza", "exa", "tree", "zoxide", "starship",
}
_DEV_OK = {"/dev/null", "/dev/zero", "/dev/random", "/dev/urandom",
           "/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/fd"}
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _split_segments(command: str) -> list:
    """Split on shell connectors while respecting quotes and backslashes."""
    segments, cur, quote = [], [], None
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote:
            cur.append(c)
            if c == quote and (i == 0 or command[i - 1] != "\\"):
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
            cur.append(c)
            i += 1
            continue
        if c == "\\":
            cur.append(c)
            if i + 1 < n:
                cur.append(command[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c in "&|;":
            nxt = command[i + 1] if i + 1 < n else ""
            if (c == "&" and nxt == "&") or (c == "|" and nxt == "|"):
                i += 2
            else:
                i += 1
            seg = "".join(cur).strip()
            if seg:
                segments.append(seg)
            cur = []
            continue
        if c == "\n":
            i += 1
            seg = "".join(cur).strip()
            if seg:
                segments.append(seg)
            cur = []
            continue
        cur.append(c)
        i += 1
    seg = "".join(cur).strip()
    if seg:
        segments.append(seg)
    return segments


def _unquoted_view(segment: str) -> str:
    """Return the segment with quoted spans blanked out."""
    out = list(segment)
    quote = None
    for i, c in enumerate(segment):
        if quote:
            if c == quote and (i == 0 or segment[i - 1] != "\\"):
                quote = None
            out[i] = " "
        elif c in "\"'":
            quote = c
            out[i] = " "
    return "".join(out)


def _redirect_targets(segment: str) -> list:
    """Extract write-redirect targets (>, >>, 2>, &> ...) from a segment."""
    view = _unquoted_view(segment)
    targets = []
    i, n = 0, len(view)
    while i < n:
        if view[i] != ">":
            i += 1
            continue
        start = i
        if i > 0 and view[i - 1] == "&":
            start = i - 1
        elif i > 0 and view[i - 1].isdigit():
            start = i - 1
        before = view[start - 1] if start > 0 else " "
        if before.isalnum() or before in "-_.":
            i += 1
            continue
        end = i + 1
        if end < n and view[end] == ">":
            end += 1
        j = end
        while j < n and view[j] in " \t":
            j += 1
        tok = []
        while j < n and view[j] not in " \t;&|<>":
            tok.append(view[j])
            j += 1
        if tok:
            targets.append("".join(tok))
        i = j
    return targets


class SafetyPolicy:
    """Rule engine classifying shell command strings into three tiers."""

    def __init__(self, *, allowed_write_roots=None, approve_write_paths=None,
                 forbidden_paths=None, allowed_launchers=None):
        self.home = os.path.expanduser("~")
        self.write_roots = [self._expand(p) for p in
                            (allowed_write_roots or DEFAULT_WRITE_ROOTS)]
        self.approve_paths = [self._expand(p) for p in
                              (approve_write_paths or DEFAULT_APPROVE_PATHS)]
        self.forbidden_paths = [self._expand(p) for p in
                                (forbidden_paths or DEFAULT_FORBIDDEN_PATHS)]
        self.launchers = (set(allowed_launchers) if allowed_launchers is not None
                          else set(DEFAULT_LAUNCHERS))

    def _expand(self, p: str) -> str:
        p = os.path.expanduser(p)
        if p.startswith("$HOME"):
            p = self.home + p[len("$HOME"):]
        return os.path.normpath(p)

    def _norm(self, target: str) -> str:
        p = target.strip().strip("\"'")
        p = os.path.expanduser(p)
        if p.startswith("$HOME"):
            p = self.home + p[len("$HOME"):]
        if not p.startswith("/"):
            p = os.path.join(self.home, p)
        return os.path.normpath(p)

    # -- public -----------------------------------------------------------
    def classify(self, command: str) -> Verdict:
        command = (command or "").strip()
        if not command:
            return Verdict(Action.ALLOW, READ_ONLY, [], "empty command")
        worst = Verdict(Action.ALLOW, SHELL, [], "no rules matched")
        for seg in _split_segments(command):
            worst = _merge(worst, self._classify_segment(seg))
        return worst

    # -- per-segment analysis ---------------------------------------------
    def _classify_segment(self, segment: str) -> Verdict:
        seg = segment.strip()
        if not seg:
            return Verdict(Action.ALLOW, READ_ONLY, [], "empty segment")
        try:
            tokens = shlex.split(seg, posix=True)
        except ValueError:
            return Verdict(Action.ALLOW, SHELL,
                           ["unbalanced quoting; analyzed opaquely"],
                           "unparseable segment")
        argv = self._strip_prefix(tokens)
        if not argv:
            return Verdict(Action.ALLOW, READ_ONLY, [], "empty after prefix strip")

        argv0 = os.path.basename(argv[0]).lower()

        # 1) elevation — absolute denial
        if argv0 in _ELEVATION:
            return Verdict(Action.DENY, FORBIDDEN,
                           ["privilege escalation"], "elevation tool as command")
        if any(t in _ELEVATION for t in argv[1:]):
            if argv0 not in _ELEVATION_SAFE_ARGV0 and (
                    "-exec" in argv or argv0 in _WRAPPERS
                    or argv0 in _SHELLS or argv0 in ("python", "python3", "perl", "node")):
                return Verdict(Action.DENY, FORBIDDEN,
                               ["privilege escalation attempt"],
                               "elevation tool invoked indirectly")

        v = Verdict(Action.ALLOW, SHELL, [], "")

        # 2) redirection targets
        for target in _redirect_targets(seg):
            tv = self._path_verdict(target, "output redirect")
            if tv:
                v = _merge(v, tv)

        # 3) command-specific rules
        v = _merge(v, self._cmd_rule(argv0, argv))
        return v

    @staticmethod
    def _strip_prefix(tokens: list) -> list:
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if _ASSIGN_RE.match(t) or t in _WRAPPERS:
                i += 1
                continue
            break
        return tokens[i:]

    def _path_verdict(self, target: str, op: str) -> Optional[Verdict]:
        p = self._norm(target)
        if p.startswith("/dev/"):
            if p in _DEV_OK:
                return None
            return Verdict(Action.DENY, FORBIDDEN,
                           [f"{op} targeting {p}"], "block device write")
        for f in self.forbidden_paths:
            if p == f or p.startswith(f + "/"):
                return Verdict(Action.DENY, FORBIDDEN,
                               [f"{op} targeting protected path {p}"],
                               "protected path write")
        for a in self.approve_paths:
            if p == a or p.startswith(a + "/"):
                return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                               [f"{op} targeting {p} needs approval"],
                               "config path write")
        for r in self.write_roots:
            if p == r or p.startswith(r + "/"):
                return None
        return Verdict(Action.APPROVE, SAFE_WRITE,
                       [f"{op} targeting {p} outside allowed roots"],
                       "write outside allowed roots")

    def _critical(self, t: str) -> bool:
        raw = t.strip().strip("\"'")
        if raw in ("/", "/*", "~", "~/*", "$HOME", "$HOME/*",
                   ".", "./*", "..", "../", "/mnt", "/mnt/*"):
            return True
        p = self._norm(t)
        return p in ("/", self.home, "/mnt")

    # -- command rules ------------------------------------------------------
    def _cmd_rule(self, argv0: str, argv: list) -> Verdict:
        if argv0 in _FS_TOOLS or argv0.startswith("mkfs"):
            return Verdict(Action.DENY, FORBIDDEN,
                           [f"{argv0} is a filesystem tool"],
                           "filesystem/partition tool")
        if argv0 in _SECURE_ERASE:
            return Verdict(Action.DENY, FORBIDDEN,
                           [f"{argv0} is a secure-erase tool"],
                           "secure-erase tool")
        if argv0 in _KILL_TOOLS:
            return Verdict(Action.APPROVE, DANGEROUS,
                           [f"{argv0} signals processes"], "process signal")
        if argv0 in _POWER:
            return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                           [f"{argv0} controls system power"], "power control")
        if argv0 in _PKG:
            return self._rule_pkg(argv)
        if argv0 == "dd":
            return self._rule_dd(argv)
        if argv0 == "rm":
            return self._rule_rm(argv)
        if argv0 == "find":
            if "-delete" in argv or ("-exec" in argv and any("rm" in a for a in argv)):
                return Verdict(Action.APPROVE, DANGEROUS,
                               ["find with delete/exec"], "recursive deletion via find")
            return Verdict(Action.ALLOW, READ_ONLY, [], "read-only find")
        if argv0 == "git":
            return self._rule_git(argv)
        if argv0 == "systemctl":
            return self._rule_systemctl(argv)
        if argv0 == "hyprctl":
            if len(argv) > 1 and argv[1] == "dispatch":
                return Verdict(Action.ALLOW, WORKSPACE, [], "window manager control")
            return Verdict(Action.ALLOW, READ_ONLY, [], "window manager query")
        if argv0 == "docker":
            return self._rule_docker(argv)
        if argv0 in ("pip", "pip3"):
            return self._rule_pip(argv)
        if argv0 == "npm":
            return self._rule_npm(argv)
        if argv0 == "cargo":
            if "install" in argv:
                return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                               ["cargo install"], "cargo install")
            return Verdict(Action.ALLOW, SAFE_WRITE, [], "cargo build")
        if argv0 == "make":
            if "install" in argv:
                return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                               ["make install"], "make install")
            return Verdict(Action.ALLOW, SAFE_WRITE, [], "make build")
        if argv0 in ("chmod", "chown"):
            return self._rule_perm(argv, argv0)
        if argv0 in ("sed", "awk", "perl"):
            if "-i" in argv or "--in-place" in argv:
                for t in argv[1:]:
                    if t.startswith("-") or "=" in t:
                        continue
                    tv = self._path_verdict(t, f"{argv0} in-place edit")
                    if tv and tv.action != Action.ALLOW:
                        return tv
                return Verdict(Action.ALLOW, SAFE_WRITE, [],
                               "in-place edit in safe location")
            return Verdict(Action.ALLOW, READ_ONLY, [], "read-only text filter")
        if argv0 == "tar":
            if any(f in argv for f in ("-x", "--extract", "-c", "--create", "-r", "--append")):
                for t in argv[1:]:
                    if not t.startswith("-"):
                        tv = self._path_verdict(t, "archive operation")
                        if tv and tv.action != Action.ALLOW:
                            return tv
                return Verdict(Action.ALLOW, SAFE_WRITE, [],
                               "archive operation in safe location")
            return Verdict(Action.ALLOW, READ_ONLY, [], "archive listing")
        if argv0 in ("curl", "wget"):
            for i, t in enumerate(argv):
                if t in ("-o", "--output", "-O", "--output-document"):
                    if i + 1 < len(argv):
                        tv = self._path_verdict(argv[i + 1], "download output")
                        if tv:
                            return tv
            return Verdict(Action.ALLOW, NETWORK, [], "network fetch")
        if argv0 in ("cp", "mv", "install", "ln"):
            dests = [t for t in argv[1:] if not t.startswith("-") and "=" not in t]
            if dests:
                tv = self._path_verdict(dests[-1], f"{argv0} destination")
                if tv and tv.action != Action.ALLOW:
                    return tv
            return Verdict(Action.ALLOW, SAFE_WRITE, [], "file operation in safe location")
        if argv0 in ("touch", "mkdir", "rmdir", "truncate", "tee", "unzip", "gzip", "gunzip"):
            for t in argv[1:]:
                if t.startswith("-") or "=" in t:
                    continue
                tv = self._path_verdict(t, f"{argv0} target")
                if tv and tv.action != Action.ALLOW:
                    return tv
            return Verdict(Action.ALLOW, SAFE_WRITE, [], "file operation in safe location")
        if argv0 in _READ_ONLY:
            return Verdict(Action.ALLOW, READ_ONLY, [], "read-only command")
        if argv0 in self.launchers or shutil.which(argv0):
            if argv0 in self.launchers:
                return Verdict(Action.ALLOW, APP_LAUNCH, [], "known application launcher")
            return Verdict(Action.ALLOW, SHELL, [], "known executable")
        return Verdict(Action.ALLOW, SHELL, ["unclassified command"],
                       "unclassified — runs with timeout/output caps")

    def _rule_rm(self, argv: list) -> Verdict:
        flags = [t for t in argv[1:] if t.startswith("-")]
        short_chars = "".join(
            f[1:] for f in flags
            if f.startswith("-") and not f.startswith("--") and len(f) > 1
        )
        recursive = ("r" in short_chars or "R" in short_chars
                     or any(f.startswith("--recursive") for f in flags))
        targets = [t for t in argv[1:] if not t.startswith("-") and t != "--"]
        if not targets:
            return Verdict(Action.APPROVE, DANGEROUS,
                           ["deletion without visible targets"],
                           "deletion with unknown targets")
        for t in targets:
            if self._critical(t):
                return Verdict(Action.DENY, FORBIDDEN,
                               [f"recursive deletion of critical path {t}"],
                               "critical-path deletion")
        if recursive:
            return Verdict(Action.APPROVE, DANGEROUS,
                           ["recursive deletion of specific paths"],
                           "recursive deletion")
        for t in targets:
            tv = self._path_verdict(t, "rm")
            if tv:
                if tv.action == Action.DENY:
                    return tv
                return Verdict(Action.APPROVE, tv.category, tv.reasons,
                               "deletion needs approval")
        return Verdict(Action.ALLOW, SAFE_WRITE, [], "file deletion in safe location")

    def _rule_dd(self, argv: list) -> Verdict:
        out = None
        for t in argv[1:]:
            if t.startswith("of="):
                out = t[3:]
        if out is None:
            return Verdict(Action.APPROVE, DANGEROUS,
                           ["low-level write without explicit output"],
                           "low-level write")
        if out.startswith("/dev/"):
            return Verdict(Action.DENY, FORBIDDEN,
                           [f"low-level write to {out}"], "device write")
        return Verdict(Action.APPROVE, DANGEROUS,
                       [f"low-level write to {out}"], "low-level write")

    def _rule_pkg(self, argv: list) -> Verdict:
        if len(argv) > 1 and re.match(r"^-[A-Za-z]*[SUR]", argv[1]):
            return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                           ["package install/remove/upgrade"],
                           "package management")
        if argv[1:2] and argv[1] in ("install", "remove", "upgrade", "uninstall"):
            return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                           ["package install/remove/upgrade"],
                           "package management")
        return Verdict(Action.ALLOW, READ_ONLY, [], "package query")

    def _rule_systemctl(self, argv: list) -> Verdict:
        sub = argv[1] if len(argv) > 1 else ""
        if sub in ("start", "stop", "restart", "reload", "mask", "unmask",
                   "enable", "disable", "poweroff", "reboot", "suspend",
                   "hibernate", "kexec", "halt", "isolate", "set-default",
                   "daemon-reload", "switch-root"):
            return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                           [f"systemctl {sub}"], "service/power control")
        return Verdict(Action.ALLOW, READ_ONLY, [], "systemctl query")

    def _rule_git(self, argv: list) -> Verdict:
        sub = argv[1] if len(argv) > 1 else "help"
        if sub == "reset" and "--hard" in argv:
            return Verdict(Action.APPROVE, DANGEROUS, ["git reset --hard"],
                           "history rewrite")
        if sub == "clean" and any(f in ("-f", "-d", "-x", "-fd", "-fdx", "--force")
                                  for f in argv[2:]):
            return Verdict(Action.APPROVE, DANGEROUS, ["git clean with force"],
                           "working tree wipe")
        if sub == "push" and "--force" in argv:
            return Verdict(Action.APPROVE, DANGEROUS, ["git push --force"],
                           "force push")
        if sub == "branch" and "-D" in argv:
            return Verdict(Action.APPROVE, DANGEROUS, ["git branch -D"],
                           "branch deletion")
        if sub in ("clone", "fetch", "pull", "push", "remote"):
            return Verdict(Action.ALLOW, NETWORK, [], "git network operation")
        if sub in ("status", "log", "diff", "show", "branch", "tag", "blame", "stash"):
            return Verdict(Action.ALLOW, READ_ONLY, [], "git inspection")
        if sub in ("add", "commit", "mv", "rm", "checkout", "switch", "merge",
                   "rebase", "init", "apply", "cherry-pick", "restore"):
            return Verdict(Action.ALLOW, SAFE_WRITE, [], "git working-tree operation")
        return Verdict(Action.ALLOW, SHELL, [], "git operation")

    def _rule_docker(self, argv: list) -> Verdict:
        sub = argv[1] if len(argv) > 1 else "ps"
        if sub in ("rm", "rmi", "prune", "system"):
            return Verdict(Action.APPROVE, DANGEROUS, [f"docker {sub}"],
                           "container/image removal")
        if sub in ("ps", "images", "logs", "inspect", "version", "stats"):
            return Verdict(Action.ALLOW, READ_ONLY, [], "docker inspection")
        return Verdict(Action.ALLOW, SHELL, [], "docker operation")

    def _rule_pip(self, argv: list) -> Verdict:
        if "install" in argv or "uninstall" in argv:
            if "--user" in argv:
                return Verdict(Action.ALLOW, SAFE_WRITE, [],
                               "user-scope pip install")
            return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                           ["pip install/uninstall"], "pip package change")
        return Verdict(Action.ALLOW, READ_ONLY, [], "pip query")

    def _rule_npm(self, argv: list) -> Verdict:
        if "install" in argv or "uninstall" in argv:
            if "-g" in argv or "--global" in argv:
                return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                               ["global npm install"], "global npm change")
            return Verdict(Action.ALLOW, SAFE_WRITE, [], "project npm install")
        return Verdict(Action.ALLOW, SHELL, [], "npm operation")

    def _rule_perm(self, argv: list, argv0: str) -> Verdict:
        targets = [t for t in argv[1:] if not t.startswith("-") and "=" not in t]
        mode = next((t for t in argv[1:]
                     if re.match(r"^[ugoa]*[+-=][rwxXst]*$", t)
                     or re.match(r"^[0-7]{3,4}$", t)), "")
        recursive = "-R" in argv or "--recursive" in argv
        if (recursive and mode.lstrip("0") in ("777", "000", "755")
                and any(self._norm(t) in ("/", "/usr", "/etc", self.home)
                        for t in targets)):
            return Verdict(Action.DENY, FORBIDDEN,
                           ["recursive permission reset on a system root"],
                           "permission reset on system root")
        for t in targets:
            tv = self._path_verdict(t, f"{argv0} target")
            if tv and tv.action == Action.DENY:
                return tv
        return Verdict(Action.APPROVE, SYSTEM_CONFIG,
                       [f"{argv0} on user files"], "permission change")


_default_policy: Optional[SafetyPolicy] = None


def default_policy() -> SafetyPolicy:
    global _default_policy
    if _default_policy is None:
        _default_policy = SafetyPolicy()
    return _default_policy


def classify(command, policy=None) -> Verdict:
    return (policy or default_policy()).classify(command)


def is_destructive(command, policy=None) -> bool:
    v = classify(command, policy)
    return v.action in (Action.APPROVE, Action.DENY)
