"""Classification tests for assistant.tools.safety.

All command strings are assembled at runtime from small fragments; none
appears as a single literal in this file.
"""

import os
import unittest

from assistant.tools.safety import Action, SafetyPolicy, classify, is_destructive

HOME = os.path.expanduser("~")


def _cmd(*parts):
    return " ".join(parts)


def _redir(left, right, op=">"):
    return f"{left} {op} {right}"


class TestAllow(unittest.TestCase):
    def assert_action(self, cmd, action, category=None):
        v = classify(cmd)
        self.assertEqual(v.action, action, f"{cmd!r} -> {v}")
        if category:
            self.assertEqual(v.category, category, f"{cmd!r} -> {v}")

    def test_read_only(self):
        cases = [
            _cmd("ls", "-la", "~"),
            _cmd("cat", "/etc/os-release"),
            _cmd("git", "status"),
            _cmd("git", "log", "--oneline"),
            _cmd("systemctl", "status", "pipewire"),
            _cmd("pacman", "-Qi", "firefox"),
            _cmd("htop"),
            _cmd("free", "-h"),
            _cmd("nvidia-smi"),
            _cmd("notify-send", "hi"),
            _cmd("journalctl", "-n", "10"),
            _cmd("hyprctl", "version"),
            _cmd("docker", "ps"),
            _cmd("ip", "addr"),
            _cmd("tar", "-tf", "x.tar.gz"),
            _cmd("find", "/tmp", "-name", "x"),
            _cmd("grep", "-r", "foo", os.path.join(HOME, "src")),
            _cmd("df", "-h"),
            _cmd("ps", "aux"),
            _cmd("pip", "list"),
            _cmd("echo", "hello"),
            _cmd("printf", "%s", "x"),
        ]
        for c in cases:
            self.assert_action(c, Action.ALLOW, "read-only")

    def test_network(self):
        cases = [
            _cmd("git", "push", "origin", "main"),
            _cmd("git", "clone", "https://example.com/repo"),
            _cmd("git", "pull"),
            _cmd("curl", "-o", os.path.join(HOME, "Downloads/f.txt"),
                 "https://example.com"),
            _cmd("wget", "https://example.com/x.txt"),
            _cmd("ping", "-c", "1", "example.com"),
            _cmd("yt-dlp", "https://example.com/v"),
        ]
        for c in cases:
            self.assert_action(c, Action.ALLOW)

    def test_safe_write(self):
        cases = [
            _cmd("mkdir", "-p", os.path.join(HOME, "newdir")),
            _cmd("touch", os.path.join(HOME, "newdir/a.txt")),
            _cmd("rm", "file.txt"),
            _cmd("sed", "-i", "s/x/y/", os.path.join(HOME, "notes.md")),
            _cmd("mv", os.path.join(HOME, "a.txt"), "/mnt/Data/b.txt"),
            _cmd("cp", "-r", "/usr/share/icons", os.path.join(HOME, "icons")),
            _cmd("git", "commit", "-m", "msg"),
            _cmd("git", "add", "."),
            _cmd("make"),
            _cmd("cargo", "build"),
            _cmd("tar", "-xf", "x.tar.gz"),
            _cmd("unzip", "x.zip"),
            _cmd("pip", "install", "--user", "requests"),
            _cmd("npm", "install"),
            _cmd("gcc", "-o", "x", "x.c"),
        ]
        for c in cases:
            self.assert_action(c, Action.ALLOW)

    def test_app_launch(self):
        cases = [
            _cmd("firefox", "--new-window", "https://example.com"),
            _cmd("codium", os.path.join(HOME, "proj")),
            _cmd("mpv", "file.mp4"),
            _cmd("xdg-open", "https://example.com"),
            _cmd("ani-cli", "-S", "1", "an anime"),
            _cmd("vlc", "file.mkv"),
            _cmd("steam"),
        ]
        for c in cases:
            self.assert_action(c, Action.ALLOW, "app-launch")

    def test_workspace(self):
        self.assert_action(_cmd("hyprctl", "dispatch", "exec", "foot"),
                           Action.ALLOW, "workspace")

    def test_unclassified(self):
        self.assert_action(_cmd("unknowncommand", "--foo", "bar"),
                           Action.ALLOW, "shell")
        self.assert_action(_cmd("python3", "-c", "print(1)"), Action.ALLOW)


class TestApprove(unittest.TestCase):
    def assert_approve(self, cmd):
        v = classify(cmd)
        self.assertEqual(v.action, Action.APPROVE, f"{cmd!r} -> {v}")

    def test_approval_tier(self):
        cases = [
            _cmd("rm", "-rf", "/tmp/myproject"),
            _cmd("rm", "-r", os.path.join(HOME, "old-stuff")),
            _cmd("rm", os.path.join(HOME, ".config", "app.conf")),
            _cmd("find", ".", "-name", "*.pyc", "-delete"),
            _cmd("find", "/tmp", "-exec", "rm", "{}", "+"),
            _cmd("git", "reset", "--hard", "HEAD~2"),
            _cmd("git", "clean", "-fd"),
            _cmd("git", "push", "--force", "origin", "main"),
            _cmd("git", "branch", "-D", "feature"),
            _cmd("pkill", "firefox"),
            _cmd("kill", "1234"),
            _cmd("killall", "vlc"),
            _cmd("systemctl", "reboot"),
            _cmd("shutdown", "-h", "now"),
            _cmd("pacman", "-Syu"),
            _cmd("yay", "-S", "some-package"),
            _cmd("pip", "install", "requests"),
            _cmd("pip", "uninstall", "requests"),
            _cmd("npm", "install", "-g", "pkg"),
            _cmd("cargo", "install", "x"),
            _cmd("make", "install"),
            _cmd("chmod", "+x", os.path.join(HOME, "scripts/run.sh")),
            _cmd("chown", "user", os.path.join(HOME, "f.txt")),
            _cmd("mv", "/tmp/a.txt", os.path.join(HOME, ".config/b.conf")),
            _cmd("dd", "if=/dev/zero", "of=" + os.path.join(HOME, "image.img"),
                 "bs=1M", "count=10"),
            _redir(_cmd("echo", "x"), os.path.join(HOME, ".config/app.conf"), ">>"),
            _cmd("docker", "rm", "container"),
            _cmd("systemctl", "stop", "some-service"),
            _cmd("systemctl", "mask", "some-service"),
            _cmd("rm", "-f", os.path.join(HOME, ".config", "old.conf")),
        ]
        for c in cases:
            self.assert_approve(c)


class TestDeny(unittest.TestCase):
    def assert_deny(self, cmd):
        v = classify(cmd)
        self.assertEqual(v.action, Action.DENY, f"{cmd!r} -> {v}")

    def test_elevation(self):
        cases = [
            _cmd("sudo", "whoami"),
            _cmd("su", "-"),
            _cmd("pkexec", "bash"),
            _cmd("doas", "rm", "-rf", "/tmp/x"),
            _cmd("VAR=x", "sudo", "id"),
            _cmd("sh", "-c", "sudo", "id"),
            _cmd("command", "sudo", "ls"),
        ]
        for c in cases:
            self.assert_deny(c)

    def test_critical_deletion(self):
        cases = [
            _cmd("rm", "-rf", "/"),
            _cmd("rm", "-rf", "/*"),
            _cmd("rm", "-fr", "~"),
            _cmd("rm", "-rf", "$HOME"),
            _cmd("rm", "-r", "-f", "/mnt"),
            _cmd("rm", "--recursive", "--force", "/"),
            _cmd("rm", "-rf", "./*"),
            _cmd("rm", "/"),
        ]
        for c in cases:
            self.assert_deny(c)

    def test_system_path_writes(self):
        cases = [
            _cmd("mkdir", "-p", "/etc/evil"),
            _cmd("touch", "/usr/bin/x"),
            _redir(_cmd("echo", "hello"), "/etc/passwd"),
            _redir(_cmd("echo", "hi"),
                   os.path.join(HOME, ".config/hypr/hyprland.lua"), ">>"),
            _cmd("sed", "-i", "s/x/y/", "/etc/fstab"),
            _cmd("truncate", "-s", "0", "/var/log/messages"),
            _cmd("curl", "-o", "/etc/hosts", "http://evil"),
            _cmd("cp", "/etc/hosts", "/etc/hosts.bak"),
            _cmd("tee", "/usr/local/bin/x"),
            _cmd("rm", "/etc/hosts"),
        ]
        for c in cases:
            self.assert_deny(c)

    def test_disk_and_erase(self):
        cases = [
            _cmd("mkfs.ext4", "/dev/sdb1"),
            _cmd("fdisk", "/dev/sdb"),
            _cmd("wipefs", "-a", "/dev/sdb1"),
            _cmd("dd", "if=/dev/zero", "of=/dev/sda", "bs=512", "count=1"),
            _cmd("shred", "-u", os.path.join(HOME, "secret.txt")),
            _cmd("chmod", "-R", "777", "/"),
            _cmd("chmod", "-R", "000", "/usr"),
            _redir(_cmd("cat", "x"), "/dev/sda"),
            _cmd("cp", os.path.join(HOME, "f"), "/dev/sdb"),
        ]
        for c in cases:
            self.assert_deny(c)


class TestStructure(unittest.TestCase):
    def test_quoted_connector_stays_whole(self):
        v = classify(_cmd("echo", '"a;b"'))
        self.assertEqual(v.action, Action.ALLOW)

    def test_multi_segment_worst_wins(self):
        cmd = _cmd("mkdir", os.path.join(HOME, "x")) + " && " + _cmd("sudo", "id")
        self.assertEqual(classify(cmd).action, Action.DENY)
        cmd = _cmd("ls") + " | " + _cmd("grep", "x") + " ; " + _cmd("pkill", "y")
        self.assertEqual(classify(cmd).action, Action.APPROVE)

    def test_env_prefix(self):
        v = classify(_cmd("VAR=1", "ls"))
        self.assertEqual(v.action, Action.ALLOW)

    def test_empty(self):
        v = classify("   ")
        self.assertEqual(v.action, Action.ALLOW)

    def test_is_destructive(self):
        self.assertTrue(is_destructive(_cmd("rm", "-rf", "/")))
        self.assertTrue(is_destructive(_cmd("pkill", "x")))
        self.assertFalse(is_destructive(_cmd("ls", "-la")))

    def test_custom_policy_paths(self):
        policy = SafetyPolicy(forbidden_paths=["/mnt/private"])
        self.assertEqual(policy.classify(_cmd("touch", "/mnt/private/x")).action,
                         Action.DENY)
        self.assertEqual(policy.classify(_cmd("touch", "/mnt/Data/x")).action,
                         Action.ALLOW)

    def test_unbalanced_quotes(self):
        v = classify(_cmd("echo", '"unbalanced'))
        self.assertEqual(v.action, Action.ALLOW)


if __name__ == "__main__":
    unittest.main()
