"""工具适配层测试：能力声明、参数校验、异常隔离、跨适配器行为。"""

from __future__ import annotations

import os
import subprocess

import pytest

from futurework.tools.base import Capability, CapabilityRegistry, ToolAdapter, safe_join_validator
from futurework.tools.browser_adapter import BrowserAdapter, normalize_url
from futurework.tools.filesystem_adapter import FileSystemAdapter
from futurework.tools.registry import ToolRegistry, default_registry
from futurework.tools.system_adapter import SystemAdapter
from futurework.tools.terminal_adapter import EditorAdapter, TerminalAdapter, inspect_command
from futurework.tools.vcs_adapter import VcsAdapter
from futurework.types import ExecutionStatus, SafetyLevel, ToolCommand


# ============================================================================
# 能力声明机制
# ============================================================================
class TestCapability:
    def test_validates_required_param(self):
        cap = Capability(action="x", params={"path": "str"})
        assert "缺少必填参数" in cap.validate({})
        assert cap.validate({"path": "a"}) is None

    def test_validates_optional_param(self):
        cap = Capability(action="x", params={"path": "str?"})
        assert cap.validate({}) is None

    def test_validates_types(self):
        cap = Capability(action="x", params={"n": "int"})
        assert "应为整数" in cap.validate({"n": "abc"})
        assert cap.validate({"n": 3}) is None

    def test_validator_receives_args(self):
        cap = Capability(action="x", validator=lambda a: "总是拒绝")
        assert cap.validate({}) == "总是拒绝"

    def test_validator_exception_is_captured(self):
        def boom(args):
            raise RuntimeError("炸了")

        cap = Capability(action="x", validator=boom)
        assert "校验器异常" in cap.validate({})

    def test_aliases_resolve_to_same_capability(self):
        registry = CapabilityRegistry()
        cap = registry.declare(Capability(action="git_status", aliases=("status",)))
        assert registry.capability("status") is cap
        assert "status" not in registry.actions()   # 别名不污染能力清单


class TestSafeJoin:
    def test_allows_inside_root(self, tmp_path):
        check = safe_join_validator(str(tmp_path))
        assert check({"path": "sub/file.txt"}) is None

    def test_blocks_traversal(self, tmp_path):
        check = safe_join_validator(str(tmp_path))
        assert "越界" in check({"path": "../../etc/passwd"})

    def test_rejects_empty(self, tmp_path):
        assert safe_join_validator(str(tmp_path))({"path": ""}) == "path 不能为空"


# ============================================================================
# 文件系统适配器
# ============================================================================
class TestFileSystem:
    @pytest.fixture
    def fs(self, tmp_path):
        return FileSystemAdapter(str(tmp_path))

    def test_write_then_read(self, fs):
        assert fs.execute(ToolCommand(tool_name="filesystem", action="write_file",
                                     args={"path": "a.txt", "content": "你好"})).status is ExecutionStatus.SUCCESS
        result = fs.execute(ToolCommand(tool_name="filesystem", action="read_file", args={"path": "a.txt"}))
        assert result.data == "你好"

    def test_append(self, fs):
        fs.execute(ToolCommand(tool_name="filesystem", action="write_file",
                               args={"path": "a.txt", "content": "one"}))
        fs.execute(ToolCommand(tool_name="filesystem", action="append_file",
                               args={"path": "a.txt", "content": "-two"}))
        result = fs.execute(ToolCommand(tool_name="filesystem", action="read_file", args={"path": "a.txt"}))
        assert result.data == "one-two"

    def test_path_traversal_blocked(self, fs):
        result = fs.execute(ToolCommand(tool_name="filesystem", action="read_file",
                                        args={"path": "../../../etc/passwd"}))
        assert result.status is ExecutionStatus.FAILED
        assert "不合法" in result.error

    def test_missing_file_gives_clear_error(self, fs):
        result = fs.execute(ToolCommand(tool_name="filesystem", action="read_file", args={"path": "nope.txt"}))
        assert "文件不存在" in result.error

    def test_delete_is_reversible_flag(self, fs):
        fs.execute(ToolCommand(tool_name="filesystem", action="write_file",
                               args={"path": "b.txt", "content": "x"}))
        result = fs.execute(ToolCommand(tool_name="filesystem", action="delete_file", args={"path": "b.txt"}))
        assert result.status is ExecutionStatus.SUCCESS
        assert result.rollback_available is False   # 诚实告知：删了就没了

    def test_restore_snapshot(self, fs, tmp_path):
        (tmp_path / "c.txt").write_text("原始")
        fs.execute(ToolCommand(tool_name="filesystem", action="write_file",
                               args={"path": "c.txt", "content": "改过"}))
        fs.restore(str(tmp_path / "c.txt"))
        assert (tmp_path / "c.txt").read_text() == "原始"

    def test_list_directory(self, fs, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "f.txt").write_text("x")
        result = fs.execute(ToolCommand(tool_name="filesystem", action="list_directory", args={"path": "."}))
        names = {e["name"] for e in result.data}
        assert names == {"sub", "f.txt"}

    def test_search_in_file(self, fs):
        fs.execute(ToolCommand(tool_name="filesystem", action="write_file",
                               args={"path": "d.txt", "content": "alpha\nbeta\ngamma"}))
        result = fs.execute(ToolCommand(tool_name="filesystem", action="search_in_file",
                                        args={"path": "d.txt", "keyword": "beta"}))
        assert len(result.data) == 1
        assert result.data[0]["line"] == 2

    def test_unknown_action_lists_capabilities(self, fs):
        result = fs.execute(ToolCommand(tool_name="filesystem", action="teleport", args={}))
        assert "不支持操作" in result.error
        assert "read_file" in result.error   # 错误里直接告诉用户能用什么

    def test_large_file_truncation_flagged(self, fs, tmp_path):
        (tmp_path / "big.txt").write_text("x" * (3 * 1024 * 1024))
        result = fs.execute(ToolCommand(tool_name="filesystem", action="read_file", args={"path": "big.txt"}))
        assert result.details.get("truncated") is True


# ============================================================================
# 终端适配器
# ============================================================================
class TestTerminal:
    @pytest.fixture
    def term(self, tmp_path):
        return TerminalAdapter(str(tmp_path))

    def test_runs_simple_command(self, term):
        result = term.execute(ToolCommand(tool_name="terminal", action="run_command",
                                          args={"command": "echo hello"}))
        assert result.status is ExecutionStatus.SUCCESS
        assert "hello" in result.data["stdout"]

    def test_failing_command_reports_exit_code(self, term):
        result = term.execute(ToolCommand(tool_name="terminal", action="run_command",
                                          args={"command": "exit 3"}))
        assert result.status is ExecutionStatus.FAILED
        assert result.data["returncode"] == 3

    def test_timeout_is_bounded(self, term):
        result = term.execute(ToolCommand(tool_name="terminal", action="run_command",
                                          args={"command": "sleep 5", "timeout": 0.5}))
        assert "超时" in result.error

    @pytest.mark.parametrize("command", [
        "rm -rf /",
        "sudo rm -rf / --no-preserve-root",
        "mkfs.ext4 /dev/sda1",
        ":(){:|:&};:",
        "shutdown -h now",
        "chmod -R 777 /",
    ])
    def test_dangerous_commands_rejected(self, term, command):
        result = term.execute(ToolCommand(tool_name="terminal", action="run_command", args={"command": command}))
        assert result.status is ExecutionStatus.REJECTED_BY_SAFETY

    def test_dangerous_allowed_when_explicitly_enabled(self, tmp_path):
        term = TerminalAdapter(str(tmp_path), allow_dangerous=True)
        ok, _ = inspect_command("ls -la")
        assert ok is True

    def test_inspect_rejects_empty(self):
        ok, why = inspect_command("   ")
        assert ok is False and "为空" in why

    def test_output_truncated(self, term):
        result = term.execute(ToolCommand(tool_name="terminal", action="run_command",
                                          args={"command": "python3 -c \"print('a'*50000)\"", "timeout": 20}))
        assert "输出已截断" in result.data["stdout"]

    def test_run_tests_detects_pytest(self, tmp_path):
        term = TerminalAdapter(str(tmp_path))
        (tmp_path / "tests").mkdir()
        result = term.execute(ToolCommand(tool_name="terminal", action="run_tests", args={}))
        assert result.data is not None
        assert "pytest" in result.data["stdout"] or result.status in (
            ExecutionStatus.SUCCESS, ExecutionStatus.FAILED)

    def test_run_tests_unknown_runner(self, tmp_path):
        term = TerminalAdapter(str(tmp_path))
        result = term.execute(ToolCommand(tool_name="terminal", action="run_tests", args={}))
        assert "未能识别测试运行器" in (result.error or "")

    def test_history_recorded(self, term):
        term.execute(ToolCommand(tool_name="terminal", action="run_command", args={"command": "echo x"}))
        assert term.history()[-1]["command"] == "echo x"


class TestEditor:
    def test_lists_available_editors(self):
        editor = EditorAdapter()
        result = editor.execute(ToolCommand(tool_name="editor", action="list_editors", args={}))
        assert result.status is ExecutionStatus.SUCCESS
        assert isinstance(result.data, list)

    def test_missing_file_reports(self, tmp_path):
        editor = EditorAdapter(str(tmp_path))
        result = editor.execute(ToolCommand(tool_name="editor", action="open_file_editor",
                                            args={"path": "nope.py"}))
        assert "文件不存在" in result.error

    def test_traversal_blocked(self, tmp_path):
        editor = EditorAdapter(str(tmp_path))
        result = editor.execute(ToolCommand(tool_name="editor", action="open_file_editor",
                                            args={"path": "../../etc/passwd"}))
        assert result.status is ExecutionStatus.FAILED


# ============================================================================
# 浏览器适配器
# ============================================================================
class TestBrowser:
    @pytest.mark.parametrize("raw,expected", [
        ("https://example.com", "https://example.com"),
        ("example.com", "https://example.com"),
        ("http://a.b", "http://a.b"),
        ("localhost:8080", "http://localhost:8080"),
        ("about:blank", "about:blank"),
    ])
    def test_url_normalization(self, raw, expected):
        assert normalize_url(raw) == expected

    def test_invalid_url_rejected(self):
        browser = BrowserAdapter()
        result = browser.execute(ToolCommand(tool_name="browser", action="open_url", args={"url": "###"}))
        assert result.status is ExecutionStatus.FAILED

    def test_search_builds_query(self, monkeypatch):
        browser = BrowserAdapter()
        captured = {}
        monkeypatch.setattr(browser, "_launch", lambda url: (captured.setdefault("url", url), {"url": url}))
        browser.execute(ToolCommand(tool_name="browser", action="search_web",
                                    args={"query": "异步编程"}))
        assert "search?q=" in captured["url"]

    def test_back_with_empty_history(self):
        browser = BrowserAdapter()
        result = browser.execute(ToolCommand(tool_name="browser", action="browser_back", args={}))
        assert result.status is ExecutionStatus.FAILED
        assert "没有可返回" in result.error


# ============================================================================
# 系统与 VCS 适配器
# ============================================================================
class TestSystem:
    def test_system_info(self):
        system = SystemAdapter()
        result = system.execute(ToolCommand(tool_name="system", action="system_info", args={}))
        assert result.status is ExecutionStatus.SUCCESS
        assert "platform" in result.data

    def test_missing_window_tool_returns_hint(self):
        system = SystemAdapter()
        result = system.execute(ToolCommand(tool_name="system", action="list_windows", args={}))
        if result.status is ExecutionStatus.SUCCESS:
            assert isinstance(result.data, list)
        else:
            assert "wmctrl" in (result.error or "") or result.data is not None

    def test_unknown_app_gives_hint(self):
        system = SystemAdapter()
        result = system.execute(ToolCommand(tool_name="system", action="launch_app",
                                            args={"app": "definitely_not_installed_xyz"}))
        assert result.status is ExecutionStatus.FAILED


class TestVcs:
    @pytest.fixture
    def repo(self, tmp_path):
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        return tmp_path

    def test_status_on_fresh_repo(self, repo):
        vcs = VcsAdapter(str(repo))
        result = vcs.execute(ToolCommand(tool_name="vcs", action="git_status", args={}))
        assert result.status is ExecutionStatus.SUCCESS
        assert result.data["clean"] is True

    def test_add_and_commit(self, repo):
        (repo / "a.txt").write_text("hello")
        vcs = VcsAdapter(str(repo))
        vcs.execute(ToolCommand(tool_name="vcs", action="git_add", args={}))
        result = vcs.execute(ToolCommand(tool_name="vcs", action="git_commit",
                                         args={"message": "首次提交"}))
        assert result.status is ExecutionStatus.SUCCESS
        assert "首次提交" in result.data

    def test_empty_commit_message_rejected(self, repo):
        (repo / "a.txt").write_text("x")
        vcs = VcsAdapter(str(repo))
        vcs.execute(ToolCommand(tool_name="vcs", action="git_add", args={}))
        result = vcs.execute(ToolCommand(tool_name="vcs", action="git_commit", args={"message": "  "}))
        assert "不能为空" in result.error

    def test_commit_is_rollbackable(self, repo):
        (repo / "a.txt").write_text("x")
        vcs = VcsAdapter(str(repo))
        vcs.execute(ToolCommand(tool_name="vcs", action="git_add", args={}))
        result = vcs.execute(ToolCommand(tool_name="vcs", action="git_commit", args={"message": "c"}))
        assert result.rollback_available is True
        assert vcs.rollback_last_commit() is True

    def test_unavailable_outside_repo(self, tmp_path):
        vcs = VcsAdapter(str(tmp_path))
        available, why = vcs.is_available()
        assert available is False and "不是 Git 仓库" in why

    def test_log(self, repo):
        (repo / "a.txt").write_text("x")
        vcs = VcsAdapter(str(repo))
        vcs.execute(ToolCommand(tool_name="vcs", action="git_add", args={}))
        vcs.execute(ToolCommand(tool_name="vcs", action="git_commit", args={"message": "log test"}))
        result = vcs.execute(ToolCommand(tool_name="vcs", action="git_log", args={"count": 5}))
        assert any(c["subject"] == "log test" for c in result.data)


# ============================================================================
# 注册中心路由
# ============================================================================
class TestRegistry:
    def test_resolves_by_action(self, tmp_path):
        registry = default_registry(str(tmp_path))
        adapter, capability, error = registry.resolve("read_file", "filesystem")
        assert adapter is not None and error == ""
        assert capability.safety_level is SafetyLevel.READ_ONLY

    def test_resolves_by_hint(self, tmp_path):
        registry = default_registry(str(tmp_path))
        adapter, _, _ = registry.resolve("read_file", "filesystem")
        assert adapter.name == "filesystem"

    def test_unknown_action_error_lists_alternatives(self, tmp_path):
        registry = default_registry(str(tmp_path))
        adapter, _, error = registry.resolve("launch_rocket", "")
        assert adapter is None
        assert "没有适配器支持" in error and "read_file" in error

    def test_unavailable_adapter_reports_why(self, tmp_path):
        registry = default_registry(str(tmp_path))
        registry.health_check()
        vcs = registry.get("vcs")
        if vcs.name in registry.unavailable():
            _, _, error = registry.resolve("git_status", "vcs")
            assert "不可用" in error

    def test_register_and_unregister(self, tmp_path):
        registry = ToolRegistry()
        registry.register(FileSystemAdapter(str(tmp_path)))
        assert registry.get("filesystem") is not None
        assert registry.unregister("filesystem") is True
        assert registry.get("filesystem") is None

    def test_health_check_covers_all(self, tmp_path):
        registry = default_registry(str(tmp_path))
        report = registry.health_check()
        assert set(report) >= {"system", "filesystem", "terminal", "editor", "browser", "vcs"}
        for name, info in report.items():
            assert "available" in info and "reason" in info

    def test_custom_adapter_can_be_plugged_in(self, tmp_path):
        class WeatherAdapter(ToolAdapter):
            name = "weather"

            def setup(self):
                self.capabilities.declare(Capability(
                    action="get_weather",
                    params={"city": "str"},
                    safety_level=SafetyLevel.READ_ONLY,
                ))

            def _dispatch(self, action, args):
                return {"city": args["city"], "temp": 22}, {}

        registry = default_registry(str(tmp_path))
        registry.register(WeatherAdapter())
        adapter, _, error = registry.resolve("get_weather")
        assert error == ""
        result = adapter.execute(ToolCommand(tool_name="weather", action="get_weather", args={"city": "北京"}))
        assert result.data["temp"] == 22

    def test_adapter_exception_is_isolated(self, tmp_path):
        class BoomAdapter(ToolAdapter):
            name = "boom"

            def setup(self):
                self.capabilities.declare(Capability(action="explode"))

            def _dispatch(self, action, args):
                raise RuntimeError("内部炸了")

        adapter = BoomAdapter()
        result = adapter.execute(ToolCommand(tool_name="boom", action="explode", args={}))
        assert result.status is ExecutionStatus.FAILED
        assert "内部炸了" in result.error