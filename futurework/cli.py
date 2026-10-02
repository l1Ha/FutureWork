"""
命令行入口：``python -m futurework``。

用于快速验证与演示，也是 CI 的冒烟测试入口。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from futurework.runtime.session import TextSession


def _cmd_interactive(args: argparse.Namespace) -> int:
    session = TextSession()
    print("FutureWork 交互模式 —— 直接用中文/英文下达指令，输入 :quit 退出，:help 查看示例。\n")
    print(_help_text())
    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0
        if not line:
            continue
        if line in (":quit", ":q", "exit"):
            return 0
        if line in (":help", ":h"):
            print(_help_text())
            continue
        if line == ":health":
            print(json.dumps(session.health(), ensure_ascii=False, indent=2))
            continue
        if line == ":undo":
            print(session.undo()["message"])
            continue

        turn = session.run(line)
        print(f"AI > {turn.feedback.text}")
        print(f"      [状态 {turn.status.value}｜{turn.latency_ms:.1f}ms｜"
              f"{'置信 ' + format(turn.fused.confidence, '.2f') if turn.fused else '无融合信号'}]")
    return 0


def _cmd_once(args: argparse.Namespace) -> int:
    session = TextSession(workdir=args.workdir)
    turn = session.run(args.text)
    if args.json:
        print(json.dumps({
            "status": turn.status.value,
            "feedback": turn.feedback.text,
            "action": turn.intent.action if turn.intent else None,
            "confidence": turn.fused.confidence if turn.fused else None,
            "latency_ms": round(turn.latency_ms, 2),
            "stage_timings": {k: round(v, 2) for k, v in turn.stage_timings.items()},
            "errors": turn.errors,
        }, ensure_ascii=False, indent=2))
    else:
        print(turn.summary())
    return 0 if turn.ok or turn.needs_confirmation else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    session = TextSession(workdir=args.workdir)
    report = session.health()
    print("=== 适配器巡检 ===")
    for name, info in report["adapters"].items():
        mark = "✓" if info["available"] else "✗"
        print(f"{mark} {name}: {info['reason']}")
        print(f"    能力: {', '.join(info['actions'])}")
    print("\n=== 熔断器 ===")
    print(json.dumps(report["breakers"], ensure_ascii=False, indent=2))
    print(f"\n可用工具能力总数: {sum(len(v) for v in report['adapters'].values())}")
    return 0


def _cmd_capabilities(args: argparse.Namespace) -> int:
    session = TextSession(workdir=args.workdir)
    caps = session.orchestrator.registry.capabilities()
    print(json.dumps(caps, ensure_ascii=False, indent=2))
    return 0


def _help_text() -> str:
    return """
可用示例：
  打开浏览器                      启动应用
  列出所有窗口                    查看当前窗口
  切换到终端窗口                  窗口导航
  读取 notes.txt                  读文件
  把 hello 写入 notes.txt          写文件（可撤销）
  在 notes.txt 里查找 hello        搜索
  列出目录 .                      浏览目录
  打开 github.com                 打开网址
  搜索 python 异步编程             网页搜索
  运行 ls -la                     执行命令
  跑测试                          自动识别测试框架
  查看 git 状态                    版本控制
  提交代码 feat: add core          提交
  删除文件 old.txt                危险操作（需确认）
  撤销                            回退上一步

模态标记（演示用）：
  把这个移到那里 [POINT@0.3,0.4]   语音 + 指向手势

其他命令：
  :health    查看系统状态      :undo   撤销上一步
  :help      显示本帮助        :quit   退出
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="futurework",
        description="FutureWork —— 用人类的交互方式指挥生产力工具",
    )
    parser.add_argument("--workdir", default=None, help="工作目录（文件/终端操作范围）")
    sub = parser.add_subparsers(dest="command")

    p_inter = sub.add_parser("interactive", aliases=["i"], help="进入交互模式")
    p_inter.set_defaults(func=_cmd_interactive)

    p_once = sub.add_parser("say", help="执行单条指令")
    p_once.add_argument("text", help="自然语言指令")
    p_once.add_argument("--json", action="store_true", help="以 JSON 输出")
    p_once.set_defaults(func=_cmd_once)

    p_doc = sub.add_parser("doctor", help="环境与适配器巡检")
    p_doc.set_defaults(func=_cmd_doctor)

    p_cap = sub.add_parser("capabilities", help="列出所有可用能力")
    p_cap.set_defaults(func=_cmd_capabilities)

    parser.set_defaults(func=lambda a: _cmd_interactive(a))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())