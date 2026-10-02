#!/usr/bin/env python3
"""
基准测试：量化"交互是否流畅"。

运行：``python examples/benchmark.py``

测三件事：
1. **端到端延迟** —— 从"人说完话"到"工具执行完"要多久；
2. **意图准确率** —— 各类自然说法是否被正确理解；
3. **多模态增益** —— 加上手势/视线/表情后，理解准确率与置信度提升多少。
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futurework.cognition.intent import IntentParser
from futurework.runtime.orchestrator import Orchestrator
from futurework.runtime.session import TextSession
from futurework.sensory.fusion import MultimodalFusionEngine
from futurework.types import EmotionType, GestureType, SpeechSignal

BAR = "─" * 74


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


# ============================================================================
# 1. 意图准确率
# ============================================================================
# (说法, 期望动作)
CORPUS = [
    ("打开浏览器", "launch_app"),
    ("启动计算器", "launch_app"),
    ("打开 github.com", "open_url"),
    ("打开 https://python.org", "open_url"),
    ("搜索 机器学习 论文", "search_web"),
    ("列出目录", "list_directory"),
    ("列出目录 ./src", "list_directory"),
    ("读取 README.md", "read_file"),
    ("把笔记写入 notes.txt", "write_file"),
    ("追加一行到 log.txt", "append_file"),
    ("在 code.py 里查找 main", "search_in_file"),
    ("删除文件 old.tmp", "delete_file"),
    ("创建目录 output", "create_directory"),
    ("把 a.txt 复制到 b.txt", "copy_file"),
    ("列出所有窗口", "list_windows"),
    ("切换到终端窗口", "focus_window"),
    ("最大化窗口", "maximize_window"),
    ("查看 git 状态", "git_status"),
    ("查看改动", "git_diff"),
    ("提交代码 feat: add", "git_commit"),
    ("跑测试", "run_tests"),
    ("运行 ls -la", "run_command"),
    ("编辑 main.py", "open_file_editor"),
    ("后退", "browser_back"),
    ("撤销", "undo"),
    ("取消", "cancel"),
    ("确认", "confirm"),
    ("不要", "reject"),
    ("open browser", "launch_app"),
    ("open example.com", "open_url"),
    ("read notes.txt", "read_file"),
    ("delete file old.tmp", "delete_file"),
    ("write hello to out.txt", "write_file"),
    ("list directory", "list_directory"),
    ("git status", "git_status"),
    ("run the command ls -la", "run_command"),
    ("run tests", "run_tests"),
    ("go back", "browser_back"),
    ("go to docs.python.org", "open_url"),
    ("minimize all", "minimize_all"),
    ("undo", "undo"),
    ("yes", "confirm"),
    # 口语化：动作放在对象之后
    ("把 TODO.md 看一下", "read_file"),
    ("把 config.json 看看", "read_file"),
    ("看一下 README.md", "unknown"),   # 未支持的说法，应落到追问而非乱执行
    # 办公生产力套件指令
    ("计算 sales.csv 的 revenue 列的 sum", "table_aggregate"),
    ("生成一份关于 项目架构 的工作报告", "generate_report"),
    ("提取 doc.md 的大纲", "extract_outline"),
    ("添加待办: 准备发布 v1.0", "todo_add"),
    ("查看所有待办", "todo_list"),
    ("完成待办 v1.0", "todo_complete"),
]


def benchmark_accuracy() -> tuple[float, list[tuple[str, str, str]]]:
    parser = IntentParser()
    correct, misses = 0, []
    for text, expected in CORPUS:
        actual = parser.parse(text).action
        if actual == expected:
            correct += 1
        else:
            misses.append((text, expected, actual))
    return correct / len(CORPUS), misses


# ============================================================================
# 2. 端到端延迟
# ============================================================================
def benchmark_latency(workdir: str, rounds: int = 60) -> dict:
    with open(os.path.join(workdir, "bench.txt"), "w", encoding="utf-8") as fh:
        fh.write("benchmark payload\n" * 20)

    latencies: dict[str, list[float]] = {}

    def timed(label: str, command: str) -> None:
        samples = []
        for _ in range(rounds):
            session = TextSession(workdir=workdir)
            start = time.perf_counter()
            session.run(command)
            samples.append((time.perf_counter() - start) * 1000)
        latencies[label] = samples

    timed("纯语言 · 只读文件", "读取 bench.txt")
    timed("纯语言 · 列目录", "列出目录 .")
    timed("纯语言 · 写文件", "把 payload 写入 out.txt")

    # 多模态：语音 + 指向 + 视线 + 表情
    multimodal = []
    for _ in range(rounds):
        session = TextSession(workdir=workdir)
        start = time.perf_counter()
        session.say(
            "读取 bench.txt",
            gesture=GestureType.POINT,
            look_at=(0.5, 0.5),
            expression=EmotionType.FOCUSED,
        )
        multimodal.append((time.perf_counter() - start) * 1000)
    latencies["多模态 · 语音+手势+视线+表情"] = multimodal

    return {
        label: {
            "mean": statistics.mean(samples),
            "p50": percentile(samples, 50),
            "p95": percentile(samples, 95),
            "p99": percentile(samples, 99),
            "max": max(samples),
        }
        for label, samples in latencies.items()
    }


# ============================================================================
# 3. 多模态增益
# ============================================================================
def benchmark_multimodal_gain() -> dict:
    """
    衡量多模态带来的**分歧检出能力**。

    注意这里刻意不度量"准确率提升"：对已经说清楚的话，加一个点头并不会
    让理解更对。真正的价值在于——当人体信号与语言信号互相矛盾时，
    系统能发现矛盾并转为追问，而不是照做。
    """
    from futurework.types import HeadPoseSignal

    # 肯定/否定句：只有这类句子才能与点头/摇头产生语义极性的对错关系
    affirm = ["确认执行", "可以开始", "就这样吧", "行"]
    deny = ["不要动", "取消操作", "别删了", "停"]

    engine = MultimodalFusionEngine()

    def run(phrase: str, **pose):
        return engine.fuse(engine.assemble(
            speech=SpeechSignal(transcript=phrase),
            head_pose=HeadPoseSignal(**pose) if pose else None,
        ))

    agreeing = [run(p, nod=True) for p in affirm]
    conflicting = [run(p, nod=True, shake=True) for p in affirm]

    agree_conf = statistics.mean(f.confidence for f in agreeing)
    conflict_conf = statistics.mean(f.confidence for f in conflicting)
    flagged = sum(1 for f in conflicting if f.requires_confirmation)
    agree_flagged = sum(1 for f in agreeing if f.requires_confirmation)

    return {
        "agree_confidence": agree_conf,
        "conflict_confidence": conflict_conf,
        "gap": agree_conf - conflict_conf,
        "conflict_detected": flagged,
        "total": len(conflicting),
        "false_alarms": agree_flagged,
    }


# ============================================================================
def main() -> int:
    workdir = tempfile.mkdtemp(prefix="futurework_bench_")

    print(BAR)
    print("FutureWork 性能基准")
    print(BAR)

    # ---- 准确率 ----
    accuracy, misses = benchmark_accuracy()
    print(f"\n【意图理解准确率】")
    print(f"  语料规模   → {len(CORPUS)} 条中英自然说法")
    print(f"  准确率     → {accuracy * 100:.1f}%（{int(accuracy * len(CORPUS))}/{len(CORPUS)}）")
    if misses:
        print(f"  未命中     →")
        for text, expected, actual in misses:
            print(f"      「{text}」期望 {expected}，实际 {actual}")

    # ---- 延迟 ----
    latency = benchmark_latency(workdir)
    print(f"\n【端到端延迟】（每类 {60} 次）")
    print(f"  {'场景':<32}{'均值':>9}{'P50':>9}{'P95':>9}{'P99':>9}{'最大':>9}")
    for label, stats in latency.items():
        print(f"  {label:<30}{stats['mean']:>9.2f}{stats['p50']:>9.2f}"
              f"{stats['p95']:>9.2f}{stats['p99']:>9.2f}{stats['max']:>9.2f}")
    print(f"  单位：毫秒。人类感知流畅阈约为 300ms（10fps 交互的下限）。")

    worst = max(s["p95"] for s in latency.values())
    verdict = "流畅" if worst < 300 else ("可用" if worst < 1000 else "偏慢")
    print(f"  结论       → P95 最差 {worst:.2f}ms，交互体验{verdict}")

    # ---- 多模态增益 ----
    gain = benchmark_multimodal_gain()
    print(f"\n【多模态增益：分歧检出】")
    print(f"  语音肯定 + 点头（一致） → 置信度 {gain['agree_confidence']:.3f}，"
          f"其中 {gain['false_alarms']}/{gain['total']} 次仍要求确认")
    print(f"  语音肯定 + 点头摇头（矛盾） → 置信度 {gain['conflict_confidence']:.3f}，"
          f"{gain['conflict_detected']}/{gain['total']} 次转为追问")
    print(f"  分歧落差 → {gain['gap']:.3f}")
    print(f"  说明：多模态的价值不在'总是更准'——对已经说清的话，")
    print(f"        再加一个点头不会让理解更对。真正的价值是发现矛盾：")
    print(f"        身体在说'是'而嘴里在说'不'时，系统该停下问一句。")

    # ---- 分阶段耗时 ----
    session = TextSession(workdir=workdir)
    turn = session.run("读取 bench.txt")
    print(f"\n【单次交互分阶段耗时】")
    total = sum(turn.stage_timings.values())
    for stage, ms in turn.stage_timings.items():
        share = ms / total * 100 if total else 0
        print(f"  {stage:<12}{ms:>8.2f}ms  ({share:>5.1f}%)")
    print(f"  {'合计':<12}{turn.latency_ms:>8.2f}ms")

    return 0 if accuracy >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())