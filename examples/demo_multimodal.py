#!/usr/bin/env python3
"""
演示一：像人一样指挥电脑 —— 多模态自然交互全流程

运行：``python examples/demo_multimodal.py``

这个脚本模拟一次真实的协作过程：
    说话 → 系统理解 → 做事 → 追问 → 点头确认 → 完成 → 说不 → 撤销

全程不碰真实文件（限制在临时目录内），可安全运行。
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futurework.runtime.session import TextSession
from futurework.types import EmotionType, GestureType

BAR = "─" * 74


def header(title: str) -> None:
    print(f"\n{BAR}\n▶ {title}\n{BAR}")


def step(session: TextSession, label: str, turn) -> None:
    intent = turn.intent.action if turn.intent else "—"
    conf = f"{turn.fused.confidence:.2f}" if turn.fused else "—"
    print(f"\n  【{label}】")
    print(f"    我说/做 → {label}")
    print(f"    理解   → {intent}（置信 {conf}）")
    print(f"    回应   → {turn.feedback.text}")
    print(f"    状态   → {turn.status.value}｜{turn.latency_ms:.1f}ms")


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="futurework_demo_")
    os.makedirs(os.path.join(workdir, "src"), exist_ok=True)
    with open(os.path.join(workdir, "src", "main.py"), "w", encoding="utf-8") as fh:
        fh.write("def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n")
    with open(os.path.join(workdir, "TODO.md"), "w", encoding="utf-8") as fh:
        fh.write("# 待办\n- [ ] 写测试\n- [ ] 发版\n")

    session = TextSession(workdir=workdir)
    print(f"工作目录：{workdir}")

    # ------------------------------------------------------------------
    header("场景一：纯语言 —— 像跟同事说话一样下指令")
    step(session, "列出目录", session.run("列出目录"))
    step(session, "读取 TODO.md", session.run("读取 TODO.md"))

    # ------------------------------------------------------------------
    header("场景二：说话 + 指向手势 —— 「把这个挪到那里」")
    turn = session.say(
        "把 TODO.md 复制到 backup.md",
        gesture=GestureType.POINT,
        gesture_position=(0.72, 0.35, 0.4),
        look_at=(0.80, 0.30),
    )
    step(session, "把 TODO.md 复制到 backup.md（同时伸手指向 + 视线跟随）", turn)
    print(f"    融合锚点 → {turn.fused.target.label if turn.fused.target else '无'}"
          f" @ {tuple(round(v, 3) for v in turn.fused.target.bounding_box) if turn.fused.target else '-'}")
    print(f"    融合依据 → {'; '.join(turn.fused.rationale)}")
    print(f"    文件已建 → {os.path.exists(os.path.join(workdir, 'backup.md'))}")

    turn = session.say("把这个移到 archive", gesture=GestureType.POINT,
                       gesture_position=(0.3, 0.6, 0.4))
    step(session, "把这个移到 archive（再说一次，但只靠手势消歧）", turn)

    # ------------------------------------------------------------------
    header("场景三：皱眉困惑 —— 系统主动追问，而不是瞎猜")
    turn = session.say("把那个改一下", expression=EmotionType.CONFUSED)
    step(session, "把那个改一下（皱眉）", turn)
    print(f"    情绪     → {turn.fused.affective_state.value}"
          f"｜注意力 {turn.fused.attention_score}")
    print(f"    歧义度   → {turn.fused.ambiguity_score}")

    # ------------------------------------------------------------------
    header("场景四：危险操作 —— 先问，再点头才动手")
    with open(os.path.join(workdir, "obsolete.log"), "w", encoding="utf-8") as fh:
        fh.write("一堆没用的旧日志")

    turn = session.run("删除文件 obsolete.log")
    step(session, "删除文件 obsolete.log", turn)
    print(f"    文件仍在 → {os.path.exists(os.path.join(workdir, 'obsolete.log'))}（确认前不动手）")

    turn = session.nod()
    step(session, "点头（不说话，直接同意）", turn)
    print(f"    文件仍在 → {os.path.exists(os.path.join(workdir, 'obsolete.log'))}（已执行）")

    # ------------------------------------------------------------------
    header("场景五：摇头拒绝 —— 同样的手势，说「不」")
    with open(os.path.join(workdir, "keep.txt"), "w", encoding="utf-8") as fh:
        fh.write("这个要保留")

    turn = session.run("删除文件 keep.txt")
    step(session, "删除文件 keep.txt", turn)
    turn = session.shake()
    step(session, "摇头（不）", turn)
    print(f"    文件仍在 → {os.path.exists(os.path.join(workdir, 'keep.txt'))}（拒绝了，很好）")

    # ------------------------------------------------------------------
    header("场景六：可撤销 —— 说错话不丢人，说「撤销」就行")
    with open(os.path.join(workdir, "main.md"), "w", encoding="utf-8") as fh:
        fh.write("原始文档内容")

    step(session, "把 改坏的标题 写入 main.md", session.run("把 改坏的标题 写入 main.md"))
    step(session, "撤销", session.say("撤销"))
    with open(os.path.join(workdir, "main.md"), encoding="utf-8") as fh:
        print(f"    文件内容 → {fh.read()!r}（已还原）")

    # ------------------------------------------------------------------
    header("场景七：安全底线 —— 危险命令被拦下")
    turn = session.say("在终端执行 rm -rf /")
    step(session, "在终端执行 rm -rf /", turn)

    # ------------------------------------------------------------------
    header("交互统计")
    health = session.health()
    print(f"    累计交互   → {health['turns']} 轮")
    print(f"    适配器     → " + "、".join(
        f"{name}{'✓' if info['available'] else '✗'}"
        for name, info in health["adapters"].items()
    ))
    print(f"    可用能力   → {sum(len(v) for v in health['adapters'].values())} 项")
    print(f"    遗留错误   → {health['bus_errors']} 条（被隔离的内部异常）")

    return 0


if __name__ == "__main__":
    sys.exit(main())