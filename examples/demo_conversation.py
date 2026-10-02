#!/usr/bin/env python3
"""
演示三：像人一样说话 —— 复合指令、纠正、补充

运行：``python examples/demo_conversation.py``

前两个演示展示了"系统能做什么"，这个演示展示"人怎么说话"。
人和人交流时的三大习惯，机器接口往往统统不支持：

1. **一次交代两件事**："复制完然后删掉它"——而且第二步依赖第一步；
2. **否定刚说过的话**："不对，应该是 b.txt"；
3. **只补一个词**："读取" → "a.txt"，而不是把整句重说一遍。

系统若不支持这些，每一轮交互都像在跟机器重新学一遍语法。
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futurework.runtime.session import TextSession
from futurework.types import ExecutionStatus

BAR = "─" * 74


def show(turn, files: str, note: str = "") -> None:
    print(f"\n  我说 → {note or turn.feedback.text}")
    print(f"  系统 → {turn.feedback.text}")
    print(f"  状态 → {turn.status.value}")
    print(f"  文件 → {files}")


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="futurework_talk_")
    listing = lambda: ", ".join(sorted(os.listdir(workdir))) or "(空)"

    with open(os.path.join(workdir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("# 季度报告\n\n营收同比增长 12%。\n")
    with open(os.path.join(workdir, "TODO.md"), "w", encoding="utf-8") as fh:
        fh.write("- [ ] 写测试\n- [ ] 发版\n")

    session = TextSession(workdir=workdir)
    print(f"工作目录：{workdir}\n初始文件：{listing()}")

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 一、一次交代两件事（第二步依赖第一步的结果）\n{BAR}")
    show(session.run("把 report.md 复制到 backup.md 然后读取 backup.md"), listing(),
         "把 report.md 复制到 backup.md 然后读取 backup.md")

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 二、危险操作也会一起说完 —— 危险的那步必须单独确认\n{BAR}")
    show(session.run("把 TODO.md 复制到 scratch.md 然后删除 scratch.md"), listing(),
         "把 TODO.md 复制到 scratch.md 然后删除 scratch.md")
    print("\n  我只是点了点头 →")
    show(session.nod(), listing())

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 三、做错了，说「不对」——系统应撤回，而不是要你重说一遍\n{BAR}")
    show(session.run("把 写错了的内容 写入 notes.txt"), listing(),
         "把 写错了的内容 写入 notes.txt")
    show(session.run("不对"), listing())
    print("\n  撤回之后 contents 已不存在——它回到了动手之前的状态")

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 四、说「不对，应该是 X」——连替换动作一起给\n{BAR}")
    show(session.run("把 另一份错误内容 写入 wrong.txt"), listing(),
         "把 另一份错误内容 写入 wrong.txt")
    show(session.run("不对，应该是 report.md"), listing())
    print("\n  系统识别出「应该是 report.md」是一个裸值，")
    print("  于是把它回填进刚才那个动作的 path 槽位，而不是当成一条新指令。")
    with open(os.path.join(workdir, "report.md"), encoding="utf-8") as fh:
        print(f"  report.md 现在是：{fh.read().splitlines()[0]!r} ← 已被覆盖，说明回填成功")

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 五、只说动作不说对象 —— 系统只追问缺的那一个槽位\n{BAR}")
    show(session.run("读取"), listing(), "读取")
    show(session.run("TODO.md"), listing(), "TODO.md")

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 六、第一步失败时，不继续执行后面的破坏性动作\n{BAR}")
    with open(os.path.join(workdir, "important.txt"), "w", encoding="utf-8") as fh:
        fh.write("重要数据")
    show(session.run("把 不存在的文件 复制到 x.txt 然后删除 important.txt"), listing(),
         "把 不存在的文件 复制到 x.txt 然后删除 important.txt")
    print("\n  important.txt 仍在 —— 复合指令遇错即停，")
    print("  否则用户会以为「复制成功了才删除」，从而丢掉本该安全的数据。")

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 七、语义相同、语气不同 —— 中英文与口语化说法都能听懂\n{BAR}")
    for line in ["read TODO.md", "读取 todo.md", "Read  TODO.md", "把 TODO.md 看一下"]:
        turn = session.run(line)
        print(f"  {line:22} → {turn.status.value:8} {turn.feedback.text[:46]}")
    print()
    print("  注意第二行：'读取 todo.md' 找不到文件，而 'read TODO.md' 能找到。")
    print("  这不是 bug——Linux/macOS 的文件名大小写敏感。")
    print("  早期版本在文本归一化时做了小写化，把 TODO.md 变成 todo.md，")
    print("  导致系统「找不到明明存在的文件」。因此现在归一化刻意保留大小写。")

    # ------------------------------------------------------------------
    print(f"\n{BAR}\n▶ 交互统计\n{BAR}")
    health = session.health()
    print(f"  累计交互 → {health['turns']} 轮")
    print(f"  最终文件 → {listing()}")
    print(f"  遗留错误 → {health['bus_errors']} 条（被隔离的内部异常）")
    return 0


if __name__ == "__main__":
    sys.exit(main())