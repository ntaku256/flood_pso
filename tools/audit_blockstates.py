#!/usr/bin/env python3
"""palette 全件を vanilla の blockstates と照合し、外部レンダラで非描画になるキーを洗い出す。

背景 (#46): Minecraft は読み込み時にデフォルト blockstate を補完するが、BlueMap 等の
外部レンダラは ``assets/minecraft/blockstates/<id>.json`` の variants キーとの照合でしか
モデルを解決しない。variants に空キー ``""`` が無いブロックで Properties を省略すると
どの variant にも一致せず**完全に非描画**になる（警告も出ない）。

vanilla アセットは同梱できないので、手元の Minecraft client jar（または展開済み
assets ディレクトリ）を指定して実行する。BlueMap が落としたものを使える:
  ~/bluemap/<instance>/data/minecraft-client-*.jar

使い方:
    python tools/audit_blockstates.py ~/bluemap/arida-hd-instance/data/minecraft-client-26.2.jar

終了コード: 0 = 全件解決可能 / 1 = 非描画になるキーあり
"""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from block_palette import BLOCKS, PALETTE_KEYS, block_state_properties_for_key  # noqa: E402

PREFIX = "assets/minecraft/blockstates/"


def load_blockstates(src: Path) -> dict[str, dict]:
    """client jar か展開済みディレクトリから blockstates を読む。"""
    out: dict[str, dict] = {}
    if src.is_dir():
        base = src / PREFIX if (src / PREFIX).is_dir() else src
        for p in base.glob("*.json"):
            out[p.stem] = json.loads(p.read_text())
        return out
    with zipfile.ZipFile(src) as z:
        for n in z.namelist():
            if n.startswith(PREFIX) and n.endswith(".json"):
                out[n[len(PREFIX):-5]] = json.loads(z.read(n))
    return out


def resolves(state: dict, props: dict[str, str] | None) -> bool:
    """レンダラがモデルを解決できるか。

    - multipart は ``when`` 無しのパートが必ずあるので解決可能とみなす
    - variants の空キー ``""`` は無条件マッチ
    - それ以外は「variants キーの k=v がすべて Properties に含まれる」ことが条件
    """
    if "variants" not in state:
        return True
    props = props or {}
    for key in state["variants"]:
        if key == "":
            return True
        kv = dict(p.split("=", 1) for p in key.split(",") if "=" in p)
        if all(props.get(a) == b for a, b in kv.items()):
            return True
    return False


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    states = load_blockstates(Path(argv[1]).expanduser())
    if not states:
        print(f"blockstates が見つからない: {argv[1]}")
        return 2

    bad: list[tuple[str, str, dict | None]] = []
    unknown: list[str] = []
    for key in PALETTE_KEYS:
        name = BLOCKS[key][0]
        bid = name.split(":", 1)[1]
        if bid not in states:
            unknown.append(f"{key} -> {name}")
            continue
        props = block_state_properties_for_key(key)
        if not resolves(states[bid], props):
            bad.append((key, name, props))

    print(f"palette KEY: {len(PALETTE_KEYS)}  blockstates: {len(states)}")
    if unknown:
        print(f"\n[warn] blockstates に無いブロック {len(unknown)} 件（jar のバージョン違い？）:")
        for u in unknown:
            print(f"    {u}")
    if bad:
        print(f"\n[NG] 外部レンダラで非描画になる KEY: {len(bad)}")
        for key, name, props in sorted(bad):
            print(f"    {key:24s} -> {name:34s} props={props}")
        return 1
    print("\n[OK] 全 KEY が variants のいずれかに解決できる")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
