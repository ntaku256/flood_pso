"""palette の BlockState Properties が外部レンダラで解決可能かを検証する。

背景 (#46): Minecraft 本体は読み込み時にデフォルト blockstate を補完するが、BlueMap の
ような外部レンダラは ``assets/minecraft/blockstates/*.json`` の variants キーとの照合で
しかモデルを解決せず、補完をしない。variants に空キー ``""`` を持たないブロック
（柱系の axis、草系の snowy）で Properties を省略すると**完全に非描画**になり、しかも
ブロック id は既知なのでレンダラのログにも警告が出ない。

vanilla のアセットを同梱せずに回帰を検出するため、ここでは
「空キーを持たないことが分かっているブロック群」を明示し、palette がそれらに必ず
Properties を与えていることを検証する。網羅的な照合は tools/audit_blockstates.py が
実際の client jar を使って行う（CI では走らせない）。

    python tests/test_block_state_properties.py     # 依存なしで単体実行
    python -m pytest tests/                        # pytest があればそちらでも動く
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from block_palette import (  # noqa: E402
    BLOCKS,
    PALETTE_KEYS,
    block_state_properties,
    block_state_properties_for_key,
)

# vanilla の blockstates に空キー "" が無く、Properties 省略だと非描画になるブロック。
# key = minecraft 名, value = 最低限含まれていなければならないプロパティ名。
NEEDS_PROPERTY = {
    "minecraft:grass_block": "snowy",
    "minecraft:podzol": "snowy",
    "minecraft:mycelium": "snowy",
    "minecraft:deepslate": "axis",
    "minecraft:oak_log": "axis",
    "minecraft:spruce_log": "axis",
    "minecraft:stripped_oak_log": "axis",
    "minecraft:verdant_froglight": "axis",
}


def test_known_blocks_get_required_property():
    """#46 で実際に非描画になっていたブロックが Properties を持つこと。"""
    for name, prop in NEEDS_PROPERTY.items():
        props = block_state_properties(name)
        assert props is not None, f"{name}: Properties が None（レンダラで非描画になる）"
        assert prop in props, f"{name}: {prop} が無い（実際: {props}）"


def test_palette_pillars_have_axis():
    """palette に含まれる柱系ブロックが漏れなく axis を持つこと。

    新しい *_log / *_wood を palette に足したときに Properties を付け忘れると落ちる。
    """
    missing = []
    for key in PALETTE_KEYS:
        name = BLOCKS[key][0]
        if not name.endswith(("_log", "_wood")):
            continue
        props = block_state_properties_for_key(key)
        if not props or "axis" not in props:
            missing.append((key, name, props))
    assert not missing, f"axis を持たない柱系ブロック: {missing}"


def test_palette_snowy_blocks_have_snowy():
    """palette に含まれる snowy 系ブロックが漏れなく snowy を持つこと。"""
    missing = []
    for key in PALETTE_KEYS:
        name = BLOCKS[key][0]
        if name not in ("minecraft:grass_block", "minecraft:podzol", "minecraft:mycelium"):
            continue
        props = block_state_properties_for_key(key)
        if not props or "snowy" not in props:
            missing.append((key, name, props))
    assert not missing, f"snowy を持たない地表ブロック: {missing}"


def test_no_palette_key_regresses_to_none():
    """NEEDS_PROPERTY のブロックを使う palette KEY が None を返さないこと。"""
    bad = [
        (key, BLOCKS[key][0])
        for key in PALETTE_KEYS
        if BLOCKS[key][0] in NEEDS_PROPERTY and not block_state_properties_for_key(key)
    ]
    assert not bad, f"Properties が None の KEY: {bad}"


def main() -> int:
    """pytest が無い環境でも動く簡易ランナー（CI はこちらを呼ぶ）。"""
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        else:
            print(f"ok   {t.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
