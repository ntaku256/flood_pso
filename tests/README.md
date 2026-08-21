# tests — CI プレビュー（オフライン決定的ゴールデン差分）

`world-full`（5.29h）や実データを使わず、**完全合成の極小 DEM** を
`FLOOD_PSO_OFFLINE=1` で決定的に生成し、俯瞰プレビュー画像をコミット済みゴールデンと
画素差分することで、地形出力の意図しない回帰を PR で検出する。

- 外部データ（和歌山 LiDAR / FGD 基本測量成果 / GSI オルソ）もネットワークも**不要**。
  fixture は `tools/ci_fixture.py` が数式だけで生成する合成 DEM（`data_cache/` は
  `.gitignore` 対象なのでリポジトリには残らない）。ライセンス上の懸念ゼロ。
- 生成物のバイト非決定は `.mca` の chunk timestamp と `level.dat` の gzip mtime のみで、
  俯瞰レンダの画素には影響しない → 画素差分は決定的（実測: 独立ビルドでバイト一致）。

## 使い方

```bash
make ci-preview                 # 合成生成 → レンダ → ゴールデン差分（閾値 0.5%）
make ci-golden                  # 地形を変える意図した変更後にゴールデンを更新
```

CI（`.github/workflows/ci-preview.yml`）は PR で自動実行し、`preview.png` / `diff.png` を
Artifacts に上げ、差分率を PR にコメントする。

## ゴールデンを更新すべきとき

`src/` の地形・ブロック化・洪水などを**意図して**変えた PR では、差分が閾値を超える。
その場合は `make ci-golden` で `tests/golden/ci_crop.png` を更新し、**同じ PR に含める**こと
（レビューで画像の変化を確認できる）。合成 DEM 諸元（`tools/ci_fixture.py` の `E0/N0/N` や
`Makefile` の `CI_CLAT/CI_CLON`）を変えた場合も同様に更新する。

## palette の blockstate ガード（#46）

`test_block_state_properties.py` は **外部レンダラ (BlueMap) で非描画になるブロック**を防ぐ回帰ガード。
依存ゼロで単体実行でき、CI (`ci-preview.yml`) の最初のステップで走る。

```bash
python tests/test_block_state_properties.py
```

Minecraft 本体は読み込み時にデフォルト blockstate を補完するが、BlueMap は
`assets/minecraft/blockstates/*.json` の variants キーとの照合でしかモデルを解決しない。
variants に空キー `""` が無いブロック（柱系の `axis`、草系の `snowy`）で Properties を
省略すると、**どの variant にも一致せず完全に非描画**になる。ブロック id 自体は既知なので
missing テクスチャにもならず、レンダラのログにも警告が出ない。

新しいブロックを `PALETTE_KEYS` に足すときは、`block_state_properties()` /
`block_state_properties_for_key()` が適切な Properties を返すか確認すること。
vanilla アセットとの**網羅照合**は手元の client jar を使う:

```bash
python tools/audit_blockstates.py ~/bluemap/<instance>/data/minecraft-client-*.jar
```
