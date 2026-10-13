"""アノテーション入力用ユーティリティ.

/by_image_tag配下の各画像ディレクトリ内のjsonを読み取り，/data/inter配下へ
annotations.csvおよびqc_annotations.csvを生成する。
notebooksから呼び出す想定。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

# 依存するパスはプロジェクトルートからの相対関係のみを利用する
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "by_image_tag"
TABLE_DIR = PROJECT_ROOT / "table"
ANNOTATIONS_PATH = TABLE_DIR / "annotations.csv"
QC_ANNOTATIONS_PATH = TABLE_DIR / "qc_annotations.csv"
CSV_HEADER = ["point_id", "img_id", "worker_id", "x", "y"]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _worker_name_to_id(name: str) -> int:
    """アノテーター名を64bit整数ハッシュへ変換."""

    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _parse_filename(json_path: Path) -> Tuple[str, str]:
    """jsonファイル名から画像IDとワーカー名を抽出."""

    stem = json_path.stem  # .json除去
    if ".png_" not in stem:
        raise ValueError(f"Unexpected file name format: {json_path.name}")
    img_root, worker_name = stem.split(".png_", 1)
    img_id = f"{img_root}.png"
    return img_id, worker_name


def _iter_points(json_path: Path) -> Iterator[Tuple[str, int, float, float]]:
    """単一json内のポイント群を絶対座標に変換して返す."""

    img_id, worker_name = _parse_filename(json_path)
    worker_id = _worker_name_to_id(worker_name)

    with json_path.open(encoding="utf-8") as fp:
        payload = json.load(fp)

    for entry in payload:
        crop = entry.get("crop_coordinates", {})
        origin_x = float(crop.get("x", 0))
        origin_y = float(crop.get("y", 0))
        points = entry.get("cell_coordinates", {}).get("points") or []
        for point in points:
            abs_x = origin_x + float(point["x"])
            abs_y = origin_y + float(point["y"])
            yield img_id, worker_id, abs_x, abs_y


def _qc_image_index(img_id: str) -> int | None:
    """QC画像であれば整数インデックスを返す."""

    stem = Path(img_id).stem
    if not stem.startswith("QC"):
        return None
    suffix = stem[2:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def build_annotation_tables(
    raw_dir: Path | None = None,
    annotations_path: Path | None = None,
    qc_annotations_path: Path | None = None,
) -> Dict[str, int]:
    """原始jsonから集約csvを生成する."""

    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    annotations_path = Path(annotations_path) if annotations_path else ANNOTATIONS_PATH
    qc_annotations_path = (
        Path(qc_annotations_path) if qc_annotations_path else QC_ANNOTATIONS_PATH
    )

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")

    _ensure_dir(annotations_path)
    _ensure_dir(qc_annotations_path)

    total_points = 0
    qc_points = 0
    point_id = 0

    # by_image_tag構造: 各画像ディレクトリ内のJSONファイルを検索
    json_files = sorted(p for p in raw_dir.glob("*/*.json") if p.is_file())

    with annotations_path.open("w", newline="", encoding="utf-8") as ann_fp, qc_annotations_path.open(
        "w", newline="", encoding="utf-8"
    ) as qc_fp:
        ann_writer = csv.writer(ann_fp, lineterminator="\n")
        qc_writer = csv.writer(qc_fp, lineterminator="\n")
        ann_writer.writerow(CSV_HEADER)
        qc_writer.writerow(CSV_HEADER)

        for json_path in json_files:
            for img_id, worker_id, abs_x, abs_y in _iter_points(json_path):
                point_id += 1
                row = [point_id, img_id, worker_id, abs_x, abs_y]
                ann_writer.writerow(row)
                total_points += 1

                qc_img_id = _qc_image_index(img_id)
                if qc_img_id is not None:
                    qc_writer.writerow([point_id, qc_img_id, worker_id, abs_x, abs_y])
                    qc_points += 1

    return {"annotations": total_points, "qc_annotations": qc_points}


__all__ = [
    "build_annotation_tables",
]