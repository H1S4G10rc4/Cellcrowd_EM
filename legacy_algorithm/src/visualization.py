"""可視化ユーティリティ.

アノテーション・クラスタ・ヒートマップの可視化関数を提供する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from PIL import Image
from scipy.spatial import ConvexHull

from .heatmap import HeatmapConfig, build_heatmap_for_image, overlay_heatmap


def load_annotations_from_json(qc_dir: Path) -> list[tuple[float, float]]:
    """QC画像ディレクトリ内のJSONファイルから全アノテーション座標を読み込む.
    
    Args:
        qc_dir: QC画像のディレクトリパス（例: by_image_tag/QC1.png/）
    
    Returns:
        (x, y) タプルのリスト（元画像上の絶対座標）
    """
    points = []
    json_files = list(qc_dir.glob("*.json"))
    
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # JSONは配列のリスト
            for item in data:
                # crop_coordinatesからオフセットを取得
                crop_x = 0.0
                crop_y = 0.0
                if "crop_coordinates" in item:
                    crop_x = float(item["crop_coordinates"].get("x", 0))
                    crop_y = float(item["crop_coordinates"].get("y", 0))
                
                if "cell_coordinates" in item and "points" in item["cell_coordinates"]:
                    for pt in item["cell_coordinates"]["points"]:
                        if "x" in pt and "y" in pt:
                            # クロップ座標を加算して元画像上の絶対座標に変換
                            abs_x = float(pt["x"]) + crop_x
                            abs_y = float(pt["y"]) + crop_y
                            points.append((abs_x, abs_y))
        except Exception as e:
            print(f"  [WARNING] Failed to load {json_path.name}: {e}")
            continue
    
    return points


def plot_qc_raw_annotations(
    qc_dir: Path,
    output_path: Path | str,
    *,
    title: str | None = None,
    color: str = "red",
    figsize: tuple[int, int] = (12, 12),
    dpi: int = 150,
    point_size: int = 5,
) -> None:
    """QC画像のJSONから直接アノテーションを読み込んでオーバーレイ.
    
    Args:
        qc_dir: QC画像のディレクトリ（例: by_image_tag/QC1.png/）
        output_path: 出力先パス
        title: 図のタイトル
        color: ポイントの色
        figsize: 図のサイズ
        dpi: 解像度
        point_size: 点のサイズ
    """
    qc_name = qc_dir.name
    image_path = qc_dir / qc_name
    
    if not image_path.exists():
        print(f"  [ERROR] Image not found: {image_path}")
        return
    
    # JSONから座標を読み込み
    points = load_annotations_from_json(qc_dir)
    print(f"  [DEBUG] Loaded {len(points)} points from JSON files in {qc_dir.name}")
    
    # 画像を読み込み
    img = np.asarray(Image.open(image_path).convert("RGB"))
    img_height, img_width = img.shape[:2]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        print(f"  [DEBUG] x range: {min(xs):.1f} - {max(xs):.1f}")
        print(f"  [DEBUG] y range: {min(ys):.1f} - {max(ys):.1f}")
        
        ax.scatter(
            xs, ys,
            c=color,
            s=point_size,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.2,
            zorder=100,
            marker='o',
        )
    
    title_text = title or f"{qc_name} - Raw Annotations"
    ax.set_title(f"{title_text} ({len(points)} points)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _match_img_id(df: pd.DataFrame, img_id: int | str) -> pd.DataFrame:
    """img_idでフィルタリング（型の不一致を考慮）."""
    # 直接比較を試みる
    subset = df[df["img_id"] == img_id]
    if not subset.empty:
        return subset
    # 整数変換を試みる
    try:
        img_id_int = int(img_id)
        subset = df[df["img_id"] == img_id_int]
        if not subset.empty:
            return subset
        # DataFrameの列が文字列の場合
        subset = df[df["img_id"].astype(str) == str(img_id)]
        if not subset.empty:
            return subset
    except (ValueError, TypeError):
        pass
    return df[df["img_id"] == img_id]  # 空のDataFrameを返す


def get_cluster_bboxes(
    df: pd.DataFrame,
    cluster_col: str,
    img_id: int | str,
    padding: int = 3,
) -> list[dict[str, Any]]:
    """クラスタごとのバウンディングボックスを取得.
    
    Args:
        df: クラスタ情報を含むDataFrame (img_id, x, y, cluster_col)
        cluster_col: クラスタIDのカラム名
        img_id: 対象画像のID
        padding: バウンディングボックスのパディング（ピクセル）
    
    Returns:
        バウンディングボックス情報のリスト
    """
    subset = _match_img_id(df, img_id)
    bboxes = []
    for cid, group in subset.groupby(cluster_col):
        if cid < 0:
            continue
        x_min, x_max = group["x"].min(), group["x"].max()
        y_min, y_max = group["y"].min(), group["y"].max()
        bboxes.append({
            "cluster_id": cid,
            "x": x_min - padding,
            "y": y_min - padding,
            "width": (x_max - x_min) + 2 * padding,
            "height": (y_max - y_min) + 2 * padding,
        })
    return bboxes


def get_cluster_convex_hulls(
    df: pd.DataFrame,
    cluster_col: str,
    img_id: int | str,
    padding: float = 2.0,
) -> list[dict[str, Any]]:
    """クラスタごとの凸包（外側の点をつないだ多角形）を取得.
    
    Args:
        df: クラスタ情報を含むDataFrame (img_id, x, y, cluster_col)
        cluster_col: クラスタIDのカラム名
        img_id: 対象画像のID
        padding: 頂点から外側に広げるパディング（ピクセル）
    
    Returns:
        凸包情報のリスト（各要素は cluster_id, vertices を含む辞書）
    """
    subset = _match_img_id(df, img_id)
    hulls = []
    
    for cid, group in subset.groupby(cluster_col):
        if cid < 0:
            continue
        
        points = group[["x", "y"]].to_numpy()
        n_points = len(points)
        
        if n_points < 3:
            # 2点以下の場合は、点を囲む小さな四角形を生成
            center = points.mean(axis=0)
            if n_points == 1:
                # 1点の場合、正方形
                size = padding * 2
                vertices = np.array([
                    [center[0] - size, center[1] - size],
                    [center[0] + size, center[1] - size],
                    [center[0] + size, center[1] + size],
                    [center[0] - size, center[1] + size],
                ])
            else:
                # 2点の場合、細長い矩形
                p1, p2 = points[0], points[1]
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    length = 1.0
                direction = direction / length
                perp = np.array([-direction[1], direction[0]])
                vertices = np.array([
                    p1 - direction * padding + perp * padding,
                    p2 + direction * padding + perp * padding,
                    p2 + direction * padding - perp * padding,
                    p1 - direction * padding - perp * padding,
                ])
            hulls.append({
                "cluster_id": cid,
                "vertices": vertices,
                "n_points": n_points,
            })
            continue
        
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # パディングを適用（各頂点を中心から外側に押し出す）
            if padding > 0:
                center = hull_points.mean(axis=0)
                directions = hull_points - center
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                norms[norms < 1e-6] = 1.0
                unit_directions = directions / norms
                hull_points = hull_points + unit_directions * padding
            
            hulls.append({
                "cluster_id": cid,
                "vertices": hull_points,
                "n_points": n_points,
            })
        except Exception:
            # 凸包計算に失敗した場合（全ての点が同一直線上など）
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            vertices = np.array([
                [x_min - padding, y_min - padding],
                [x_max + padding, y_min - padding],
                [x_max + padding, y_max + padding],
                [x_min - padding, y_max + padding],
            ])
            hulls.append({
                "cluster_id": cid,
                "vertices": vertices,
                "n_points": n_points,
            })
    
    return hulls


def plot_annotations_overlay(
    image_path: Path | str,
    points_df: pd.DataFrame,
    img_id: int | str,
    output_path: Path | str,
    *,
    title: str | None = None,
    color: str = "red",
    edge_color: str = "darkred",
    figsize: tuple[int, int] = (12, 12),
    dpi: int = 150,
    point_size: int = 20,
) -> None:
    """生アノテーションのポイントを同一色でオーバーレイして保存.
    
    Args:
        image_path: 背景画像のパス
        points_df: アノテーションデータ (img_id, x, y)
        img_id: 対象画像のID
        output_path: 出力先パス
        title: 図のタイトル
        color: ポイントの色
        edge_color: ポイントのエッジ色
        figsize: 図のサイズ
        dpi: 解像度
        point_size: ポイントのサイズ
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    subset = _match_img_id(points_df, img_id)
    
    # デバッグ: データの状態を出力
    print(f"  [DEBUG] img_id={img_id}, points_df has {len(points_df)} rows, subset has {len(subset)} rows")
    if not subset.empty:
        print(f"  [DEBUG] x range: {subset['x'].min():.1f} - {subset['x'].max():.1f}")
        print(f"  [DEBUG] y range: {subset['y'].min():.1f} - {subset['y'].max():.1f}")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    
    if not subset.empty:
        # 画像サイズに対して適切な点サイズを計算
        img_height, img_width = img.shape[:2]
        # 点サイズを画像サイズに比例させる（500pxで100程度）
        auto_size = max(50, int(img_width / 5))
        actual_size = point_size if point_size > 50 else auto_size
        
        print(f"  [DEBUG] Image size: {img_width}x{img_height}, point_size={actual_size}")
        
        # 目立つ赤い点で描画（白いエッジで縁取り）
        ax.scatter(
            subset["x"].values, 
            subset["y"].values,
            c=color, 
            s=actual_size, 
            alpha=1.0,
            edgecolors="white", 
            linewidths=1.5,
            zorder=100,  # 最前面に
            marker='o',
        )
    else:
        print(f"  [WARNING] No points found for img_id={img_id}")
    
    title_text = title or "Raw Annotations"
    n_points = len(subset)
    ax.set_title(f"{title_text} ({n_points} points)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_bboxes_overlay(
    image_path: Path | str,
    df: pd.DataFrame,
    cluster_col: str,
    img_id: int | str,
    output_path: Path | str,
    *,
    title: str | None = None,
    padding: int = 3,
    figsize: tuple[int, int] = (10, 10),
    dpi: int = 150,
    seed: int = 42,
) -> None:
    """クラスタの矩形をオーバーレイして保存（レガシー用）.
    
    Args:
        image_path: 背景画像のパス
        df: クラスタ情報を含むDataFrame
        cluster_col: クラスタIDのカラム名
        img_id: 対象画像のID
        output_path: 出力先パス
        title: 図のタイトル
        padding: バウンディングボックスのパディング
        figsize: 図のサイズ
        dpi: 解像度
        seed: 色生成用乱数シード
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    bboxes = get_cluster_bboxes(df, cluster_col, img_id, padding=padding)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    
    rng = np.random.default_rng(seed)
    for bbox in bboxes:
        color = rng.random(3)
        rect = Rectangle(
            (bbox["x"], bbox["y"]), bbox["width"], bbox["height"],
            linewidth=1.5, edgecolor=color, facecolor=(*color, 0.2)
        )
        ax.add_patch(rect)
    
    title_text = title or f"{cluster_col}"
    ax.set_title(f"{title_text} ({len(bboxes)} clusters)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_hulls_overlay(
    image_path: Path | str,
    df: pd.DataFrame,
    cluster_col: str,
    img_id: int | str,
    output_path: Path | str,
    *,
    title: str | None = None,
    padding: float = 2.0,
    figsize: tuple[int, int] = (10, 10),
    dpi: int = 150,
    seed: int = 42,
    linewidth: float = 1.5,
    alpha: float = 0.25,
) -> None:
    """クラスタの凸包（外側の点をつないだ多角形）をオーバーレイして保存.
    
    Args:
        image_path: 背景画像のパス
        df: クラスタ情報を含むDataFrame
        cluster_col: クラスタIDのカラム名
        img_id: 対象画像のID
        output_path: 出力先パス
        title: 図のタイトル
        padding: 凸包のパディング（ピクセル）
        figsize: 図のサイズ
        dpi: 解像度
        seed: 色生成用乱数シード
        linewidth: 凸包の線幅
        alpha: 塗りつぶしの透明度
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    hulls = get_cluster_convex_hulls(df, cluster_col, img_id, padding=padding)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    
    rng = np.random.default_rng(seed)
    for hull_info in hulls:
        color = rng.random(3)
        vertices = hull_info["vertices"]
        
        # 多角形を描画
        polygon = Polygon(
            vertices,
            closed=True,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=(*color, alpha),
            zorder=5,
        )
        ax.add_patch(polygon)
    
    title_text = title or f"{cluster_col}"
    ax.set_title(f"{title_text} ({len(hulls)} clusters)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_qc_visualizations(
    qc_names: list[str],
    by_image_tag_dir: Path,
    fig_dir: Path,
    all_annotations: pd.DataFrame,
    hdbscan_cluster_df: pd.DataFrame,
    final_cluster_df: pd.DataFrame,
    worker_table: pd.DataFrame,
    heatmap_config: HeatmapConfig,
) -> None:
    """QC画像の各種可視化を出力.
    
    Args:
        qc_names: QC画像名のリスト (例: ["QC1.png", "QC2.png", ...])
        by_image_tag_dir: 画像ディレクトリのルート
        fig_dir: 出力先ディレクトリ
        all_annotations: 全アノテーションデータ
        hdbscan_cluster_df: HDBSCANクラスタリング結果
        final_cluster_df: GMM分割後クラスタデータ
        worker_table: ワーカー重みテーブル
        heatmap_config: ヒートマップ設定
    """
    import shutil
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    for qc_name in qc_names:
        qc_dir = by_image_tag_dir / qc_name
        image_path = qc_dir / qc_name
        
        if not image_path.exists():
            print(f"[SKIP] {qc_name} が見つかりません")
            continue
        
        # img_id (QC画像は整数)
        img_id = int(qc_name[2:-4])
        base_name = qc_name.replace(".png", "")
        
        print(f"Processing {qc_name}...")
        
        # 1. 生アノテーションのオーバーレイ（JSONから直接読み込み）
        plot_qc_raw_annotations(
            qc_dir,
            fig_dir / f"{base_name}_01_raw_annotations.png",
            title=f"{qc_name} - Raw Annotations",
            color="red",
            point_size=5,
        )
        
        # 2. HDBSCANクラスタの凸包オーバーレイ
        plot_cluster_hulls_overlay(
            image_path,
            hdbscan_cluster_df,
            "cluster_id",
            img_id,
            fig_dir / f"{base_name}_02_hdbscan_clusters.png",
            title=f"{qc_name} - HDBSCAN Clusters",
        )
        
        # 3. GMM分割後クラスタの凸包オーバーレイ
        plot_cluster_hulls_overlay(
            image_path,
            final_cluster_df,
            "final_cluster_id",
            img_id,
            fig_dir / f"{base_name}_03_final_clusters.png",
            title=f"{qc_name} - Final Clusters (after GMM split)",
        )
        
        # 4. ヒートマップのオーバーレイ
        heatmap_src = qc_dir / f"{qc_name}_heatmap_overlay.png"
        if heatmap_src.exists():
            shutil.copy(heatmap_src, fig_dir / f"{base_name}_04_heatmap.png")
        else:
            points_img = final_cluster_df[final_cluster_df["img_id"] == img_id]
            if not points_img.empty:
                image_array, heatmap = build_heatmap_for_image(
                    points_img[["x", "y", "worker_id"]],
                    worker_table,
                    image_path=image_path,
                    config=heatmap_config,
                )
                fig = overlay_heatmap(
                    image_array, heatmap,
                    alpha=heatmap_config.alpha,
                    cmap=heatmap_config.cmap,
                    title=f"{qc_name} - Weighted Heatmap",
                )
                fig.savefig(fig_dir / f"{base_name}_04_heatmap.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
    
    print(f"\nQC可視化出力完了: {fig_dir.as_posix()}")


def export_all_heatmaps(
    by_image_tag_dir: Path,
    cluster_df: pd.DataFrame,
    worker_table: pd.DataFrame,
    heatmap_config: HeatmapConfig,
) -> None:
    """全画像ディレクトリにヒートマップを出力.
    
    Args:
        by_image_tag_dir: 画像ディレクトリのルート
        cluster_df: クラスタデータ (final_cluster_id, x, y, worker_id, img_id)
        worker_table: ワーカー重みテーブル
        heatmap_config: ヒートマップ設定
    """
    image_dirs = sorted([d for d in by_image_tag_dir.iterdir() if d.is_dir()])
    print(f"処理対象画像数: {len(image_dirs)}")
    
    for img_dir in image_dirs:
        img_name = img_dir.name
        image_path = img_dir / img_name
        
        if not image_path.exists():
            print(f"[SKIP] 画像が見つかりません: {image_path}")
            continue
        
        # img_idはファイル名（文字列）をそのまま使用
        img_id = img_name
        
        points_img = cluster_df[cluster_df["img_id"] == img_id]
        if points_img.empty:
            print(f"[SKIP] アノテーションなし: {img_name}")
            continue
        
        # ヒートマップ生成
        image_array, heatmap = build_heatmap_for_image(
            points_img[["x", "y", "worker_id"]],
            worker_table,
            image_path=image_path,
            config=heatmap_config,
        )
        
        # .npyファイル出力
        npy_path = img_dir / f"{img_name}_heatmap.npy"
        np.save(npy_path, heatmap)
        
        # ヒートマップオーバーレイ画像出力
        fig = overlay_heatmap(
            image_array,
            heatmap,
            alpha=heatmap_config.alpha,
            cmap=heatmap_config.cmap,
            title=f"{img_name} heatmap",
        )
        overlay_path = img_dir / f"{img_name}_heatmap_overlay.png"
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    print("全画像のヒートマップ出力完了")


def _render_image_to_array(
    img: np.ndarray,
    draw_func: callable | None = None,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """画像をmatplotlibで描画し、余白なしのnumpy配列として返す.
    
    Args:
        img: 背景画像（numpy配列）
        draw_func: 追加の描画を行う関数 (ax を引数に取る)
        target_size: リサイズ先のサイズ (width, height)
    
    Returns:
        描画結果のnumpy配列 (RGB)
    """
    import io
    
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(img)
    
    if draw_func:
        draw_func(ax)
    
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    # 余白を完全に削除
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    
    # バッファに描画
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches=None, pad_inches=0)
    buf.seek(0)
    result = np.asarray(Image.open(buf).convert("RGB"))
    plt.close(fig)
    buf.close()
    
    # リサイズ
    if target_size and (result.shape[1], result.shape[0]) != target_size:
        result_img = Image.fromarray(result)
        result_img = result_img.resize(target_size, Image.LANCZOS)
        result = np.asarray(result_img)
    
    return result


def export_qc_summary_figure(
    qc_names: list[str],
    by_image_tag_dir: Path,
    fig_dir: Path,
    final_cluster_df: pd.DataFrame,
    worker_table: pd.DataFrame,
    heatmap_config: HeatmapConfig,
    *,
    cell_size: int = 500,
    dpi: int = 150,
    label_height: int = 40,
) -> None:
    """QC1-5のサマリー図（3列×5行）を出力.
    
    列: raw annotation, cluster (final), heatmap
    行: QC1, QC2, QC3, QC4, QC5
    
    PILで画像を直接連結し、余白なしで出力する。
    
    Args:
        qc_names: QC画像名のリスト (例: ["QC1.png", "QC2.png", ...])
        by_image_tag_dir: 画像ディレクトリのルート
        fig_dir: 出力先ディレクトリ
        final_cluster_df: GMM分割後クラスタデータ
        worker_table: ワーカー重みテーブル
        heatmap_config: ヒートマップ設定
        cell_size: 各セルの画像サイズ（正方形）
        dpi: 解像度
        label_height: 下部ラベル領域の高さ
    """
    from PIL import ImageDraw, ImageFont
    
    n_rows = len(qc_names)
    n_cols = 3
    target_size = (cell_size, cell_size)
    
    # 全体のキャンバスサイズ
    canvas_width = cell_size * n_cols
    canvas_height = cell_size * n_rows + label_height
    
    # 白いキャンバスを作成
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    
    for row_idx, qc_name in enumerate(qc_names):
        qc_dir = by_image_tag_dir / qc_name
        image_path = qc_dir / qc_name
        
        if not image_path.exists():
            print(f"[SKIP] {qc_name} が見つかりません")
            continue
        
        # img_id (QC画像は整数)
        img_id = int(qc_name[2:-4])
        
        # 画像を読み込み
        img = np.asarray(Image.open(image_path).convert("RGB"))
        
        # --- 列1: Raw Annotations ---
        points = load_annotations_from_json(qc_dir)
        def draw_raw(ax):
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax.scatter(xs, ys, c="red", s=3, alpha=0.9, edgecolors="none", zorder=100)
        
        raw_img = _render_image_to_array(img, draw_raw, target_size)
        canvas.paste(Image.fromarray(raw_img), (0, row_idx * cell_size))
        
        # --- 列2: Final Clusters ---
        hulls = get_cluster_convex_hulls(final_cluster_df, "final_cluster_id", img_id, padding=2.0)
        def draw_clusters(ax):
            rng = np.random.default_rng(42)
            for hull_info in hulls:
                color = rng.random(3)
                vertices = hull_info["vertices"]
                polygon = Polygon(
                    vertices,
                    closed=True,
                    linewidth=0.8,
                    edgecolor=color,
                    facecolor=(*color, 0.25),
                    zorder=5,
                )
                ax.add_patch(polygon)
        
        cluster_img = _render_image_to_array(img, draw_clusters, target_size)
        canvas.paste(Image.fromarray(cluster_img), (cell_size, row_idx * cell_size))
        
        # --- 列3: Heatmap ---
        heatmap_npy_path = qc_dir / f"{qc_name}_heatmap.npy"
        if heatmap_npy_path.exists():
            # .npyからヒートマップを読み込んでオーバーレイ
            heatmap_data = np.load(heatmap_npy_path)
            def draw_heatmap(ax):
                heatmap_normalized = heatmap_data / (heatmap_data.max() + 1e-8)
                ax.imshow(heatmap_normalized, cmap=heatmap_config.cmap, alpha=heatmap_config.alpha)
            heatmap_img = _render_image_to_array(img, draw_heatmap, target_size)
        else:
            # ヒートマップを生成
            points_img = final_cluster_df[final_cluster_df["img_id"] == img_id]
            if not points_img.empty:
                _, heatmap_data = build_heatmap_for_image(
                    points_img[["x", "y", "worker_id"]],
                    worker_table,
                    image_path=image_path,
                    config=heatmap_config,
                )
                def draw_heatmap(ax):
                    heatmap_normalized = heatmap_data / (heatmap_data.max() + 1e-8)
                    ax.imshow(heatmap_normalized, cmap=heatmap_config.cmap, alpha=heatmap_config.alpha)
                heatmap_img = _render_image_to_array(img, draw_heatmap, target_size)
            else:
                # データがない場合は元画像のみ
                heatmap_img = _render_image_to_array(img, None, target_size)
        
        canvas.paste(Image.fromarray(heatmap_img), (cell_size * 2, row_idx * cell_size))
    
    # 下部に列ラベルを追加
    draw = ImageDraw.Draw(canvas)
    col_labels = ["raw annotation", "cluster (candidate cells region)", "heatmap"]
    
    # フォント設定（システムフォントがない場合はデフォルト）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
    
    label_y = cell_size * n_rows + label_height // 2 - 8
    for col_idx, label in enumerate(col_labels):
        # テキストの幅を計算して中央揃え
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        x = col_idx * cell_size + (cell_size - text_width) // 2
        draw.text((x, label_y), label, fill=(0, 0, 0), font=font)
    
    # 保存
    output_path = fig_dir / "qc_summary.png"
    canvas.save(output_path, dpi=(dpi, dpi))
    
    print(f"QCサマリー図出力完了: {output_path.as_posix()}")


__all__ = [
    "get_cluster_bboxes",
    "get_cluster_convex_hulls",
    "load_annotations_from_json",
    "plot_annotations_overlay",
    "plot_qc_raw_annotations",
    "plot_cluster_bboxes_overlay",
    "plot_cluster_hulls_overlay",
    "export_qc_visualizations",
    "export_qc_summary_figure",
    "export_all_heatmaps",
]

