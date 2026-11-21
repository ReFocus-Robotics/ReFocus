"""Package evaluation camera frames into timeline montages for the Scheduler.

Frames come from observations the policy already receives; nothing is re-rendered.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np


TIMELINE_SNAPSHOTS = 16
CAMERA_TILE = 128


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _lookup(observation, key):
    if isinstance(observation, dict):
        if key in observation:
            return observation[key]
        if key.startswith("video."):
            nested = observation.get("video")
            if isinstance(nested, dict) and key[6:] in nested:
                return nested[key[6:]]
    return None


def latest_rgb(value):
    """Return the current HxWx3 uint8 frame from a frame, stack, or batch of one."""
    if value is None:
        return None
    array = _to_numpy(value)
    while array.ndim > 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and (array.shape[-1] in (3, 4) or array.shape[1] in (3, 4)):
        array = array[-1]
    if array.ndim != 3:
        return None
    if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        return None
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def observation_views(observation, cameras):
    """Pick the first available key for each ``(label, keys)`` camera."""
    views = {}
    for label, keys in cameras:
        for key in keys:
            image = latest_rgb(_lookup(observation, key))
            if image is not None:
                views[label] = image
                break
    return views


def extract_instruction(observation):
    """Read an instruction already present in the observation, never invent one."""
    target_names = {'instruction', 'language_instruction', 'language'}
    stack = [observation]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower().split('.')[-1] in target_names:
                    if isinstance(child, (list, tuple)) and child:
                        child = child[0]
                    elif hasattr(child, 'shape') and getattr(child, 'size', 0) == 1:
                        child = _to_numpy(child).reshape(-1)[0]
                    if hasattr(child, 'item') and not isinstance(child, (str, bytes)):
                        try:
                            child = child.item()
                        except (ValueError, TypeError):
                            pass
                    if isinstance(child, bytes):
                        child = child.decode('utf-8', errors='replace')
                    if isinstance(child, str) and child.strip():
                        return child.strip()
                if isinstance(child, dict):
                    stack.append(child)
    return None


def should_capture(step_index: int, horizon_steps: int, count: int = TIMELINE_SNAPSHOTS):
    """Deterministically choose up to 16 approximately even timeline points."""
    if step_index < 0 or horizon_steps < 0:
        return False
    targets = {int(round(v)) for v in np.linspace(0, horizon_steps, count)}
    return step_index in targets


def encode_snapshot(views, control_step: int, stage_index: int):
    """Store each camera view as a small JPEG."""
    from PIL import Image

    result = {"control_step": int(control_step), "stage_index": int(stage_index), "views": {}}
    for label, array in views.items():
        image = Image.fromarray(array).resize((CAMERA_TILE, CAMERA_TILE), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72, optimize=True)
        result["views"][label] = buffer.getvalue()
    return result


def _font(draw):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        return ImageFont.load_default()


def build_montage(snapshots, *, task: str, update: int, success: bool,
                  episode_slot: int | None = None, episode_count: int = 5,
                  control_count: int | None = None):
    """Build a 4x4 timeline; each time cell is a 2x2 panel of the recorded cameras."""
    from PIL import Image, ImageDraw

    header_h, caption_h = 28, 22
    tile = CAMERA_TILE
    panel_w, panel_h = tile * 2, tile * 2
    cell_w, cell_h = panel_w, panel_h + caption_h
    canvas = Image.new("RGB", (cell_w * 4, header_h + cell_h * 4), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = _font(draw)
    label = "SUCCESS" if success else "FAILURE"
    episode = (f"Episode {episode_slot + 1}/{episode_count} | "
               if episode_slot is not None else "")
    controls = (int(control_count) if control_count is not None else
                max((int(item.get("control_step", 0)) for item in snapshots), default=0))
    draw.text((8, 6), f"{task} | Eval update {update} | {episode}{label} | {controls} controls",
              fill=(255, 255, 255), font=font)
    names = list(dict.fromkeys(name for item in snapshots for name in item["views"]))[:4]
    for index in range(16):
        row, col = divmod(index, 4)
        left, top = col * cell_w, header_h + row * cell_h
        snapshot = snapshots[index] if index < len(snapshots) else None
        for panel in range(4):
            px, py = left + (panel % 2) * tile, top + (panel // 2) * tile
            name = names[panel] if panel < len(names) else None
            raw = snapshot["views"].get(name) if snapshot and name else None
            if raw:
                canvas.paste(Image.open(io.BytesIO(raw)).convert("RGB"), (px, py))
            else:
                draw.rectangle((px, py, px + tile - 1, py + tile - 1), fill=(44, 44, 44))
            if name:
                draw.rectangle((px, py, px + tile - 1, py + 16), fill=(0, 0, 0))
                draw.text((px + 3, py + 2), name, fill=(255, 255, 255), font=font)
        if snapshot:
            caption = f"step {snapshot['control_step']} | stage {snapshot['stage_index']}"
        else:
            caption = "no frame"
        draw.text((left + 4, top + panel_h + 3), caption, fill=(255, 255, 255), font=font)
    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=80, optimize=True)
    return output.getvalue()


def save_episode_montage(root, round_id, task, update, slot, snapshots, success,
                         control_count=None):
    root = Path(root)
    relative = Path("visual_feedback") / f"round_{int(round_id)}" / f"update_{int(update):03d}" / task / f"episode_{int(slot):04d}.jpg"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_montage(snapshots, task=task, update=update, success=success,
                            episode_slot=slot, control_count=control_count)
    temporary = target.with_suffix(".jpg.tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return relative.as_posix()


def image_cases(root, round_id, task, checkpoint_steps, rows):
    """Resolve exactly five complete episodes and montage files per checkpoint."""
    root = Path(root).resolve()
    cases = []
    for update in checkpoint_steps:
        matches = [row for row in rows if row.get("task") == task and row.get("update") == update]
        if len(matches) != 5:
            raise ValueError(f"Expected five complete eval episodes for {task} at {update}, found {len(matches)}")
        matches.sort(key=lambda row: row.get("slot", 0))
        for row in matches:
            if not row.get("complete"):
                raise ValueError(f"Incomplete eval trace for {task} at update {update}")
            relative = row.get("visual_montage")
            if not relative:
                raise ValueError(f"Missing visual montage for {task} at update {update}")
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"Invalid or missing visual montage for {task} at update {update}")
            cases.append({
                "case_id": f"{task}-u{int(update):03d}-slot{int(row.get('slot', 0)):04d}",
                "update": int(update),
                "success": bool(row["success"]),
                "stage": int(row.get("stage", 0)),
                "image_path": path,
            })
    return cases
