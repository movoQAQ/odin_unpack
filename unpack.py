import argparse
import csv
import json
import math
import os
import sqlite3
import struct
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# -----------------------------
# Basic helpers
# -----------------------------

def safe_topic_name(topic: str) -> str:
    name = topic.strip("/")
    if not name:
        return "root"
    return name.replace("/", "__").replace("\\", "__")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bytes_from_ros_array(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, np.ndarray):
        return data.tobytes()
    return bytes(data)


def jsonable(obj: Any, max_array_items: int = 200) -> Any:
    """
    Convert rosbags message object to JSON-safe object.
    Large numeric arrays are truncated to avoid huge JSON files.
    """
    if is_dataclass(obj):
        return jsonable(asdict(obj), max_array_items=max_array_items)

    if isinstance(obj, dict):
        return {str(k): jsonable(v, max_array_items=max_array_items) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        if len(obj) > max_array_items:
            return {
                "_truncated": True,
                "_length": len(obj),
                "head": [jsonable(x, max_array_items=max_array_items) for x in obj[:max_array_items]],
            }
        return [jsonable(x, max_array_items=max_array_items) for x in obj]

    if isinstance(obj, np.ndarray):
        flat = obj.reshape(-1)
        if flat.size > max_array_items:
            return {
                "_truncated": True,
                "_shape": list(obj.shape),
                "_dtype": str(obj.dtype),
                "head": flat[:max_array_items].tolist(),
            }
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {
            "_bytes": True,
            "_length": len(obj),
        }

    return obj


# -----------------------------
# SQLite rosbag2 reader
# -----------------------------

def read_topics(conn: sqlite3.Connection) -> Dict[int, Dict[str, str]]:
    cursor = conn.cursor()

    # Different rosbag2 versions may have extra columns.
    rows = cursor.execute("PRAGMA table_info(topics)").fetchall()
    columns = [r[1] for r in rows]

    required = {"id", "name", "type", "serialization_format"}
    missing = required - set(columns)
    if missing:
        raise RuntimeError(f"topics table missing columns: {missing}")

    result = {}
    for row in cursor.execute("SELECT id, name, type, serialization_format FROM topics"):
        tid, name, typ, ser = row
        result[int(tid)] = {
            "name": name,
            "type": typ,
            "serialization_format": ser,
        }
    return result


def iter_messages(
    conn: sqlite3.Connection,
    topics_filter: Optional[set],
) -> Iterable[Tuple[int, int, bytes]]:
    cursor = conn.cursor()

    if topics_filter is None:
        query = "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
        for topic_id, timestamp, data in cursor.execute(query):
            yield int(topic_id), int(timestamp), data
    else:
        placeholders = ",".join("?" for _ in topics_filter)
        query = f"""
            SELECT topic_id, timestamp, data
            FROM messages
            WHERE topic_id IN ({placeholders})
            ORDER BY timestamp
        """
        for topic_id, timestamp, data in cursor.execute(query, list(topics_filter)):
            yield int(topic_id), int(timestamp), data


def write_topic_info(db3_path: Path, out_dir: Path, topics: Dict[int, Dict[str, str]]) -> None:
    conn = sqlite3.connect(str(db3_path))
    cursor = conn.cursor()

    counts = dict(cursor.execute(
        "SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"
    ).fetchall())

    with (out_dir / "topic_info.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic_id", "topic_name", "type", "serialization_format", "message_count"])
        for tid, meta in sorted(topics.items()):
            writer.writerow([
                tid,
                meta["name"],
                meta["type"],
                meta["serialization_format"],
                counts.get(tid, 0),
            ])

    conn.close()


# -----------------------------
# Typestore and custom msgs
# -----------------------------

def make_typestore(distro: str):
    distro = distro.lower().strip()

    if distro in {"latest", "auto"}:
        store = Stores.LATEST
    else:
        store_name = f"ROS2_{distro.upper()}"
        if not hasattr(Stores, store_name):
            valid = [x for x in dir(Stores) if x.startswith("ROS2_")] + ["LATEST"]
            raise RuntimeError(f"Unknown distro '{distro}'. Valid examples: {valid}")
        store = getattr(Stores, store_name)

    return get_typestore(store)


def register_custom_msg_root(typestore, root: Optional[Path]) -> None:
    """
    Scan a directory like:
        C:\\workspace\\src\\robot_msgs\\msg\\RobotStatus.msg

    and register every pkg/msg/Name.msg found under it.
    """
    if root is None:
        return

    if not root.exists():
        raise FileNotFoundError(f"custom msg root not found: {root}")

    add_types = {}

    msg_files = list(root.rglob("*.msg"))
    for msg_file in msg_files:
        parts = msg_file.parts

        if "msg" not in parts:
            continue

        msg_index = parts.index("msg")
        if msg_index == 0:
            continue

        pkg_name = parts[msg_index - 1]
        msg_name = msg_file.stem
        msg_type = f"{pkg_name}/msg/{msg_name}"

        text = msg_file.read_text(encoding="utf-8")
        add_types.update(get_types_from_msg(text, msg_type))

    if add_types:
        typestore.register(add_types)
        print(f"[OK] Registered custom msg types: {len(add_types)}")
    else:
        print("[WARN] No .msg files found for custom registration.")


# -----------------------------
# Image export
# -----------------------------

def export_image_msg(msg: Any, out_path: Path) -> bool:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    raw = bytes_from_ros_array(msg.data)

    if height <= 0 or width <= 0:
        return False

    try:
        if encoding in {"rgb8", "8uc3"}:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            img = PILImage.fromarray(arr, mode="RGB")

        elif encoding == "bgr8":
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            arr = arr[:, :, ::-1]
            img = PILImage.fromarray(arr, mode="RGB")

        elif encoding in {"rgba8"}:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
            img = PILImage.fromarray(arr, mode="RGBA")

        elif encoding in {"bgra8"}:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
            arr = arr[:, :, [2, 1, 0, 3]]
            img = PILImage.fromarray(arr, mode="RGBA")

        elif encoding in {"mono8", "8uc1"}:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
            img = PILImage.fromarray(arr, mode="L")

        elif encoding in {"mono16", "16uc1"}:
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(height, width)
            img = PILImage.fromarray(arr)

        else:
            print(f"[WARN] Unsupported image encoding: {encoding}")
            return False

        img.save(out_path)
        return True

    except Exception as exc:
        print(f"[WARN] Failed to export image {out_path}: {exc}")
        return False


# -----------------------------
# PointCloud2 export
# -----------------------------

POINTFIELD_DTYPE = {
    1: ("b", 1),   # INT8
    2: ("B", 1),   # UINT8
    3: ("h", 2),   # INT16
    4: ("H", 2),   # UINT16
    5: ("i", 4),   # INT32
    6: ("I", 4),   # UINT32
    7: ("f", 4),   # FLOAT32
    8: ("d", 8),   # FLOAT64
}


def get_cloud_fields(msg: Any) -> List[Tuple[str, int, int, int]]:
    """
    Return list of scalar fields: (name, offset, datatype, count).
    Only scalar numeric fields are used.
    """
    fields = []
    for field in msg.fields:
        name = str(field.name)
        offset = int(field.offset)
        datatype = int(field.datatype)
        count = int(field.count)

        if datatype not in POINTFIELD_DTYPE:
            continue

        # PCD ASCII here only exports scalar fields. If count > 1, expand name_0, name_1 ...
        fields.append((name, offset, datatype, count))

    return fields


def export_pointcloud2_msg(msg: Any, out_path: Path) -> bool:
    width = int(msg.width)
    height = int(msg.height)
    point_step = int(msg.point_step)
    raw = bytes_from_ros_array(msg.data)
    endian = ">" if bool(msg.is_bigendian) else "<"

    fields = get_cloud_fields(msg)
    if not fields:
        print(f"[WARN] No usable fields in PointCloud2: {out_path}")
        return False

    expanded_names = []
    expanded_specs = []

    for name, offset, datatype, count in fields:
        fmt, _size = POINTFIELD_DTYPE[datatype]
        for i in range(count):
            field_name = name if count == 1 else f"{name}_{i}"
            expanded_names.append(field_name)
            expanded_specs.append((offset + i * _size, endian + fmt))

    # Prefer x y z first if available, then others.
    order = []
    for wanted in ["x", "y", "z", "intensity", "ring", "time"]:
        if wanted in expanded_names:
            order.append(expanded_names.index(wanted))

    for i in range(len(expanded_names)):
        if i not in order:
            order.append(i)

    expanded_names = [expanded_names[i] for i in order]
    expanded_specs = [expanded_specs[i] for i in order]

    point_count = width * height

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS " + " ".join(expanded_names) + "\n")
        f.write("SIZE " + " ".join("4" for _ in expanded_names) + "\n")
        f.write("TYPE " + " ".join("F" for _ in expanded_names) + "\n")
        f.write("COUNT " + " ".join("1" for _ in expanded_names) + "\n")
        f.write(f"WIDTH {point_count}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {point_count}\n")
        f.write("DATA ascii\n")

        valid_count = 0
        for idx in range(point_count):
            base = idx * point_step
            values = []

            try:
                for offset, fmt in expanded_specs:
                    value = struct.unpack_from(fmt, raw, base + offset)[0]
                    if isinstance(value, float) and not math.isfinite(value):
                        value = 0.0
                    values.append(float(value))
            except Exception:
                continue

            f.write(" ".join(f"{v:.8g}" for v in values) + "\n")
            valid_count += 1

    return valid_count > 0


# -----------------------------
# Main unpack logic
# -----------------------------

def unpack_db3(
    db3_path: Path,
    out_dir: Path,
    distro: str,
    only_topics: Optional[List[str]],
    max_per_topic: Optional[int],
    custom_msg_root: Optional[Path],
) -> None:
    ensure_dir(out_dir)

    conn = sqlite3.connect(str(db3_path))
    topics = read_topics(conn)
    write_topic_info(db3_path, out_dir, topics)

    topic_name_to_id = {v["name"]: k for k, v in topics.items()}

    if only_topics:
        missing = [t for t in only_topics if t not in topic_name_to_id]
        if missing:
            print(f"[WARN] Topics not found in db3: {missing}")

        topic_ids = {topic_name_to_id[t] for t in only_topics if t in topic_name_to_id}
    else:
        topic_ids = None

    typestore = make_typestore(distro)
    register_custom_msg_root(typestore, custom_msg_root)

    skipped_csv = (out_dir / "skipped_messages.csv").open("w", newline="", encoding="utf-8")
    skipped_writer = csv.writer(skipped_csv)
    skipped_writer.writerow(["timestamp", "topic", "type", "reason"])

    counts: Dict[str, int] = {}
    jsonl_files: Dict[str, Any] = {}

    try:
        for topic_id, timestamp, rawdata in iter_messages(conn, topic_ids):
            meta = topics[topic_id]
            topic = meta["name"]
            msgtype = meta["type"]

            current = counts.get(topic, 0)
            if max_per_topic is not None and current >= max_per_topic:
                continue

            topic_dir = out_dir / safe_topic_name(topic)
            ensure_dir(topic_dir)

            try:
                msg = typestore.deserialize_cdr(rawdata, msgtype)
            except Exception as exc:
                skipped_writer.writerow([timestamp, topic, msgtype, f"deserialize failed: {exc}"])
                continue

            index = counts.get(topic, 0)

            if msgtype == "sensor_msgs/msg/Image":
                filename = topic_dir / f"{index:06d}_{timestamp}.png"
                ok = export_image_msg(msg, filename)
                if not ok:
                    skipped_writer.writerow([timestamp, topic, msgtype, "image export failed"])

            elif msgtype == "sensor_msgs/msg/PointCloud2":
                filename = topic_dir / f"{index:06d}_{timestamp}.pcd"
                ok = export_pointcloud2_msg(msg, filename)
                if not ok:
                    skipped_writer.writerow([timestamp, topic, msgtype, "pointcloud export failed"])

            else:
                jsonl_path = topic_dir / "messages.jsonl"
                if topic not in jsonl_files:
                    jsonl_files[topic] = jsonl_path.open("a", encoding="utf-8")

                record = {
                    "timestamp": timestamp,
                    "topic": topic,
                    "type": msgtype,
                    "msg": jsonable(msg),
                }
                jsonl_files[topic].write(json.dumps(record, ensure_ascii=False) + "\n")

            counts[topic] = index + 1

    finally:
        for f in jsonl_files.values():
            f.close()
        skipped_csv.close()
        conn.close()

    print("\n[OK] Unpack finished.")
    print(f"Output directory: {out_dir.resolve()}")
    print("\nExported message counts:")
    for topic, count in sorted(counts.items()):
        print(f"  {topic}: {count}")

    print("\nTopic summary written to:")
    print(f"  {out_dir / 'topic_info.csv'}")
    print("Skipped message log written to:")
    print(f"  {out_dir / 'skipped_messages.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unpack ROS2 rosbag2 sqlite3 .db3 on Windows without ROS2."
    )
    parser.add_argument(
        "db3",
        type=str,
        help="Path to rosbag2 .db3 file, e.g. playable_bag_0.db3",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="unpacked_output",
        help="Output directory.",
    )
    parser.add_argument(
        "--distro",
        type=str,
        default="jazzy",
        help="ROS2 typestore: jazzy, humble, iron, foxy, latest, etc.",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Only export selected topics. Example: --topics /odin1/image/undistorted /odin1/cloud_slam",
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=None,
        help="Limit exported messages per topic for testing.",
    )
    parser.add_argument(
        "--custom-msg-root",
        type=str,
        default=None,
        help="Folder containing custom .msg files, e.g. C:\\ws\\src",
    )

    args = parser.parse_args()

    db3_path = Path(args.db3)
    if not db3_path.exists():
        raise FileNotFoundError(f"db3 file not found: {db3_path}")

    out_dir = Path(args.out)
    custom_msg_root = Path(args.custom_msg_root) if args.custom_msg_root else None

    unpack_db3(
        db3_path=db3_path,
        out_dir=out_dir,
        distro=args.distro,
        only_topics=args.topics,
        max_per_topic=args.max_per_topic,
        custom_msg_root=custom_msg_root,
    )


if __name__ == "__main__":
    main()