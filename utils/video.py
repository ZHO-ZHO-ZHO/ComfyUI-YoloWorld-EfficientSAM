import os
import datetime
import uuid

import supervision as sv


MAX_VIDEO_LENGTH_SEC = 3


def generate_file_name(extension="mp4"):
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4()
    return f"{current_datetime}_{unique_id}.{extension}"


def calculate_end_frame_index(source_video_path: str) -> int:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return min(
        video_info.total_frames,
        video_info.fps * MAX_VIDEO_LENGTH_SEC
    )


def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
