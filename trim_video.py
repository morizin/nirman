# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "moviepy",
# ]
# ///

from moviepy.editor import VideoFileClip


def trim_video(input_file: str, output_file: str, start_time: float, end_time: float):
    video = VideoFileClip(input_file)
    trimmed = video.subclip(start_time, end_time)
    trimmed.write_videofile(output_file, codec="libx264")
    video.close()


# Example: trim from 10s to 30s
trim_video(
    "../DASH CAM 2016 01 29 (42 Miles of Potholes) [BQo87tGRM74].mkv",
    "output.mkv",
    start_time=10,
    end_time=30,
)
