import ffmpeg

# URL of the streaming camera
stream_url = "http://giaothong.hochiminhcity.gov.vn/"

# Define output file name
output_file = "recorded_video.mp4"

# Run ffmpeg command to record the stream
ffmpeg.input(stream_url, t=6).output(output_file).run(overwrite_output=True)
