import argparse
import threading
import os
import select
import sys


import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GstRtspServer, GLib


def check_for_keypress(loop):
    """Check for keypress and stop the loop if detected."""
    print("Press any key to stop the server...")
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
            loop.quit()
            break


def stop_loop_after_timeout(loop, timeout):
    """Stop the loop after a certain timeout."""
    def stop_loop():
        print(f"Timeout reached: {timeout} seconds. Stopping the server...")
        loop.quit()
        return False  # Returning False stops the timeout callback

    GLib.timeout_add_seconds(timeout, stop_loop)


def main():
    parser = argparse.ArgumentParser(description='RTSP Server')
    parser.add_argument('--video-path', required=True, help='Path to the video file to stream')
    parser.add_argument('--ip', default='127.0.0.1', help='IP address to bind the RTSP server to')
    parser.add_argument('--port', default='3010', help='Port for the RTSP server')
    parser.add_argument('--endpoint', default='ds-rtsp-input', help='RTSP endpoint name')
    parser.add_argument('--timeout', default=60*5, type=int, help='Server timeout in seconds')
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f'File {args.video_path} not found')

    # Initialize GStreamer
    Gst.init(None)

    # Create an instance of the RTSP server
    server = GstRtspServer.RTSPServer.new()
    server.set_service(args.port)
    server.set_address(args.ip)

    # Create a default media factory
    factory = GstRtspServer.RTSPMediaFactory.new()
    # factory.set_launch(f'( filesrc location={args.video_path} loop=true ! qtdemux ! h264parse ! rtph264pay name=pay0 pt=96 )')
    factory.set_launch(f'( filesrc location={args.video_path} loop=true ! qtdemux ! rtph264pay name=pay0 )')

    server.get_mount_points().add_factory(f"/{args.endpoint}", factory)

    # Start the server
    server.attach(None)

    # Enter a main loop
    loop = GLib.MainLoop()

    # Start a thread to listen for a keypress
    threading.Thread(target=check_for_keypress, args=(loop,), daemon=True).start()

    # Optionally, stop the loop after a timeout
    if args.timeout:
        stop_loop_after_timeout(loop, args.timeout)

    print(f"RTSP server is running at rtsp://{args.ip}:{args.port}/{args.endpoint}")
    print("You can verify the stream using VLC media player or the following command: ")
    print(f"    gst-launch-1.0 rtspsrc location= rtsp://{args.ip}:{args.port}/{args.endpoint} latency=100 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink")

    loop.run()


if __name__ == '__main__':
    main()
