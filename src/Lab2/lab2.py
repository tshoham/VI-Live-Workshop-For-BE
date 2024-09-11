import argparse
from collections import Counter
import configparser
import math
import os
from pathlib import Path
import sys

import gi
gi.require_version("Gst", "1.0")  # This is reqired before importing Gst
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
import pyds

from src.common.bus_call import bus_call
from src.common.platform_info import PlatformInfo
from src.common.FPS import PERF_DATA
from src.constants import DEEPSTREAM_CONFIGS_DIR, OUTPUT_DIR
from src.common.utils import check_and_normalize_inputs, get_file_name_no_ext, generate_filename
from src.common.gst_utils import create_element, create_sink, create_source_bin, get_tracker, SinkConfig, SrcConfig, \
    verify_component
from src.common.insights_utils import FrameData, BBoxData, read_tracker_config, write_insights_on_cleanup, generate_buffer_main_key

PGIE_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest3_pgie_config.txt")
assert os.path.exists(PGIE_CONFIG_PATH), f"File {PGIE_CONFIG_PATH} does not exist"

TRACKER_CONFIG_FILE = "dstest2_tracker_config.txt"

silent = False  # Must be global since used in callback
perf_data = None
measure_latency = False
insights_data = {} # {stream_id&name: [FrameData]}
stream_metadata = {} # {stream_id&name: {frameWidth, frameHeight}}
insights_output_path = None # Must be global since used in callback

MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000

TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720

OSD_PROCESS_MODE = 0  # 0: CPU mode, 1: GPU mode (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsosd.html)
OSD_DISPLAY_TEXT = 1
OSD_DISPLAY_CLOCK = 1

def get_metadata_from_buffer(batch_meta, u_data, show_display=False, timestamp_type=None):
    global insights_data
    global stream_metadata

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            # see https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html
            # for more details on the NvDsFrameMeta object
        except StopIteration:
            break

        # stream info
        stream_id = frame_meta.pad_index
        stream_name = Path(u_data.input_uris[stream_id]).stem

        # frame number and dimensions
        frame_number = frame_meta.frame_num + 1
        frame_width = frame_meta.source_frame_width
        frame_height = frame_meta.source_frame_height

        main_key = generate_buffer_main_key(stream_id, stream_name)
        if main_key not in stream_metadata:
            stream_metadata[main_key] = {
                "frameWidth": frame_width,
                "frameHeight": frame_height
            }

        frame_data = FrameData(frame=frame_number)

        # objects data in current frame
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        if not silent:
            text_display = f"Stream={stream_name} Frame={frame_number} #Objects={num_rects}"
            print(text_display)

        obj_counter = Counter()

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                # see https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html
                # for more details on the NvDsObjectMeta object
            except StopIteration:
                break

            # obj_meta.rect_params contains the most updated coordinates of an object (after tracker, etc.)
            bbox_data = BBoxData(obj_meta.object_id,
                obj_meta.obj_label,
                frame_number,
                obj_meta.rect_params.left / frame_width,
                obj_meta.rect_params.top / frame_height,
                obj_meta.rect_params.width / frame_width,
                obj_meta.rect_params.height / frame_height,
                obj_meta.confidence)
            frame_data.bboxes.append(bbox_data)

            if u_data.raw_detections:
                detector_bbox_coords = obj_meta.detector_bbox_info.org_bbox_coords
                bbox_detector = BBoxData(
                    obj_meta.object_id,
                    obj_meta.obj_label,
                    frame_number,
                    detector_bbox_coords.left / frame_width,
                    detector_bbox_coords.top / frame_height,
                    detector_bbox_coords.width / frame_width,
                    detector_bbox_coords.height / frame_height,
                    obj_meta.confidence
                )
                frame_data.detector_bboxes.append(bbox_detector)

            obj_counter[obj_meta.obj_label] += 1

            # add confidence to the display text of object
            obj_meta.text_params.display_text = f"{obj_meta.obj_label} {obj_meta.confidence:.2f}"

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        insights_data[main_key].append(frame_data)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

def tracker_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    global measure_latency
    if measure_latency:
        num_sources_in_batch = pyds.nvds_measure_buffer_latency(hash(gst_buffer))
        if num_sources_in_batch == 0:
            print("Unable to get number of sources in GstBuffer for latency measurement")

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    # NvDsBatchMeta -> NvDsFrameMeta -> NvDsObjectMeta
    # go over the list of frame_meta in batch_meta, and then the list of object_meta in each frame_meta
    get_metadata_from_buffer(batch_meta, u_data)

    return Gst.PadProbeReturn.OK

def start_rtsp_streaming(sink_config: SinkConfig):
    server = GstRtspServer.RTSPServer.new()
    server.props.service = str(sink_config.rtsp_port)
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(f"( udpsrc name=pay0 port={sink_config.udp_port} buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string){sink_config.codec}, payload=96 \" )")
    factory.set_shared(True)
    server.get_mount_points().add_factory(f"/{sink_config.mount_point}", factory)

    print(f"\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:{sink_config.rtsp_port}/{sink_config.mount_point} ***\n\n")

def create_elements(pipeline, sink_config: SinkConfig):
    # Create nvstreammux instance to form batches from one or more sources
    streammux = create_element("nvstreammux", "Stream-muxer", pipeline)
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    
    queue1 = create_element("queue", "queue1", pipeline)
    queue2 = create_element("queue", "queue2", pipeline)
    queue3 = create_element("queue", "queue3", pipeline)
    queue4 = create_element("queue", "queue4", pipeline)
    queue5 = create_element("queue", "queue5", pipeline)
    queue6 = create_element("queue", "queue6", pipeline)

    print("Creating pgie\n")
    pgie = create_element("nvinfer", "primary-inference", pipeline)
    pgie.set_property("config-file-path", PGIE_CONFIG_PATH)

    tracker = get_tracker(ds_configs_dir=DEEPSTREAM_CONFIGS_DIR, tracker_config_file=TRACKER_CONFIG_FILE,
                          container=pipeline)

    tiler = create_element("nvmultistreamtiler", "nvtiler", pipeline)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    
    nvvidconv = create_element("nvvideoconvert", "convertor", pipeline)
    nvosd = create_element("nvdsosd", "onscreendisplay", pipeline)

    nvosd.set_property("process-mode", OSD_PROCESS_MODE)
    nvosd.set_property("display-text", OSD_DISPLAY_TEXT)
    nvosd.set_property("display-clock", OSD_DISPLAY_CLOCK)

    sink = create_sink(sink_config, pipeline)

    return streammux, queue1, queue2, queue3, queue4, queue5, queue6, pgie, tracker, tiler, nvvidconv, nvosd, sink

def create_and_link_sources(src_config: SrcConfig, number_sources, streammux, pipeline):
    for i in range(number_sources):
        print(f"Creating source_bin {i}\n")
        uri_name = src_config.input_uris[i]
        
        source_bin = create_source_bin(src_config, i)
        verify_component(source_bin, "Source bin")

        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        verify_component(sinkpad, "Streammux sink pad")

        srcpad = source_bin.get_static_pad("src")
        verify_component(srcpad, "Source bin src pad")

        srcpad.link(sinkpad)

def main(src_config: SrcConfig, sink_config: SinkConfig,
         write_insights=False):
    global insights_data

    for input_id, input_uri in enumerate(src_config.input_uris):
        stream_name = Path(input_uri).stem
        insights_data[generate_buffer_main_key(input_id, stream_name)] = []

    global perf_data
    perf_data = PERF_DATA(len(src_config.input_uris))

    number_sources = len(src_config.input_uris)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline\n")
    pipeline = Gst.Pipeline()
    verify_component(pipeline, "Pipeline")

    streammux, queue1, queue2, queue3, queue4, queue5, queue6, pgie, tracker, tiler, nvvidconv, nvosd, sink = create_elements(pipeline, sink_config)

    # # Batching for multiple camera inputs for streamux and pgie
    streammux.set_property("batch-size", number_sources)
    
    print("Adding batch size to pgie according to number of sources\n")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(f"WARNING: Overriding infer-config batch-size {pgie_batch_size} with {number_sources=}\n")
        pgie.set_property("batch-size", number_sources)
        
    # Tiler properties for output display accoring to number of inputs
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)

    create_and_link_sources(src_config, number_sources, streammux, pipeline)

    print("Linking elements in the Pipeline\n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(tiler)
    tiler.link(queue4)
    queue4.link(nvvidconv)
    nvvidconv.link(queue5)
    queue5.link(nvosd)
    nvosd.link(queue6)
    queue6.link(sink)

    # Create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    if sink_config.type == "rtsp":
        start_rtsp_streaming(sink_config)

    tracker_src_pad = tracker.get_static_pad("src")
    verify_component(tracker_src_pad, "Tracker src pad")
    tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, src_config)
    # perf callback function to print fps every 5 sec
    GLib.timeout_add(5000, perf_data.perf_print_callback)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(src_config.input_uris):
        print(f"{i=}, {source=}")

    print("Starting pipeline\n")
    # Start playback and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # Cleanup
    print("Exiting app\n")

    # Write insights to file on cleanup, add the tracker data to the insights
    if write_insights:
        tracker_config_path = str(DEEPSTREAM_CONFIGS_DIR / TRACKER_CONFIG_FILE)
        tracker_data = read_tracker_config(tracker_config_path)
        write_insights_on_cleanup(insights_data, stream_metadata, insights_output_path, src_config.raw_detections,
                                  tracker_data)

    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description="deepstream_example multistream, inference reference app")
    parser.add_argument("-i", "--input", help="Path to input streams", nargs="+", metavar="URIs", required=True)

    # Section for sink type
    parser.add_argument("--sink-type", default="file", choices=SinkConfig.VALID_SINK_TYPES, help="Choose the sink type")
    parser.add_argument("-o", "--output-path", help="Path to save the output file")

    # Section for RTSP output Streaming
    parser.add_argument("-p", "--rtsp-port", default=3000, help="Set the RTSP Streaming Port", type=int)  # 8554
    parser.add_argument("-m", "--mount-point", default="ds-test", help="Set the RTSP Streaming Mount Point")
    parser.add_argument("--udp-port", default=5400, help="Set the UDP Port for RTSP Streaming", type=int)
    parser.add_argument("-cc", "--codec", default="H264", help="RTSP Streaming Codec H264/H265",
                        choices=SinkConfig.VALID_CODECS)
    parser.add_argument("-b", "--bitrate", default=4000000, help="Set the encoding bitrate ", type=int)
    parser.add_argument("-e", "--enc-type", default='hw', help="hw:Hardware encoder, sw:Software encoder",
                        choices=SinkConfig.VALID_ENC_TYPES)

    parser.add_argument("--output-frames-dir", help="Path to save the output frames")

    parser.add_argument("--write-insights", action="store_true", default=False, help="Write insights to a file")

    parser.add_argument("--memtype", type=int, choices=SrcConfig.VALID_CUDA_MEMORY_TYPES, default=-1,
                        help="Decoder CUDA memory type (-1: Do not set, 0: NVBUF_MEM_CUDA_DEVICE, 1: NVBUF_MEM_CUDA_PINNED, 2: NVBUF_MEM_CUDA_UNIFIED)")
    parser.add_argument("--raw-detections", action="store_true", default=False,
                        help="Enable raw detections debug (write raw detections to file)")
    
    parser.add_argument("-s", "--silent", action="store_true", default=False, help="Disable verbose output")
    args = parser.parse_args()

    args.input = check_and_normalize_inputs(args.input)

    global silent
    silent = args.silent

    assert not (args.sink_type == "rtsp" and args.write_insights), "Cannot use RTSP when writing insights"

    output_path = args.output_path
    if args.sink_type in ["file", "hls"] and not output_path:
        executable_file_name_no_ext = get_file_name_no_ext(sys.argv[0])
        # Inputs can be files/uris/rtsp streams. So, generate output file name based on timestamp
        extension = None if args.sink_type == "hls" else "mp4"  # HLS sink requires a directory
        output_path = generate_filename(postfix=f"{executable_file_name_no_ext}_output", extension=extension)
        output_path = str(OUTPUT_DIR / output_path)
        print(f"Generated output file name: {output_path}\n")

    if args.write_insights:
        global insights_output_path
        insights_output_path = str(Path(output_path).with_suffix(''))
        os.makedirs(insights_output_path, exist_ok=True)

    src_config = SrcConfig(
        input_uris=args.input,
        file_loop=False,
        memtype=args.memtype,
        raw_detections=args.raw_detections,
        )

    sink_config = SinkConfig(
        type=args.sink_type,
        output_path=output_path,
        rtsp_port=args.rtsp_port,
        mount_point=args.mount_point,
        udp_port=args.udp_port,
        codec=args.codec,
        bitrate=args.bitrate,
        enc_type=args.enc_type
        )

    print(f"Arguments: {vars(args)}")
    return src_config, sink_config, args.write_insights


if __name__ == "__main__":
    src_config, sink_config, write_insights = parse_args()
    sys.exit(main(src_config, sink_config, write_insights))
