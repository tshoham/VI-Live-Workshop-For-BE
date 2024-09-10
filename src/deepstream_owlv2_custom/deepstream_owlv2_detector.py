#!/usr/bin/env python3

import argparse
import os
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import pyds
from transformers import AutoProcessor, Owlv2Model

from src.common.platform_info import PlatformInfo
from src.common.bus_call import bus_call
from src.common.gst_utils import create_element, create_sink, create_source_bin, verify_component, SinkConfig, \
    SrcConfig
from src.common.utils import check_and_normalize_inputs, get_file_name_no_ext, generate_filename, \
    replace_model_engine_file_in_nvinfer_plugin
from src.constants import DEEPSTREAM_CONFIGS_DIR, MODELS_DIR, OUTPUT_DIR
from owlv2_utils import add_obj_meta_to_frame, expand_prompt, get_layers_info, nvds_infer_parse_custom_owl_output, \
    OwlInfoState, OwlV2TextEncoding


IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
MUXER_BATCH_TIMEOUT_USEC = 33000

PGIE_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "owlv2_pgie_config.txt")
print(f"{PGIE_CONFIG_PATH=}")
assert os.path.exists(PGIE_CONFIG_PATH), f"File {PGIE_CONFIG_PATH} does not exist"

# TODO: image_sizes - need to add to config
# "google/owlv2-base-patch16-ensemble": 960,
# "google/owlv2-large-patch14-ensemble": 1008,


def probe_callback(pad, info, u_data):
    # This function will be called for every frame.
    # You can add your processing code here.
    # The Gst.Buffer can be accessed through info.get_buffer().

    # For example, to print the PTS of each frame:
    buffer = info.get_buffer()
    if not buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK
    print("PTS: ", buffer.pts)

    return Gst.PadProbeReturn.OK  # Continue with the default pad handling.


def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if (user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
                continue

            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

            owl_layers_info = get_layers_info(tensor_meta)

            owl_info = OwlInfoState.get_embedding()
            frame_object_list = nvds_infer_parse_custom_owl_output(owl_layers_info, owl_info)
            try:
                l_user = l_user.next
            except StopIteration:
                break

            # Add the object metadata to the frame
            for frame_object in frame_object_list:
                add_obj_meta_to_frame(frame_object, batch_meta, frame_meta, owl_info.text)

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta

        # Acquiring a display meta object. The memory ownership remains in the C code so downstream plugins can still
        # access it. Otherwise the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the memory will not be claimed by the garbage
        # collector. Reading the display_text field here will return the C address of the allocated string.
        # Use pyds.get_string() to get the string content.
        text = f"Frame Number={frame_number} Number of Objects={num_rects}"
        py_nvosd_text_params.display_text = text

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(src_config, sink_config):
    platform_info = PlatformInfo()

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline\n")
    pipeline = Gst.Pipeline()
    verify_component(pipeline, "Pipeline")

    # Create nvstreammux instance to form batches from one or more sources
    streammux = create_element("nvstreammux", "Stream-muxer", pipeline)

    is_live = False
    number_sources = len(src_config.input_uris)
    for i in range(number_sources):
        print(f"Creating source_bin {i}\n")
        uri_name = src_config.input_uris[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True

        source_bin = create_source_bin(src_config, i)
        verify_component(source_bin, "Source bin")

        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        verify_component(sinkpad, "Streammux sink pad")

        srcpad = source_bin.get_static_pad("src")
        verify_component(srcpad, "Source bin src pad")

        srcpad.link(sinkpad)

    pgie = create_element("nvinfer", "primary-inference", pipeline)
    nvvidconv = create_element("nvvideoconvert", "convertor", pipeline)
    nvosd = create_element("nvdsosd", "onscreendisplay", pipeline)
    sink = create_sink(sink_config, pipeline)

    if os.environ.get("USE_NEW_NVSTREAMMUX") != "yes":  # Only set these properties if not using new gst-nvstreammux
        streammux.set_property("width", IMAGE_WIDTH)
        streammux.set_property("height", IMAGE_HEIGHT )
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    if is_live:
        print("At least one of the sources is live")
        streammux.set_property("live-source", 1)

    streammux.set_property("batch-size", 1)
    pgie.set_property("config-file-path", PGIE_CONFIG_PATH)
    replace_model_engine_file_in_nvinfer_plugin(pgie, platform_info)

    nvosd.set_property("display-clock", 1)

    # We link the elements together
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # Create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to the sink pad of the osd element,
    # since by that time, the buffer would have had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    verify_component(osdsinkpad, "OSD sink pad")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    # Start play back and listen to events
    print("Starting pipeline\n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # Cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='Run using Owlv2')
    parser.add_argument("-i", "--input", help="Path to input streams", nargs="+", metavar="URIs", required=True)
    parser.add_argument('--prompt', default="[a person,a car,a motorcycle,a traffic light]")
    parser.add_argument('--threshold', default="0.1,0.1,0.1,0.1")
    parser.add_argument('--color-info')

    # Section for sink type
    parser.add_argument("--sink-type", default="file", choices=SinkConfig.VALID_SINK_TYPES, help="Choose the sink type")
    parser.add_argument("-o", "--output-path", help="Path to save the output file")

    # parser.add_argument('--model', default="google/owlv2-base-patch16-ensemble") currently only 1 model supported
    # parser.add_argument('--image_encoder_engine', default="../data/owlv2_image_encoder_patch16.engine") currently only 1 model supported
    args = parser.parse_args()

    args.input = check_and_normalize_inputs(args.input)

    output_path = args.output_path
    if args.sink_type in ["file", "hls"] and not output_path:
        executable_file_name_no_ext = get_file_name_no_ext(sys.argv[0])
        # Inputs can be files/uris/rtsp streams. So, generate output file name based on timestamp
        extension = None if args.sink_type == "hls" else "mp4"  # HLS sink requires a directory
        output_path = generate_filename(postfix=f"{executable_file_name_no_ext}_output", extension=extension)
        output_path = str(OUTPUT_DIR / output_path)
        print(f"Generated output file name: {output_path}\n")

    src_config = SrcConfig(
        input_uris=args.input,
        file_loop=False,
        memtype=-1,
        raw_detections=False
        )

    sink_config = SinkConfig(
        type=args.sink_type,
        output_path=output_path
        )

    args.prompt = args.prompt.strip("][()").split(',')
    if args.color_info:
        args.color_info = args.color_info.strip("][()").split(',')

    text = args.prompt
    if args.color_info:
        color_promt = expand_prompt(text, args.color_info)
        print(f"{color_promt=}")
        text.extend(color_promt)

    thresholds = args.threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]

    # Init text encoding model name
    OwlV2TextEncoding(model_name="google/owlv2-base-patch16-ensemble")
    owl_info_state = OwlInfoState()
    owl_info_state.add_embedding(text, thresholds)

    # Need to add scale and offset to config https://forums.developer.nvidia.com/t/pytorch-normalization-in-deepstream-config/158154
    return src_config, sink_config


if __name__ == "__main__":
    src_config, sink_config = parse_args()

    sys.exit(main(src_config, sink_config))
