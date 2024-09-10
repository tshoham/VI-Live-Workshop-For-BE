import configparser
from dataclasses import dataclass
import os

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.common.platform_info import PlatformInfo


@dataclass
class SrcConfig:
    VALID_CUDA_MEMORY_TYPES = [-1, 0, 1, 2]

    input_uris: list
    file_loop: bool
    memtype: int  # 0: NVBUF_MEM_CUDA_DEVICE, 1: NVBUF_MEM_CUDA_PINNED, 2: NVBUF_MEM_CUDA_UNIFIED
    raw_detections: bool  # Debug raw detections (write to file)

    def __post_init__(self):
        if not isinstance(self.input_uris, list):
            raise ValueError("input_uris must be a list")
        if not all(isinstance(uri, str) for uri in self.input_uris):
            raise ValueError("input_uris must contain only strings")
        if not isinstance(self.file_loop, bool):
            raise ValueError("file_loop must be a boolean value")
        if self.memtype not in SrcConfig.VALID_CUDA_MEMORY_TYPES:
            raise ValueError(f"Invalid memtype. Must be one of {SrcConfig.VALID_CUDA_MEMORY_TYPES}")


@dataclass
class SinkConfig:
    VALID_SINK_TYPES = ["screen", "fake", "file", "rtsp", "hls"]
    VALID_ENC_TYPES = ['hw', 'sw']
    VALID_CODECS = ['H264', 'H265']

    type: str = "screen"
    output_path: str = "output.mp4"

    rtsp_port: int = 8554
    mount_point: str = "/test"
    udp_port: int = 5400
    codec: str = "H264"
    bitrate: int = 4000000
    enc_type: str = "hw"

    def __post_init__(self):
        if self.type not in SinkConfig.VALID_SINK_TYPES:
            raise ValueError(f"Invalid sink type. Must be one of {SinkConfig.VALID_SINK_TYPES}")
        if self.enc_type not in SinkConfig.VALID_ENC_TYPES:
            raise ValueError(f"Invalid encoder type. Must be one of {SinkConfig.VALID_ENC_TYPES}")
        if self.codec not in SinkConfig.VALID_CODECS:
            raise ValueError(f"Invalid codec. Must be one of {SinkConfig.VALID_CODECS}")


def verify_component(component, name, err_msg=None):
    if not component:
        err_msg = err_msg or f"Failed to create {name}"
        raise RuntimeError(err_msg)


def create_element(element, name, container=None):
    print(f"Creating {name}")
    elem = Gst.ElementFactory.make(element, name)
    verify_component(elem, name)

    if container:
        container.add(elem)

    return elem


def create_encoder_element(sink_config: SinkConfig, container=None, return_caps=False):
    if sink_config.enc_type == 'hw':
        caps_property_str = "video/x-raw(memory:NVMM), format=I420"
        if sink_config.codec == "H264":
            encoder_plugin_name = "nvv4l2h264enc"
        elif sink_config.codec == "H265":
            encoder_plugin_name = "nvv4l2h265enc"
    else:  # sink_config.enc_type == 'sw':
        caps_property_str = "video/x-raw, format=I420"
        if sink_config.codec == "H264":
            encoder_plugin_name = "x264enc"
        elif sink_config.codec == "H265":
            encoder_plugin_name = "x265enc"

    encoder = create_element(encoder_plugin_name, f"encoder_{encoder_plugin_name}", container)
    encoder.set_property('bitrate', sink_config.bitrate)
    # encoder.set_property('iframeinterval', 2)

    return (encoder, caps_property_str) if return_caps else encoder


def create_rtsp_sink_bin(sink_config: SinkConfig):
    bin_name = "sink-rtsp-bin"
    sink_bin = Gst.Bin.new(bin_name)
    verify_component(sink_bin, "sink rtsp bin")

    # Convert RGBA back to RGB
    nvvidconv_postosd = create_element("nvvideoconvert", "convertor_postosd", sink_bin)

    encoder, caps_property_str = create_encoder_element(sink_config, sink_bin, return_caps=True)

    # Create a caps filter
    caps = create_element("capsfilter", "filter", sink_bin)
    caps.set_property("caps", Gst.Caps.from_string(caps_property_str))

    platform_info = PlatformInfo()
    if platform_info.is_integrated_gpu() and sink_config.enc_type == 'hw':
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        # encoder.set_property('bufapi-version', 1)

    # Make the payload-encode video into RTP packets
    if sink_config.codec == "H264":
        rtppay = create_element("rtph264pay", "rtppay", sink_bin)
    elif sink_config.codec == "H265":
        rtppay = create_element("rtph265pay", "rtppay", sink_bin)

    # Make the UDP sink
    udp_sink = create_element("udpsink", "udpsink", sink_bin)
    udp_sink.set_property('host', '224.224.255.255')
    udp_sink.set_property('port', sink_config.udp_port)
    udp_sink.set_property('async', False)
    udp_sink.set_property('sync', 1)

    udp_sink.set_property("qos", 0)  # In production this should probably be removed

    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(udp_sink)

    # Add ghost pads
    sink_bin.add_pad(Gst.GhostPad.new("sink", nvvidconv_postosd.get_static_pad("sink")))

    return sink_bin


def create_hls_sink_bin(sink_config: SinkConfig):
    print(f"Creating HLS sink bin in {sink_config.output_path}")

    sink_bin = Gst.Bin.new("sink-hls-bin")
    verify_component(sink_bin, "sink hls bin")

    # Convert RGBA back to RGB
    nvvidconv_postosd = create_element("nvvideoconvert", "convertor_postosd", sink_bin)

    encoder = create_encoder_element(sink_config, container=sink_bin)

    sink = create_element("hlssink2", "hlssink2", sink_bin)
    output_directory = sink_config.output_path
    os.makedirs(output_directory, exist_ok=True)

    sink.set_property('send_keyframe_requests', False)
    sink.set_property('target_duration', 5)  # TODO: Make this configurable
    sink.set_property('playlist-length', 0)
    sink.set_property('max-files', 1000)  # TODO: Make this configurable
    sink.set_property('location', f"{output_directory}/%05d.ts")
    sink.set_property('playlist-location', f"{output_directory}/playlist.m3u8")

    nvvidconv_postosd.link(encoder)
    encoder.link(sink)

    # Add ghost pads
    sink_bin.add_pad(Gst.GhostPad.new("sink", nvvidconv_postosd.get_static_pad("sink")))

    return sink_bin


def create_sink(sink_config: SinkConfig, container=None):
    if sink_config.type == "file":
        sink = create_element("nvvideoencfilesinkbin", "file-sink")
        print(f"FileSink {sink_config.output_path=}\n")
        sink.set_property("output-file", sink_config.output_path)
    elif sink_config.type == "fake":
        sink = create_element("nvvideoencfilesinkbin", "file-sink")
        sink.set_property("enable-last-sample", 0)
        sink.set_property("sync", 0)
    elif sink_config.type == "screen":
        platform_info = PlatformInfo()

        if platform_info.is_integrated_gpu():
            sink = create_element("nv3dsink", "nv3d-sink")
        else:
            if platform_info.is_platform_aarch64():
                sink = create_element("nv3dsink", "nv3d-sink")
            else:
                sink = create_element("nveglglessink", "egl-nvvideo-renderer")

        sink.set_property("qos", 0)  # In production this should probably be removed
    elif sink_config.type == "rtsp":
        sink = create_rtsp_sink_bin(sink_config)
    elif sink_config.type == "hls":
        sink = create_hls_sink_bin(sink_config)

    if container:
        container.add(sink)

    return sink


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not audio
    print(f"{gstname=}")
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia decoder plugin nvdec_*.
        # We do this by checking if the pad caps contain NVMM memory features.
        print(f"{features=}")
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                raise RuntimeError("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            raise RuntimeError("Error: Decodebin did not pick nvidia decoder plugin\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    src_config: SrcConfig = user_data["src_config"]

    print(f"Decodebin child added: {name}\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if src_config.memtype != -1 and not PlatformInfo().is_integrated_gpu() and name.find("nvv4l2decoder") != -1:
        # Use CUDA unified memory in the pipeline so frames can be easily accessed on CPU in Python
        print(f"Setting CUDA memtype to {src_config.memtype}. (0:DEVICE, 1: PINNED, 2: UNIFIED)")
        Object.set_property("cudadec-memtype", src_config.memtype)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(src_config: SrcConfig, index):
    uri = src_config.input_uris[index]
    file_loop = src_config.file_loop

    print(f"Creating source bin {index=} for {uri=}")

    # Create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    verify_component(nbin, "source bin")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    if file_loop:
        # use nvurisrcbin to enable file-loop
        uri_decode_bin = create_element("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin = create_element("uridecodebin", "uri-decode-bin")

    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)

    user_data = {'src_bin': nbin, "src_config": src_config}
    uri_decode_bin.connect("child-added", decodebin_child_added, user_data)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    verify_component(bin_pad, "GhostPad", err_msg="Failed to add ghost pad in source bin")

    return nbin


def get_tracker(ds_configs_dir, tracker_config_file, container=None):
    tracker_config_path = str(ds_configs_dir / tracker_config_file)
    assert os.path.exists(tracker_config_path), f"File {tracker_config_path} does not exist"

    tracker = create_element("nvtracker", "tracker", container)

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read(tracker_config_path)
    # config.sections()

    for key in config["tracker"]:
        if key == "tracker-width":
            tracker_width = config.getint("tracker", key)
            tracker.set_property("tracker-width", tracker_width)
        if key == "tracker-height":
            tracker_height = config.getint("tracker", key)
            tracker.set_property("tracker-height", tracker_height)
        if key == "gpu-id":
            tracker_gpu_id = config.getint("tracker", key)
            tracker.set_property("gpu_id", tracker_gpu_id)
        if key == "ll-lib-file":
            tracker_ll_lib_file = config.get("tracker", key)
            tracker.set_property("ll-lib-file", tracker_ll_lib_file)
        if key == "ll-config-file":
            tracker_ll_config_file = config.get("tracker", key)

            tracker_ll_config_file = str(ds_configs_dir / tracker_ll_config_file)
            if not os.path.exists(tracker_ll_config_file):
                print(f"WARNING: File {tracker_ll_config_file} does not exist. Default values will be used\n")

            tracker.set_property("ll-config-file", tracker_ll_config_file)

    return tracker
