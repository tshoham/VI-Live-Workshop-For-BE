# region Imports
import argparse
import configparser
import platform
import os
import sys

import gi
gi.require_version("Gst", "1.0")
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
import pyds

from src.common.platform_info import PlatformInfo
from src.common.bus_call import bus_call
from src.common.utils import get_file_name_no_ext, generate_filename
from src.constants import DEEPSTREAM_CONFIGS_DIR, OUTPUT_DIR
# endregion Imports

# region Consts+Paths
PGIE_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest1_pgie_config.txt")
DSTEST2_SGIE1_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest2_sgie1_config.txt")
DSTEST2_SGIE2_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest2_sgie2_config.txt")
DSTEST2_TRACKER_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest2_tracker_config.txt")
assert os.path.exists(DSTEST2_SGIE1_CONFIG_PATH), f"File {DSTEST2_SGIE1_CONFIG_PATH} does not exist"
assert os.path.exists(DSTEST2_SGIE2_CONFIG_PATH), f"File {DSTEST2_SGIE2_CONFIG_PATH} does not exist"
assert os.path.exists(DSTEST2_TRACKER_CONFIG_PATH), f"File {DSTEST2_TRACKER_CONFIG_PATH} does not exist"
assert os.path.exists(PGIE_CONFIG_PATH), f"File {PGIE_CONFIG_PATH} does not exist"

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000
# endregion Consts+Paths

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    On screen display buffer probe.
    In place modifications of the frame meta data

    Args:
        pad (_type_): _description_
        info (_type_): _description_
        u_data (_type_): _description_

    Returns:
        _type_: _description_
    """
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

        # Intiallizing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0,
        }
        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(
                0.0, 0.0, 1.0, 0.8
            )  # 0.8 is alpha (opacity)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        text = f"Frame Number={frame_number} Number of Objects={num_rects} " \
            f"Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]} Person_count={obj_counter[PGIE_CLASS_ID_PERSON]}"
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

    #past tracking meta data
    l_user=batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            # Note that l_user.data needs a cast to pyds.NvDsUserMeta
            # The casting is done by pyds.NvDsUserMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone
            user_meta=pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
            try:
                # Note that user_meta.user_meta_data needs a cast to pyds.NvDsTargetMiscDataBatch
                # The casting is done by pyds.NvDsTargetMiscDataBatch.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                pPastDataBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
            except StopIteration:
                break
            for miscDataStream in pyds.NvDsTargetMiscDataBatch.list(pPastDataBatch):
                print("streamId=",miscDataStream.streamID)
                print("surfaceStreamID=",miscDataStream.surfaceStreamID)
                for miscDataObj in pyds.NvDsTargetMiscDataStream.list(miscDataStream):
                    print("numobj=",miscDataObj.numObj)
                    print("uniqueId=",miscDataObj.uniqueId)
                    print("classId=",miscDataObj.classId)
                    print("objLabel=",miscDataObj.objLabel)
                    for miscDataFrame in pyds.NvDsTargetMiscDataObject.list(miscDataObj):
                        print('frameNum:', miscDataFrame.frameNum)
                        print('tBbox.left:', miscDataFrame.tBbox.left)
                        print('tBbox.width:', miscDataFrame.tBbox.width)
                        print('tBbox.top:', miscDataFrame.tBbox.top)
                        print('tBbox.right:', miscDataFrame.tBbox.height)
                        print('confidence:', miscDataFrame.confidence)
                        print('age:', miscDataFrame.age)
        try:
            l_user=l_user.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def create_element(factory_name, element_name):
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        sys.stderr.write(f"Unable to create {element_name}\n")
    return element

def cofigure_tracker(tracker):
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read(DSTEST2_TRACKER_CONFIG_PATH)
    config.sections()
    
    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
            
def get_output_path(args):
    executable_file_name_no_ext = get_file_name_no_ext(args[0])
    input_file_name_no_ext = get_file_name_no_ext(args[1])
    output_path = generate_filename(postfix=f"{input_file_name_no_ext}_{executable_file_name_no_ext}_output", extension="mp4")
    output_path = str(OUTPUT_DIR / output_path)
    
    print(f"Creating FileSink for {output_path}\n")
    return output_path

def add_probe(nvosd):
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # region Gstream Elements
    
    # Create Pipeline element that will form a connection of other elements
    # Standard GStreamer initialization
    Gst.init(None)
    
    print("Creating Pipeline\n")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")

    ## sources
    # file_source element for reading from the file
    file_source = create_element("filesrc", "file-source")
    file_source.set_property("location", args[1])

    ## Parser
    # Since the data format in the input file is elementary h264 stream, we need a h264parser
    h264parser = create_element("h264parse", "h264-parser")

    ## Decoder
    # Use nvdec_h264 for hardware accelerated decode on GPU
    decoder = create_element("nvv4l2decoder", "nvv4l2-decoder")

    ## Streammux
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = create_element("nvstreammux", "Stream-muxer")
    print("Playing file %s " % args[1])
    if os.environ.get("USE_NEW_NVSTREAMMUX") != "yes":  # Only set these properties if not using new gst-nvstreammux
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("batch-size", 1)

    ## detectors and trackers
    # Use nvinfer to run inferencing on decoder's output, behaviour of inferencing is set through config file
    pgie = create_element("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", PGIE_CONFIG_PATH)
    
    tracker = create_element("nvtracker", "tracker")
    cofigure_tracker(tracker)
    
    sgie1 = create_element("nvinfer", "secondary1-nvinference-engine")
    sgie1.set_property('config-file-path', DSTEST2_SGIE1_CONFIG_PATH)

    sgie2 = create_element("nvinfer", "secondary2-nvinference-engine")
    sgie2.set_property('config-file-path', DSTEST2_SGIE2_CONFIG_PATH)
     
    ## Convertor + OSD
    # Convert frames (format, scaling, color spaces)
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = create_element("nvvideoconvert", "convertor")

    # Create OSD (on screen display") to draw on the converted RGBA buffer
    nvosd = create_element("nvdsosd", "onscreendisplay")

    ## sink
    # file sink
    output_path = get_output_path(args)
    file_sink = Gst.ElementFactory.make("nvvideoencfilesinkbin", "file-sink")
    file_sink.set_property("output-file", output_path)
    
    # rtsp sink
    nvvidconv_postosd = create_element("nvvideoconvert", "convertor_postosd")
    
    # Create a caps filter - hardware encoder
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    
    # Make the encoder - rtsp streaming codec H264
    encoder = create_element("nvv4l2h264enc", "encoder")
    encoder.set_property('bitrate', 4000000)
    platform_info = PlatformInfo()
    if platform_info.is_integrated_gpu():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        #encoder.set_property('bufapi-version', 1)

    # Make the payload-encode video into RTP packets for H264
    rtppay = create_element("rtph264pay", "rtppay")
    
    # Make the UDP sink
    updsink_port_num = 5400
    rtsp_sink = create_element("udpsink", "udpsink")
    rtsp_sink.set_property('host', '224.224.255.255')
    rtsp_sink.set_property('port', updsink_port_num)
    rtsp_sink.set_property('async', False)
    rtsp_sink.set_property('sync', 1)
    
    # endregion Gstream Elements

    # region Add Elements To Pipeline

    # Simple Pipeline with 4-class-detecor and file sink
    # # print("Adding elements to Pipeline dstest1 \n")
    # # pipeline.add(file_source)
    # # pipeline.add(h264parser)
    # # pipeline.add(decoder)
    # # pipeline.add(streammux)
    # # pipeline.add(pgie)
    # # pipeline.add(nvvidconv)
    # # pipeline.add(nvosd)
    # # pipeline.add(file_sink)
    
    # Pipeline with 4-class-detecor, tracker, vehicle make and type classifiers and file sink
    print("Adding elements to Pipeline dstest2 \n")
    pipeline.add(file_source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(sgie2)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(file_sink)
    
    # Simple Pipeline with 4-class-detecor and rtsp sink
    # # print("Adding elements to Pipeline dstest1 with rtsp out \n")
    # # pipeline.add(file_source)
    # # pipeline.add(h264parser)
    # # pipeline.add(decoder)
    # # pipeline.add(streammux)
    # # pipeline.add(pgie)
    # # pipeline.add(nvvidconv)
    # # pipeline.add(nvosd)
    # # pipeline.add(nvvidconv_postosd)
    # # pipeline.add(caps)
    # # pipeline.add(encoder)
    # # pipeline.add(rtppay)
    # # pipeline.add(rtsp_sink)
    
    # Pipeline with 4-class-detecor, tracker, vehicle make and type classifiers and rtsp sink
    # # print("Adding elements to Pipeline dstest2 \n")
    # # pipeline.add(file_source)
    # # pipeline.add(h264parser)
    # # pipeline.add(decoder)
    # # pipeline.add(streammux)
    # # pipeline.add(pgie)
    # # pipeline.add(tracker)
    # # pipeline.add(sgie1)
    # # pipeline.add(sgie2)
    # # pipeline.add(nvvidconv)
    # # pipeline.add(nvosd)
    # # pipeline.add(nvvidconv_postosd)
    # # pipeline.add(caps)
    # # pipeline.add(encoder)
    # # pipeline.add(rtppay)
    # # pipeline.add(rtsp_sink)

    # endregion Add Elements To Pipeline
    
    # region Link Elements In Pipeline
    
    # Link the elements together
    # test 1: file-source -> h264-parser -> nvh264-decoder -> pgie -> nvvidconv -> nvosd -> video-renderer
    
    # test 2: same as 1 with rtsp out
    # file-source -> h264-parser -> nvh264-decoder -> pgie -> nvvidconv -> nvosd -> nvvidconv_postosd -> caps -> encoder -> rtppay -> udpsink

    # test 3: file-source -> h264-parser -> nvh264-decoder -> pgie -> tracker -> sgie1 -> sgie2 -> nvvidconv -> nvosd -> video-renderer
    
    # test 4: same as 3 with rtsp out
    # file-source -> h264-parser -> nvh264-decoder -> pgie -> tracker -> sgie1 -> sgie2 -> nvvidconv -> nvosd -> nvvidconv_postosd -> caps -> encoder -> rtppay -> udpsink

    print("Linking elements in the Pipeline\n")
    
    # linking source to h264parser to decoder (No need to edit)
    file_source.link(h264parser)
    h264parser.link(decoder)

    # linking decoder to streammux (No need to edit)
    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to get the sink pad of streammux\n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to get source pad of decoder\n")
    srcpad.link(sinkpad)
    
    # linking streamux to pgie/tracker/sgie to video converter 
    # (tests 1 + 2)
    # # streammux.link(pgie)
    # # pgie.link(nvvidconv)
    
    # (tests 3 + 4)
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(sgie2)
    sgie2.link(nvvidconv)
    
    
    # link converter to on screen display ( no need to edit)
    nvvidconv.link(nvosd)
    
    # # link sink - to file or to rtsp
    
    # link osd to sink (tests 1 + 3)
    nvosd.link(file_sink)
    
    # link osd with rtsp out (tests 2 + 4)
    # # nvosd.link(nvvidconv_postosd)
    # # nvvidconv_postosd.link(caps)
    # # caps.link(encoder)
    # # encoder.link(rtppay)
    # # rtppay.link(rtsp_sink)

    # endregion Link Elements In Pipeline

    # Create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # region Rtsp Out
    # Start streaming (If running rtsp out) - if running rtsp tests (2 or 4) then uncomment this region
    rtsp_port_num = 8554
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, "H264"))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)
    
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)
    # endregion Rtsp Out

    add_probe(nvosd)

    # start play back and listen to events
    print("Starting pipeline\n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    # handle a generic argparse
    sys.exit(main(sys.argv))
