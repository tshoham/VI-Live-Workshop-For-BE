#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import os
import sys
import json
import configparser

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import pyds

from src.constants import DEEPSTREAM_CONFIGS_DIR
from src.common.platform_info import PlatformInfo
from src.common.bus_call import bus_call
from src.common.utils import get_file_name_no_ext

SCREEN_AVAILABLE = False

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000

DSTEST2_PGIE_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest2_pgie_config.txt")
DSTEST2_SGIE1_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest2_sgie1_config.txt")
DSTEST2_SGIE2_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest2_sgie2_config.txt")
DSTEST2_TRACKER_CONFIG_PATH = str(DEEPSTREAM_CONFIGS_DIR / "dstest2_tracker_config.txt")
assert os.path.exists(DSTEST2_PGIE_CONFIG_PATH), f"File {DSTEST2_PGIE_CONFIG_PATH} does not exist"
assert os.path.exists(DSTEST2_SGIE1_CONFIG_PATH), f"File {DSTEST2_SGIE1_CONFIG_PATH} does not exist"
assert os.path.exists(DSTEST2_SGIE2_CONFIG_PATH), f"File {DSTEST2_SGIE2_CONFIG_PATH} does not exist"
assert os.path.exists(DSTEST2_TRACKER_CONFIG_PATH), f"File {DSTEST2_TRACKER_CONFIG_PATH} does not exist"
insights_buffer = {}
frame_rate = 30
sequence_index = 0
hls_target_duration = 5 # 5 seconds
sequence_start_time = 0

def on_eos(bus, msg):
    global insights_buffer
    global sequence_index

    # Write the remaining buffer to a .json file
    if insights_buffer:
        with open(f"hlssink/{sequence_index}.json", "w") as f:
            json.dump(insights_buffer, f)
        print(f"Writing final sequence {sequence_index} to file")

def osd_sink_pad_buffer_probe(pad, info, u_data):
    global insights_buffer
    global sequence_index
    global hls_target_duration
    global sequence_start_time

    frame_number = 0
    # Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE: 0,
        PGIE_CLASS_ID_PERSON: 0,
        PGIE_CLASS_ID_BICYCLE: 0,
        PGIE_CLASS_ID_ROADSIGN: 0,
    }
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
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(
            frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

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
        
    # Past tracking meta data
    l_user = batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            # Note that l_user.data needs a cast to pyds.NvDsUserMeta
            # The casting is done by pyds.NvDsUserMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if (user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
            try:
                # Note that user_meta.user_meta_data needs a cast to pyds.NvDsTargetMiscDataBatch
                # The casting is done by pyds.NvDsTargetMiscDataBatch.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave it alone
                pPastDataBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
            except StopIteration:
                break
            def nvds_to_dict(nvds_obj):
                return {attr: getattr(nvds_obj, attr) for attr in dir(nvds_obj) if not attr.startswith('__') and not callable(getattr(nvds_obj, attr))}

            frameStartTime = 0
            for miscDataStream in pyds.NvDsTargetMiscDataBatch.list(pPastDataBatch):
                print("streamId=", miscDataStream.streamID)
                print("surfaceStreamID=", miscDataStream.surfaceStreamID)
                for miscDataObj in pyds.NvDsTargetMiscDataStream.list(miscDataStream):
                    print("numobj=", miscDataObj.numObj)
                    print("uniqueId=", miscDataObj.uniqueId)
                    print("classId=", miscDataObj.classId)
                    print("objLabel=", miscDataObj.objLabel)
                    for miscDataFrame in pyds.NvDsTargetMiscDataObject.list(miscDataObj):                       
                        frameNum = miscDataFrame.frameNum
                        print("frameNum:", frameNum)
                        print("tBbox.left:", miscDataFrame.tBbox.left)
                        print("tBbox.width:", miscDataFrame.tBbox.width)
                        print("tBbox.top:", miscDataFrame.tBbox.top)
                        print("tBbox.right:", miscDataFrame.tBbox.height)
                        print("confidence:", miscDataFrame.confidence)
                        print("age:", miscDataFrame.age)
                        
                        frameStartTime = frameNum / frame_rate
                        bbox_data = {
                            "id": miscDataObj.uniqueId,
                            "name": miscDataObj.objLabel,
                            "bbox": [
                                {
                                    "frameNum": frameNum,
                                    "x": miscDataFrame.tBbox.left,
                                    "y": miscDataFrame.tBbox.top,
                                    "width": miscDataFrame.tBbox.width,
                                    "height": miscDataFrame.tBbox.height,
                                    "start": round(frameStartTime, 2),
                                    "confidence": miscDataFrame.confidence
                                }
                            ]
                        }

                        objLabel = miscDataObj.objLabel
                        uniqueId = miscDataObj.uniqueId

                        if objLabel not in insights_buffer:
                            insights_buffer[objLabel] = {}
                        if uniqueId not in insights_buffer[objLabel]:
                            insights_buffer[objLabel][uniqueId] = []

                        # group insights by object label and unique id
                        insights_buffer[objLabel][uniqueId].append(bbox_data)

            if frameStartTime - sequence_start_time >= hls_target_duration:
                # Write the buffer to a .json file
                with open(f"hlssink/{sequence_index}.json", "w") as f:
                    json.dump(insights_buffer, f)
                print(f"Writing sequence {sequence_index} to file, frameStartTime: {frameStartTime}")
                
                sequence_index += 1
                # Clear the buffer and update the start time
                insights_buffer = {}
                # Update sequence_start_time with the start time of the next frame
                sequence_start_time = frameStartTime
                        
        try:
            l_user = l_user.next
        except StopIteration:
            break
        
    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <h264_elementary_stream>\n" % args[0])
        sys.exit(1)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization

    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline\n")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")

    # Source element for reading from the file
    print("Creating Source\n")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write("Unable to create Source\n")

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser\n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write("Unable to create h264 parser\n")

    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder\n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write("Unable to create Nvv4l2 Decoder\n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write("Unable to create tracker\n")

    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write("Unable to make sgie1\n")

    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie2:
        sys.stderr.write("Unable to make sgie2\n")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")

    if SCREEN_AVAILABLE:
        # Finally render the osd output
        if platform_info.is_integrated_gpu():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write("Unable to create nv3dsink\n")
        else:
            if platform_info.is_platform_aarch64():
                print("Creating nv3dsink \n")
                sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            else:
                print("Creating EGLSink\n")
                sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write("Unable to create egl sink\n")
    else:
        # executable_file_name_no_ext = get_file_name_no_ext(args[0])
        # input_file_name_no_ext = get_file_name_no_ext(args[1])
        # output_path = f"{input_file_name_no_ext}_{executable_file_name_no_ext}_output.mp4"

        # print(f"Creating FileSink for {output_path}\n")
        # sink = Gst.ElementFactory.make("nvvideoencfilesinkbin", "file-sink")
        # sink.set_property("output-file", output_path)

        # Create HLS sink to output HLS content
        sink = Gst.ElementFactory.make("hlssink2", "hlssink2")
        if not sink:
            sys.stderr.write("Unable to create hlssink2 sink \n")

        output_directory = "hlssink"
        os.makedirs(output_directory, exist_ok=True)
        sink.set_property('send_keyframe_requests', False)
        sink.set_property('target_duration', hls_target_duration)
        sink.set_property('playlist-length', 0)
        sink.set_property('max-files', 1000)
        sink.set_property('location', f"{output_directory}/%05d.ts")
        sink.set_property('playlist-location', f"{output_directory}/playlist.m3u8")

    hw_encoder = Gst.ElementFactory.make("nvv4l2h264enc", "hw_encoder")
    if not hw_encoder:
        sys.stderr.write(" Unable to create hw_encoder \n")

    hw_encoder.set_property('iframeinterval', 2)
    hw_encoder.set_property('bitrate', 2500*1000)

        
    print("Playing file %s " % args[1])
    source.set_property("location", args[1])
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    # Set properties of pgie and sgie
    pgie.set_property("config-file-path", DSTEST2_PGIE_CONFIG_PATH)
    sgie1.set_property("config-file-path", DSTEST2_SGIE1_CONFIG_PATH)
    sgie2.set_property("config-file-path", DSTEST2_SGIE2_CONFIG_PATH)

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read(DSTEST2_TRACKER_CONFIG_PATH)
    config.sections()

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
            tracker.set_property("ll-config-file", tracker_ll_config_file)

    print("Adding elements to Pipeline\n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(sgie2)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(hw_encoder)
    pipeline.add(sink)

    # We link the elements together
    # file-source -> h264-parser -> nvh264-decoder -> nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline\n")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to get the sink pad of streammux\n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(sgie2)
    sgie2.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(hw_encoder)
    hw_encoder.link(sink)

    # Create and event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    bus.connect("message::eos", on_eos)


    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("Starting pipeline\n")

    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # Cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
