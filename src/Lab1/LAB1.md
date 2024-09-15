# Lab 1

## Background:
VI Live will have configurable presets where the customers can choose which AI's will run on which cameras.
This means that our video pipelines need to be easily adjustable.
We mentioned that GStreamer's benefits consist of "Modular Architecture", and we will see here how this allows to create custom pipelines using various plugins (both gstreamer and deepstream plugins).

## Purpose
lab1.py is a sample that consists of 4 different pipeline options.
We will go over all of the code very briefly and focus on the pipeline creation and element linking. 

## Instructions

### Pipeline 1: Simple Detector Pipeline

1. At the beginning of the main method a pipeline was created ```pipeline = Gst.Pipeline()```.
    
    Add the elements **a** through **h** below to the pipeline. Use ```pipeline.add(name_of_element)``` (all the elements are in **bold** numbered a, b, c...).
    
    Add your code where it says ```""" Add the elements to the pipeline here """```
    - This pipeline will get input from a h264 file source. The file will be parsed and decoded.
        1. **file_source**
        2. **h264parser**
        3. **decoder**
     
        > **NOTE:**
        > H264 is a video file encoded with H.264 compression
        > H264 is the input of lab1 pipeline.
        > The file will go through a parser that will parse the stream and a decoder that will decode the stream into raw video frames for further processing

    - The pipeline will then continue to the streamux.
      
        4. **streammux**
      
        > **NOTE:**
        > streammux is a plugin used for multiplexing multiple input streams into a single output.
        > Multiplexing is a technique used in telecommunications and computer networking to combine multiple signals into one signal over a shared medium.

    - The next element is pgie. This is Nvidia's 4 class detector. It detects "Vehicle , RoadSign, TwoWheeler, Person".
    The inference element attaches some metadata to the buffer. We can later extract meaningful information from this buffer using a probe.
        
        5. **pgie**
        > **NOTE:**
        > PGIE stands for "Primary GPU Inference Engine. SGIE is "Secondary GPU Inference Engine" 

    - The next to elements are for on screen display. They convert frames to rgba and create the on-screen-display
        
        6. **nvvidconv**
        7. **nvosd**

        > **NOTE:**
        > On screen display is just for the demos and will most likely not run in production as FE will handle the displaying of the videos.

    - Lastly, the sink is linked in the pipeline.
    
        8. **file_sink**
        

2. Link all the elements in the order you created them:
    - link using the syntax:
    ```element_name.link(next_element)```
    - 2 elements will be linked using "pads" (instead of the syntax above). 
        The Decoder will be linked to streammux using pads as follows:
        ```
        srcpad = decoder.get_static_pad("src")
        sinkpad = streammux.request_pad_simple("sink_0")
        srcpad.link(sinkpad)
        ```

    - Add your code where it says ```""" Link the elements together here """```.

        > **NOTE:**
        > Pads are interfaces through which data flows in and out of elements.
        > An element can have multiple pads, so in this case we have a new pad for each source.
        > Pads allow  modularity, flexibility, data flow control, dynamic linking, and compatibility.


3. Open the folder in dev container and run Lab 1.
4. Look at the results and the detections.
 
> [!IMPORTANT]
> Copy the output file to you local machine to view the video.
>
> `scp azureuser@40.124.109.198:~/VideoIndexer-Nvidia-Live-PoC/outputs/"file_name" "local_path"`

> [!NOTE]
> In the video output, notice at the top left corner there, is printed metadata. We added this metadata to the "nvosd" using the "add_probe" function using ```osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)```. 
> We can preform in place modifications on the metadata.

### Pipeline 2: Detector + Tracker + Classifiers Pipeline
This pipeline will have more than the simple detector. You will add a tracker and 2 more secondary inference engines - car make classifier and car model classifier.

1. Up until pgie (e from the previous step) all elements will be the same, including the pgie.

    Add elements **a** through **e** to the pipeline.

1. After the pgie, **add** a tracker and 2 more AI inferences to the pipeline:

    1. **tracker**
    2. **sgie1** - this is a car make classifier
    3. **sgie2** - this is a car type classifier

1. After sgie2, the pipeline will continue to the converter, osd, and sink as in the previous example.
    Add **f** through **h** elements to the pipeline as well.

2. Link all the elements. Add the "tracker", "sgie1", "sgie2" after the "Pgie" and before "nvvidconv". Take in to account the linking of streammux and the decoder.
3. Open the folder in dev container and run Lab 1.
4. Look at the results. Notice the tracker and car make + type that were added.

### Pipeline 3: Simple Detector Pipeline With RTSP Outpus

> [!IMPORTANT]
> To view the rtsp output open your vlc player in you local machine
> Click on ```Media``` then ```Open Network Stream```. Past the following URL: ```rtsp://40.124.109.198:554/ds-test```
> **Don't** click Play yet!
>
> Have this ready before running the Lab in the VM. Once you run the Lab, wait until you see that the frames are being processd, and the click Play. 
> This may fail once before connecting successfully.

1. In this example we will run the same AI inference as the first example. Here we will output to rtsp instead of a file (no need for file_sink). After the **nvosd** we will add the following elements to the pipeline (elemnets **a** through **g** from the first example should be added to the pipeline):

    1. **nvvidconv_postosd**
    2. **caps**
    3. **encoder**
    4. **rtppay**
    5. **rtsp_sink**

2. Link these elements to the pipeline after nvosd (after element **g** from the first example).
3. Uncomment the ```region Rtsp Out``. This code sets up and starts the RTSP. 
4. Open the folder in dev container and run Lab 1.
5. View the RTSP outputput

### Pipeline 4: Detector + Tracker + Classifiers Pipeline With RTSP Outpus

This example is a combination of example 2 and 3 - the AI inferences from example 2 and the rtsp output from example 3.

1. Add all the relevant elements to the pipeline
2. Link all the elements.
3. Run the example.

> [!NOTE]
> Notice the input is a h264 stream. You can use ```https://anyconv.com/mp4-to-h264-converter/``` to convert standard video.
> To run the test app: Go to Run and Debug (on the left) and choose "Lab 1"
> You can see the run args (which is the input video) in ```launch.json```
