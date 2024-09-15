# Lab 2

## Background
VI Live will need to handle many streams simultaniously, which is one of the benefits of Nvidia's Deepstream.
As we mentioned, Deepstream efficiently processes multiple videos simultaneously.

This lab will highlight multi-stream processing within the pipeline.

In addition, this pipeline will write results to a file - this is what we will most likely use in VI since the BE needs to receive the results and save them. 

## Purpose
lab2.py is a simplification of pipelines that data science team created. 
This example allows for a lot of versatile pipelines. 

We will focus on a simple pipeline that gets input from file sources, and outputs the results to files. In our example, the pipeline will run a detector and tracker. The pipeline will also write insights to a file.

We will write the part of the code that handle multiple inputs.

> [!IMPORTANT]
> In lab1 we used h264 files and had file-source, a parser, and decoder.
> 
> In lab2, the code uses `uridecodebin` so that any type of input (e.g. RTSP/File) that is GStreamer supported container format and any codec can be used as input.
>
> Here the "source" is based on Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin") and Gst.Bin.
>
> Gst.Bin is a GStreamer container that hides the complexity of internal elements from the pipeline.
> To take a closer look at this code you can checkout the function ```create_source_bin```

## Instructions

1. Write the function `def create_and_link_sources(src_config: SrcConfig, number_sources, streammux, pipeline)` (Look for it in the code and write the implementation)

    The paramenter `src_config` has an array of input_uris `src_config.input_uris`.
    Loop through these uris. 
    
    For each index do the following:

    1. Call function `create_source_bin(src_config, i)` to create the source element.
        - Where `i` is the index of the loop

    2. Add the source element you created to the pipeline (`pipeline.add(src_element)`
    3. Create a sinkpad for the streammux and srcpad for the source using the following syntax
        `sinkpad = streammux.request_pad_simple(sinkpad_name)`
        `srcpad = source_bin.get_static_pad("src")`
        - **The sinkpad_name will be `"sink_i"`.**
    4. Link the srcpad to the sinkpad

        > **NOTE:**
        > Pads are interfaces through which data flows in and out of elements.
        > An element can have multiple pads, so in this case we have a new pad for each source.
        > Pads allow  modularity, flexibility, data flow control, dynamic linking, and compatibility.

2. Set the property `batch-size` of the pgie element and for streamumx (look for "Your implementation goes here" in the code)

    1. Set the "batch-size" property of the streammux to the number of sources: `streammux.set_property("batch-size", number_sources)`
    1. Set the "batch-size" property of the pgie to the number of sources: `pgie.set_property("batch-size", number_sources)`

> [!NOTE]
> Batch-size can be set in the AI inference configuration file, and can also be updated from within the pipeline.
> This is something we will need to update in VI when cameras are added/removed.

3. Open the folder in dev container and run Lab 2.
4. View the file created in your local machine (copy to you compputer). Notice the tiled results.

