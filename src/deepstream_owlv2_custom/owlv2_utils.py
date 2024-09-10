import ctypes
from dataclasses import dataclass

import numpy as np
import pyds
import torch
from transformers import AutoProcessor, Owlv2Model


# Constants for the network. We should make these common
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920

# We are using : "google/owlv2-base-patch16-ensemble": 960, (can be modified for larger models)
NETWORK_INPUT_SIZE = 960  # Size of the input image expected by the network 960x960
UNTRACKED_OBJECT_ID = 0xffffffffffffffff

num_patches_per_side = NETWORK_INPUT_SIZE // 16
num_patches = num_patches_per_side**2
# owl_get_patch_size - need to add to config
# "google/owlv2-base-patch16-ensemble": 16,
# "google/owlv2-large-patch14-ensemble": 14,

# see https://github.com/NVIDIA-AI-IOT/nanoowl/issues/6



@dataclass
class OwlEncodeTextOutput:
    text_embeds: torch.Tensor

    def slice(self, start_index, end_index):
        return OwlEncodeTextOutput(text_embeds=self.text_embeds[start_index:end_index])


@dataclass
class OwlEncodeImageOutput:
    image_embeds: torch.Tensor
    image_class_embeds: torch.Tensor
    logit_shift: torch.Tensor
    logit_scale: torch.Tensor
    pred_boxes: torch.Tensor
    objectness_logits:torch.Tensor


@dataclass
class OwlDecodeOutput:
    labels: torch.Tensor
    scores: torch.Tensor
    boxes: torch.Tensor
    objectness_logits: torch.Tensor


@dataclass
class OwlInfo:
    text: list[str]
    text_encodings: torch.Tensor
    thresholds: list[float]

    def __post_init__(self):
        if not isinstance(self.text, list):
            raise ValueError("text must be a list of strings")

        if len(self.text) != self.text_encodings.shape[0]:
            raise ValueError("Length of text list must be equal to length of text_encodings")

        if len(self.thresholds) != self.text_encodings.shape[0]:
            raise ValueError("Length of thresholds must be equal to length of text_encodings")

    def add_ex(self, text: list[str], encoding: torch.Tensor, threshold):
        """ Explicit add new text and encoding to the OwlInfo object. """

        self.text.extend(text)
        self.text_encodings = torch.cat([self.text_encodings, encoding], dim=0)
        self.thresholds.extend(threshold)

    def add(self, owl_info: "OwlInfo"):
        self.add_ex(owl_info.text, owl_info.text_encodings, owl_info.thresholds)

    def remove(self, index):
        """ Remove the text and encoding at the given index. """

        self.text.pop(index)
        self.text_encodings = torch.cat([self.text_encodings[:index], self.text_encodings[index + 1:]], dim=0)
        self.thresholds.pop(index)


class OwlV2TextEncoding:
    _instance = None

    def __new__(cls, model_name="google/owlv2-base-patch16-ensemble"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            cls._instance.model_name = model_name
            cls._instance.processor = AutoProcessor.from_pretrained(model_name)
            cls._instance.model = Owlv2Model.from_pretrained(model_name)

        return cls._instance

    def encode(self, text: str | list[str], thresholds: float | list[float] = 0.1) -> OwlInfo:
        if not isinstance(text, list):
            text = [text]

        inputs = self._instance.processor(text=text, return_tensors="pt")
        text_encodings = self._instance.model.get_text_features(**inputs)

        if not isinstance(thresholds, list):
            thresholds = [thresholds] * text_encodings.shape[0]

        owl_info = OwlInfo(text, text_encodings, thresholds)
        return owl_info


class OwlInfoState:
    _instance = None
    owl_info = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    @classmethod
    def add_embedding(cls, text: str | list[str], thresholds: float | list[float] = 0.1):
        owl_encoder = OwlV2TextEncoding()
        owl_info = owl_encoder.encode(text, thresholds)
        if cls.owl_info is None:
            cls.owl_info = owl_info
        else:
            cls.owl_info.add(owl_info)

    @classmethod
    def remove_embedding(cls, index):
        if cls.owl_info is not None:
            cls.owl_info.remove(index)

    @classmethod
    def get_embedding(cls):
        return cls.owl_info

    @classmethod
    def print_embedding(cls, verbose=False):
        for i in range(len(cls.owl_info.text)):
            print(f"{i}: {cls.owl_info.text[i]}, Threshold: {cls.owl_info.thresholds[i]}")
            if verbose:
                print(f"Embedding: {cls.owl_info.text_encodings[i]}")
        print()


def decode_owlv2(
    image_output : OwlEncodeImageOutput,
    text_output : OwlEncodeTextOutput,
    threshold: float | list[float] = 0.1,
    objectness_threshold: float = 0.1) -> OwlDecodeOutput:

    if isinstance(threshold, (int, float)):
        threshold = [threshold] * len(text_output.text_embeds)  # apply single threshold to all labels

    objectness = image_output.objectness_logits
    objectness_mask = objectness > objectness_threshold

    objectness_mask_2d = objectness_mask.reshape(objectness_mask.shape[0], -1)
    # num_filtered_objects = objectness_mask.sum(dim=1)
    # to log print(f'Number of objects that passed the threshold: {num_filtered_objects}')

    # Apply the mask to the data
    boxes = image_output.pred_boxes[objectness_mask_2d]
    logit_scale = image_output.logit_scale[objectness_mask_2d]
    logit_shift = image_output.logit_shift[objectness_mask_2d]
    image_class_embeds = image_output.image_class_embeds[objectness_mask_2d]
    objectness = objectness[objectness_mask_2d]

    # Normalize image_class_embeds
    norm = np.linalg.norm(image_class_embeds, axis=-1, keepdims=True)
    image_class_embeds = image_class_embeds / (norm + 1e-6)

    query_embeds = text_output.text_embeds.detach().numpy()
    # Normalize query_embeds
    norm = np.linalg.norm(query_embeds, axis=-1, keepdims=True)
    query_embeds = query_embeds / (norm + 1e-6)

    # Compute logits
    logits = np.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
    # pdb.set_trace()
    logits = (logits + logit_shift) * logit_scale

    # Compute scores
    scores_sigmoid = 1 / (1 + np.exp(-logits))  # Sigmoid function
    scores_max_values = scores_sigmoid.max(axis=-1)
    labels = scores_sigmoid.argmax(axis=-1)
    scores = scores_max_values

    # Compute masks
    masks = []
    for i, thresh in enumerate(threshold):
        label_mask = labels == i
        score_mask = scores > thresh
        obj_mask = np.logical_and(label_mask, score_mask)
        masks.append(obj_mask)

    mask = masks[0]
    for mask_t in masks[1:]:
        mask = np.logical_or(mask, mask_t)

    return OwlDecodeOutput(labels=labels[mask], scores=scores[mask], boxes=boxes[mask],
                           objectness_logits=objectness[mask])


def layer_finder(output_layer_info, name):
    """ Return the layer contained in output_layer_info which corresponds to the given name. """
    for layer in output_layer_info:
        # dataType == 0 <=> dataType == FLOAT
        if layer.dataType == 0 and layer.layerName == name:
            return layer

    return None


def make_nodi(i, bboxes, labels, scores):
    """ Creates a NvDsInferObjectDetectionInfo object from one layer of SSD.
        Return None if the class Id is invalid, if the detection confidence
        is under the threshold or if the width/height of the bounding box is
        null/negative.
        Return the created NvDsInferObjectDetectionInfo object otherwise.
    """
    res = pyds.NvDsInferObjectDetectionInfo()
    res.detectionConfidence = scores[i]  # Add confidence
    res.classId = int(labels[i])
    box = bboxes[i]
    # box = [int(x) for x in box] # assume left,top and right,bottom coordinates

    res.left = int(max(box[0]* IMAGE_WIDTH,0))
    res.top = int(max(box[1] * IMAGE_HEIGHT,0))
    res.width = int(min((box[2] - box[0]) * IMAGE_WIDTH, IMAGE_WIDTH - res.left))
    res.height = int(min((box[3] - box[1]) * IMAGE_HEIGHT, IMAGE_HEIGHT - res.top))

    if res.width <= 0 or res.height <= 0:
        return None

    # if res.detectionConfidence == 1.0:  # Confidence == 1.0 is invalid??
    #    return None

    # TODO add conditions for box size / confidence / class id / etc
    # if not box_size_param.is_percentage_sufficiant(res.height, res.width):
    #    return None
    return res


# Need to add the rest of the info
def nvds_infer_parse_custom_owl_output(output_layer_info, owl_info: OwlInfo):
    """ Get data from output_layer_info and fill object_list
        with several NvDsInferObjectDetectionInfo.

        Keyword arguments:
        - output_layer_info : represents the neural network's output.
            (NvDsInferLayerInfo list)

        Return:
        - Bounding boxes. (NvDsInferObjectDetectionInfo list)
    """
    text_encodings = owl_info.text_encodings
    thresholds = owl_info.thresholds

    num_detection = 0
    object_list = []

    # see https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-ssd-parser/ssd_parser.py

    image_output = OwlEncodeImageOutput(output_layer_info["image_embeds"], output_layer_info["image_class_embeds"],
                                        output_layer_info["logit_shift"], output_layer_info["logit_scale"],
                                        output_layer_info["pred_boxes"], output_layer_info["objectness_logits"])
    text_output = OwlEncodeTextOutput(text_encodings)

    output = decode_owlv2(image_output, text_output, thresholds, objectness_threshold=0.1)
    #output = decode_owlv2_pytorch(image_output, text_output, thresholds,objectness_threshold=0.1)

    owl_boxes = output.boxes

    num_detection = owl_boxes.shape[0]
    # print(f"bboxes: {num_detection}")

    if num_detection > 0:
        for i in range(num_detection):
            obj = make_nodi(i, owl_boxes, output.labels, output.scores)
            if obj:
                object_list.append(obj)

    return object_list


def get_layers_info(tensor_meta):
    layers_info = {}

    for i in range(tensor_meta.num_output_layers):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
        layer_name = layer.layerName
        ndims = layer.dims.numDims
        layer_dims = layer.dims.d

        # Get a pointer to the layer data, cast it to a ctypes pointer to float,
        # and then convert it to a numpy array.
        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
        layer_data = np.ctypeslib.as_array(ptr, shape=(layer.dims.numElements,))

        # Reshape the numpy array according to layer_dims
        new_shape = np.concatenate(([-1], layer_dims[:ndims]))
        reshaped_layer_data = layer_data.reshape(new_shape)
        # Add the reshaped numpy array to the dictionary
        layers_info[layer_name] = reshaped_layer_data

    return layers_info


def add_obj_meta_to_frame(frame_object, batch_meta, frame_meta, label_names):
    """ Inserts an object into the metadata """
    # This is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution
    rect_params = obj_meta.rect_params
    rect_params.left = frame_object.left
    rect_params.top = frame_object.top
    rect_params.width = frame_object.width
    rect_params.height = frame_object.height

    detector_bbox_info = obj_meta.detector_bbox_info
    detector_bbox_info.org_bbox_coords.left = frame_object.left
    detector_bbox_info.org_bbox_coords.top = frame_object.top
    detector_bbox_info.org_bbox_coords.width = frame_object.width
    detector_bbox_info.org_bbox_coords.height = frame_object.height

    # print(f"Left: {rect_params.left}, Top: {rect_params.top}, "
    #       f"Width: {rect_params.width}, Height: {rect_params.height}")

    # Set object info including class, detection confidence, etc
    obj_meta.confidence = frame_object.detectionConfidence
    obj_meta.class_id = frame_object.classId

    # There is no tracking ID upon detection. The tracker will assign an ID
    obj_meta.object_id = UNTRACKED_OBJECT_ID

    lbl_id = frame_object.classId
    if lbl_id >= len(label_names):
        lbl_id = 0

    # Set the object classification label
    obj_meta.obj_label = label_names[lbl_id]

    # Semi-transparent yellow backgroud
    rect_params.has_bg_color = 0
    rect_params.bg_color.set(1, 1, 0, 0.4)

    # Red border of width 3
    rect_params.border_width = 3
    rect_params.border_color.set(1, 0, 0, 1)

    # Set display text for the object
    txt_params = obj_meta.text_params
    if txt_params.display_text:
        pyds.free_buffer(txt_params.display_text)

    txt_params.x_offset = int(rect_params.left)
    txt_params.y_offset = max(0, int(rect_params.top) - 10)
    txt_params.display_text = label_names[lbl_id] + " " + "{:0.2f}".format(frame_object.detectionConfidence)

    # Font , font-color and font-size
    txt_params.font_params.font_name = "Serif"
    txt_params.font_params.font_size = 10
    # set(red, green, blue, alpha); set to White
    txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    # Text background color
    txt_params.set_bg_clr = 1
    # set(red, green, blue, alpha); set to Black
    txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

    # Insert the object into current frame meta, this object has no parent
    obj_meta.unique_component_id = 1 # this value has been confirmed with nvinferserver's config

    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)


def expand_prompt(prompt, color_obj_list):
    basic_colors = ["red", "blue", "black", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "white"]

    # Replace the prompt with colored versions
    color_prompt = []
    for item in prompt:
        print(f"{item=}")
        print(f"{color_obj_list=}")
        if any(obj in item for obj in color_obj_list):
            item = item.strip()[2:]
            color_prompt.extend([f"a {color} {item}" for color in basic_colors])
        #else:
        #    new_prompt.append(item)

    return color_prompt
