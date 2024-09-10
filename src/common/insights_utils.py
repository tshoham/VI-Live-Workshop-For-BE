import json
import os
import yaml
import configparser

from src.constants import DEEPSTREAM_CONFIGS_DIR

class BBoxData:
    def __init__(self, obj_id, class_name, frame, x, y, width, height, confidence):
        self.obj_id = obj_id
        self.class_name = class_name
        self.frame = frame
        self.x = round(x, 3)
        self.y = round(y, 3)
        self.width = round(width, 3)
        self.height = round(height, 3)
        self.confidence = round(confidence, 2)

    def get_bbox_list(self):
        return [self.x, self.y, self.width, self.height]
    
    def get_bbox_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    def get_confidence(self):
        return self.confidence
    
    def get_class_name(self):
        return self.class_name
    
    def get_obj_id(self):
        return self.obj_id
    

class FrameData:
    def __init__(self, frame:int):
        self.frame = frame
        self.bboxes = []
        self.detector_bboxes = []

    def empty_bboxes(self):
        return len(self.bboxes) == 0
    
    def empty_detector_bboxes(self):
        return len(self.detector_bboxes) == 0
    
    @property
    def empty(self):
        return self.empty_bboxes() and self.empty_detector_bboxes()
        
    def parse_bboxes_as_raw_detections(self) -> dict:
        scores =[]
        boxes = []
        labels = []
        for detection in self.detector_bboxes:
            scores.append(detection.get_confidence())
            boxes.append(detection.get_bbox_list())
            labels.append(detection.get_class_name())
        
        return {
            "frame_id": self.frame,
            "scores": scores,
            "boxes": boxes,
            "labels": labels
        }
    
    def parse_bboxes_as_insights_by_obj_id(self) -> dict:
        insights_by_obj_id = {}
        for insight in self.bboxes:
            obj_id = insight.get_obj_id()
            if obj_id not in insights_by_obj_id:
                insights_by_obj_id[obj_id] = {
                    "id": obj_id,
                    "className": insight.get_class_name(),
                    "instances": []
                }
            
            instances = insight.get_bbox_dict()
            instances['frame'] = self.frame
            instances['confidence'] = insight.get_confidence()
            insights_by_obj_id[obj_id]['instances'].append(instances)

        return insights_by_obj_id


def read_tracker_config(tracker_config_path):
    config = configparser.ConfigParser()
    config.read(tracker_config_path)
    config.sections()

    tracker_data = {}
    for key in config["tracker"]:
        tracker_data[key] = config.get("tracker", key)
    
    if 'll-config-file' in tracker_data and tracker_data['ll-config-file']:
        ll_config_file = str(DEEPSTREAM_CONFIGS_DIR / tracker_data['ll-config-file'])
        if os.path.exists(ll_config_file):
            with open(ll_config_file) as f:
                lines = f.readlines()
        else:
            print(f"WARNING: When writing insights, file {ll_config_file} does not exist.")
            return tracker_data
        
        if lines[0].startswith('%YAML:1.0'):
            lines = lines[1:]
        yaml_content = ''.join(lines) 

        try:
            yaml_data = yaml.safe_load(yaml_content)
            tracker_data['ll-config'] = yaml_data
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file when writing insights: {exc}")

    return tracker_data


def parse_results(insights_data, stream_id_and_name, raw_detections=False):
    insights_by_obj_id = {}
    raw_detections_list = []
    
    for frame_data in insights_data[stream_id_and_name]:
        if raw_detections and not frame_data.empty_detector_bboxes():
            raw_detections_list.append(frame_data.parse_bboxes_as_raw_detections())
        for obj_id, obj_data in frame_data.parse_bboxes_as_insights_by_obj_id().items():
            if obj_id not in insights_by_obj_id:
                insights_by_obj_id[obj_id] = {
                    "id": obj_id,
                    "className": obj_data['className'],
                    "instances": []
                }
            insights_by_obj_id[obj_id]['instances'].extend(obj_data['instances'])
    
    return list(insights_by_obj_id.values()), raw_detections_list


def write_insights_on_cleanup(insights_data, stream_metadata, insights_output_path, raw_detections=False, tracker_data=None):
    for stream_id_and_name, metadata in stream_metadata.items():
        stream_id = stream_id_and_name.split("_")[0]
        stream_name = '_'.join(stream_id_and_name.split('_')[1:])
        res_dict = {}
        res_dict['streamID'] = int(stream_id)
        res_dict['streamName'] = stream_name
        res_dict['frameWidth'] = metadata['frameWidth']
        res_dict['frameHeight'] = metadata['frameHeight']

        res_dict['results'], raw_detections_list = parse_results(insights_data, stream_id_and_name, raw_detections)
        if raw_detections_list != []:
            res_dict['raw_detections'] = raw_detections_list

        if tracker_data:
            res_dict['tracker'] = tracker_data

        print(f"Writing stream results {stream_id_and_name} to file")
        #with open(f"{insights_output_path}/{stream_id_and_name}.json", "w") as f:
        with open(f"{insights_output_path}/{stream_name}.json", "w") as f:
            json.dump(res_dict, f)

    return True


def generate_buffer_main_key(stream_id, stream_name):
    return f"{stream_id}_{stream_name}"