"""Fixed supervised linear evidence fusion. Runtime has no fit/teacher/review imports."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .multiview import MultiViewGeometryVision

FEATURE_NAMES=("rack_bright","rack_dark","source_occluded","robot_at_board","tool_on_board",
    "near_holder_fraction","wrist_available","blade_visible","blade_joined","blade_height_fraction",
    "gripper_raw","gripper_open","gripper_closed","right_qvel_max","joined_and_closed")
CLASSES=("gripper","holder","unknown")

def fusion_features(observation,measurements,wrist,bright,dark,rules):
    raw=observation.observation_state[13] if observation.observation_state else None
    velocity=observation.observation_velocity
    opened=raw is not None and raw>=rules["gripper_open_raw"]
    closed=raw is not None and raw<=rules["gripper_closed_raw"]
    blade=wrist.get("blade")
    image=observation.images.get("cam_right_wrist")
    values=[bright,dark,float(measurements.get("source_occluded",False)),float(measurements.get("robot_at_board",False)),
        float(measurements.get("tool_on_board",False)),measurements.get("robot_near_holder_projection",{}).get("new_dark_fraction",0.),
        float(wrist.get("available",False)),float(blade is not None),float(wrist.get("joined",False)),
        float(blade[3]/image.height) if blade is not None and image else 0.,raw if raw is not None else 0.,
        float(opened),float(closed),min(5.,max(abs(v) for v in velocity[7:13])) if velocity else 0.,
        float(bool(wrist.get("joined")) and closed)]
    value=np.asarray(values,dtype=np.float64)
    if value.shape!=(len(FEATURE_NAMES),) or not np.isfinite(value).all():
        raise ValueError("invalid_student_features")
    return value

class FixedFusion:
    def __init__(self,path):
        path=Path(path)
        if path.stat().st_size>64000:
            raise ValueError("student_model_capacity")
        raw=path.read_bytes()
        self.sha256=hashlib.sha256(raw).hexdigest()
        data=json.loads(raw)
        if data.get("schema")!="fixed_softmax_fusion_v1" or data["features"]!=list(FEATURE_NAMES) or data["classes"]!=list(CLASSES):
            raise ValueError("student_model_schema")
        for name,shape in (("mean",(15,)),("scale",(15,)),("coef",(3,15)),("intercept",(3,))):
            value=np.asarray(data[name],dtype=np.float64)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError("student_parameter_shape")
            value.flags.writeable=False
            setattr(self,name,value)
        if (self.scale<=0).any():raise ValueError("student_scale")
        self.minimum_score=float(data["minimum_score"])
        if not .34<=self.minimum_score<=1.:raise ValueError("student_score_threshold")

    def predict(self,x):
        logits=self.coef@((x-self.mean)/self.scale)+self.intercept
        scores=np.exp(logits-logits.max());scores/=scores.sum()
        index=int(scores.argmax())
        label=CLASSES[index] if scores[index]>=self.minimum_score else "unknown"
        return label,scores

class TrainedFusionVision(MultiViewGeometryVision):
    provenance="real_fixed_supervised_fusion"

    def __init__(self,config,rules,model_path):
        self.student=FixedFusion(model_path)
        super().__init__(config,rules)

    def reset(self):
        super().reset()
        self.forward_calls=0

    def query_wrist(self,rack_present,rack_clear,opened):
        # Same two current views at training and inference, plus one bounded reference.
        return True

    def relation_prediction(self,observation,measurements,wrist,bright,dark):
        x=fusion_features(observation,measurements,wrist,bright,dark,self.rules)
        self.forward_calls+=1
        label,scores=self.student.predict(x)
        predicted_label=label
        guard=None
        usable=wrist.get("available") and observation.observation_state is not None
        if not usable:label="unknown"
        # Learned relation cannot invent a connected blade without current pixel evidence.
        if label=="gripper" and not (wrist.get("joined") and x[12]):label="unknown"
        if usable and not measurements.get("source_occluded") and bright>=self.config.get("rack_blade_present_fraction",.60) and dark<.35 and not wrist.get("joined"):
            label="holder";guard="clear_holder_and_current_wrist_no_connection"
        elif label=="holder" and wrist.get("joined") and x[12]:
            label="unknown";guard="holder_proxy_conflicts_with_current_connection"
        measurements["learned_relation"]={"label":label,"scores":{name:round(float(v),6) for name,v in zip(CLASSES,scores)},
            "predicted_label":predicted_label,"evidence_guard":guard,
            "score_meaning":"uncalibrated_softmax_not_accuracy_probability","features":[round(float(v),7) for v in x],
            "feature_version":"fixed_fusion_features_v1","parameter_sha256":self.student.sha256,
            "forward_calls":self.forward_calls,"online_updates":0,"current_inputs_usable":bool(usable)}
        return label

    def usage(self):
        return super().usage()|{"student_parameter_values":78,"student_history_items":0,"student_mutable_state_slots":1,
                               "training_updates":0,"suggestion_history_items":0}
