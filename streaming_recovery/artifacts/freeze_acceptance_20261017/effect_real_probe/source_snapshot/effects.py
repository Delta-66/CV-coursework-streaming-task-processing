"""有界的操作—效果关联。记录首个已观察变化，不把初始图差异追认给较晚动作。"""
from copy import deepcopy
from math import isfinite
import json


class CutEffectTracker:
    def __init__(self,rules):
        self.rules=rules
        self.last_layout=None
        self.operation=None
        self.change=None
        self.last_time=None
        self.interrupted=False

    def interrupt(self,now):
        self.last_layout=self.operation=self.change=None
        self.last_time=now
        self.interrupted=True

    def _matches(self,a,b):
        threshold=self.rules.get("effect_piece_match_pixels",12.)
        return all(abs(x-y)<=threshold for x,y in zip((a[0]+a[2]/2,a[1]+a[3]/2),
                                                       (b[0]+b[2]/2,b[1]+b[3]/2)))

    def _remaining(self,old,current):
        # 一对一保留全部旧块，再判断额外新块；旧块单纯移位/消失不是新增。
        remaining=list(range(len(current)))
        for box in old:
            matches=[i for i in remaining if self._matches(box,current[i])]
            if len(matches)!=1:
                return None
            remaining.remove(matches[0])
        return remaining

    def _layout(self,facts):
        value=facts.measurements.get("cut_layout")
        if value is None:
            if facts.cut_piece not in ("yes","no"):
                return None,"local_layout_unknown"
            pieces=None
        else:
            keys={"timestamp","ref","camera","frame","image_size","clear","small"}
            if (not isinstance(value,dict) or set(value)!=keys or value["timestamp"]!=facts.timestamp or
                value["ref"]!=facts.ref or value["camera"]!="cam_high" or value["frame"]!="cam_high_image_pixels" or
                type(value["clear"]) is not bool or not isinstance(value["image_size"],list) or len(value["image_size"])!=2 or
                any(type(n) is not int or not 1<=n<=512 for n in value["image_size"]) or
                not isinstance(value["small"],list) or len(value["small"])>4):
                return None,"invalid_layout_contract"
            pieces=value["small"]
            for box in pieces:
                if (not isinstance(box,list) or len(box)!=4 or
                    any(isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) for v in box) or
                    min(box)<0 or box[2]<=0 or box[3]<=0 or
                    box[0]+box[2]>value["image_size"][0] or box[1]+box[3]>value["image_size"][1]):
                    return None,"invalid_layout_contract"
            if not value["clear"]:
                return None,"local_layout_unknown"
            if facts.cut_piece not in ("yes","no") or (facts.cut_piece=="yes")!=bool(pieces):
                return None,"invalid_layout_contract"
        layout={"timestamp":facts.timestamp,"ref":facts.ref,"cut_piece":facts.cut_piece,
                "small":deepcopy(pieces),"first_seen_at":facts.timestamp,"first_seen_ref":facts.ref}
        previous=self.last_layout
        if previous and previous["cut_piece"]==layout["cut_piece"]:
            same=(previous["small"] is None and pieces is None) or (previous["small"] is not None and pieces is not None and
                   len(previous["small"])==len(pieces) and self._remaining(previous["small"],pieces)==[])
            if same:
                layout["first_seen_at"],layout["first_seen_ref"]=previous["first_seen_at"],previous["first_seen_ref"]
        return layout,"available"

    def observe(self,facts,operation):
        now=facts.timestamp
        if self.last_time is not None and now-self.last_time>self.rules["max_evidence_gap_s"]:
            self.interrupt(now)
        self.last_time=now
        layout,reason=self._layout(facts)
        if reason=="invalid_layout_contract":
            self.interrupt(now)
            return self.summary(False,reason)
        if self.operation is None and operation and self.last_layout is not None and self.last_layout["timestamp"]<now:
            self.operation={"timestamp":now,"ref":facts.ref,"baseline":deepcopy(self.last_layout)}
            self.change=None
            self.interrupted=False
        supported=False
        if layout is not None and self.operation is not None:
            baseline=self.operation["baseline"]
            basis=None
            box=None
            if baseline["cut_piece"]=="no" and layout["cut_piece"]=="yes":
                basis="observed_no_to_yes_transition"
            elif (baseline["cut_piece"]=="yes" and baseline["small"] is not None and layout["small"] is not None):
                remaining=self._remaining(baseline["small"],layout["small"])
                if remaining:
                    basis="previous_pieces_retained_plus_new_component"
                    box=layout["small"][remaining[0]]
            if basis:
                same=(self.change is not None and self.change["basis"]==basis and
                      ((box is None and self.change["box"] is None) or
                       (box is not None and self.change["box"] is not None and self._matches(box,self.change["box"]))))
                if not same:
                    self.change={"timestamp":layout["first_seen_at"],"ref":layout["first_seen_ref"],
                                 "operation_timestamp":self.operation["timestamp"],"basis":basis,"box":deepcopy(box)}
                supported=self.change["timestamp"]>self.operation["timestamp"]
                reason="ordered_new_effect" if supported else "effect_not_strictly_after_operation"
            else:
                self.change=None
                reason="no_distinct_new_effect_relative_to_operation"
        if layout is not None:
            self.last_layout=layout
        return self.summary(supported,reason)

    def summary(self,supported=False,reason="observation_interrupted"):
        baseline=self.operation["baseline"] if self.operation else None
        return {"supports_current_operation":supported,"reason":reason,
                "operation_ref":({"timestamp":self.operation["timestamp"],"ref":self.operation["ref"]} if self.operation else None),
                "baseline_ref":({k:baseline[k] for k in ("timestamp","ref","cut_piece","first_seen_at","first_seen_ref")} if baseline else None),
                "first_change":deepcopy(self.change),"interrupted":self.interrupted}

    def usage(self):
        layouts=[x for x in (self.last_layout,self.operation["baseline"] if self.operation else None) if x is not None]
        return {"effect_state_slots":5,"effect_layout_records":len(layouts),"effect_layout_boxes":sum(len(x["small"] or []) for x in layouts),
                "effect_operation_records":int(self.operation is not None),"effect_change_records":int(self.change is not None),
                "effect_change_boxes":int(self.change is not None and self.change["box"] is not None),
                "effect_evidence_refs":2*len(layouts)+int(self.operation is not None)+int(self.change is not None),
                "effect_state_chars":len(json.dumps({k:v for k,v in vars(self).items() if k!="rules"},ensure_ascii=False))}
