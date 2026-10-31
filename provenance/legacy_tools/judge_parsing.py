"""Versioned Experiment 2 postprocessing; preserve all original model output."""

import json


def parse_judgment(text, has_context):
    decoder = json.JSONDecoder()
    objects = []
    position = text.find("{")
    while position >= 0:
        try:
            candidate, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            if objects and "}" not in text[position:]:
                break
            raise ValueError("Malformed judge JSON object.")
        objects.append(validate_judgment(candidate, has_context))
        position = text.find("{", end)
    if not objects:
        raise ValueError("Judge did not produce a JSON object.")
    metrics = ["correctness", "relevance", "faithfulness"]
    if any(any(candidate[metric] != objects[0][metric] for metric in metrics)
           for candidate in objects[1:]):
        raise ValueError("Conflicting complete judge JSON objects.")
    return objects[0]


def validate_judgment(scores, has_context):
    if not isinstance(scores, dict) or not {"correctness", "relevance", "faithfulness", "reason"} <= set(scores):
        raise ValueError("Judge JSON is missing required rubric fields.")
    for metric in ["correctness", "relevance", "faithfulness"]:
        value = scores.get(metric)
        if metric == "faithfulness" and value is None:
            continue
        if type(value) is not int or value not in [0, 1, 2]:
            raise ValueError(f"Invalid {metric} score.")
    if not isinstance(scores.get("reason"), str) or not scores["reason"].strip():
        raise ValueError("Missing judge explanation.")
    if not has_context:
        scores["faithfulness"] = None
    return {key: scores[key] for key in ["correctness", "relevance", "faithfulness", "reason"]}
