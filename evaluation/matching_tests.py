import torch
import numpy as np
import pytest
import os
import sys
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
import utils as u

if __name__ == "__main__":
    pytest.main()

def test_perfect_match():
    pred_boxes = torch.tensor([[10, 20, 30, 40, 0.9], [15, 25, 35, 45, 0.8]])
    true_boxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    iou_thresh = 0.5
    conf_thresh = 0.5
    matched_preds, matched_truths, fp, fn = u.match_boxes1(pred_boxes, true_boxes, iou_thresh, conf_thresh)
    assert len(matched_preds) == 2
    assert len(fp) == 0
    assert len(fn) == 0

def test_no_match():
    pred_boxes = torch.tensor([[100, 200, 300, 400, 0.9], [150, 250, 350, 450, 0.8]])
    true_boxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    iou_thresh = 0.5
    conf_thresh = 0.5
    matched_preds, matched_truths, fp, fn = u.match_boxes1(pred_boxes, true_boxes, iou_thresh, conf_thresh)
    assert len(matched_preds) == 0
    assert len(fp) == 2
    assert len(fn) == 2

def test_partial_match():
    pred_boxes = torch.tensor([[10, 20, 30, 40, 0.9], [150, 250, 350, 450, 0.8]])
    true_boxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    iou_thresh = 0.5
    conf_thresh = 0.5
    matched_preds, matched_truths, fp, fn = u.match_boxes1(pred_boxes, true_boxes, iou_thresh, conf_thresh)
    assert len(matched_preds) == 1
    assert len(fp) == 1
    assert len(fn) == 1

def test_multiple_predictions_single_truth():
    pred_boxes = torch.tensor([[10, 20, 30, 40, 0.9], [10, 20, 30, 40, 0.85], [40, 50, 60, 70, 0.7]])
    true_boxes = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
    iou_thresh = 0.5
    conf_thresh = 0.5
    matched_preds, matched_truths, fp, fn = u.match_boxes1(pred_boxes, true_boxes, iou_thresh, conf_thresh)
    assert len(matched_preds) == 2
    assert len(fp) == 1
    assert len(fn) == 1

def test_multiple_truths_single_prediction():
    pred_boxes = torch.tensor([[10, 20, 30, 40, 0.9], [40, 50, 60, 70, 0.85]])
    true_boxes = torch.tensor([[10, 20, 30, 40], [10, 20, 30, 40]])
    iou_thresh = 0.5
    conf_thresh = 0.5
    matched_preds, matched_truths, fp, fn = u.match_boxes1(pred_boxes, true_boxes, iou_thresh, conf_thresh)
    assert len(matched_preds) == 1
    assert len(fp) == 1
    assert len(fn) == 1

def test_low_confidence_predictions():
    pred_boxes = torch.tensor([[10, 20, 30, 40, 0.4], [15, 25, 35, 45, 0.3]])
    true_boxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    iou_thresh = 0.5
    conf_thresh = 0.5
    matched_preds, matched_truths, fp, fn = u.match_boxes1(pred_boxes, true_boxes, iou_thresh, conf_thresh)
    assert len(matched_preds) == 0
    assert len(fp) == 0
    assert len(fn) == 2