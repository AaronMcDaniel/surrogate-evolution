
import os
import sys
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
import utils as u
import torch
import numpy as np
import criterion as c
import pytest
import torchvision.ops as ops


if __name__ == "__main__":
    pytest.main()

# print debugging

# pred_boxes = torch.tensor([[5, 5, 4, 4]], dtype=torch.float32)
# true_boxes = torch.tensor([[6, 6, 2, 2]], dtype=torch.float32)
# # correct iou = 0.25, correct diou = 0.25

# # returns 0-D tensor scalar value
# print(c.diou_loss(pred_boxes, true_boxes))
# print(c.iou_loss(pred_boxes, true_boxes))

# # returns 2-D tensor 
# print(u.iou(pred_boxes, true_boxes))
# print(u.diou(pred_boxes, true_boxes))

# ops.box_iou takes in [x1, y1, x2, y2], so need to convert before testing
# test_p = u.convert_boxes_to_x1y1x2y2(pred_boxes)
# test_t = u.convert_boxes_to_x1y1x2y2(true_boxes)
# returns 2-D tensor
# print(1 - ops.box_iou(test_p, test_t))

pred_boxes = torch.tensor([
    [0, 0, 10, 10, 0.9],
    [1, 1, 9, 9, 0.85],
    [15, 15, 25, 25, 0.95],
    [30, 30, 40, 40, 0.8],
    [35, 35, 45, 45, 0.75],
    [30, 30, 40, 40, 0.7],
    [50, 50, 60, 60, 0.6],
    [0, 0, 5, 5, 0.95],
    [1, 1, 5, 5, 0.6]
])
true_boxes = torch.tensor([
    [0, 0, 10, 10],
    [30, 30, 40, 40],
    [50, 50, 60, 60],
    [0, 0, 5, 5]
])
test_p = u.convert_boxes_to_x1y1x2y2(pred_boxes)
test_t = u.convert_boxes_to_x1y1x2y2(true_boxes)
iou_torchvision = ops.box_iou(test_p, test_t)
iou_custom = u.iou_matrix(pred_boxes, true_boxes)
diou_custom = u.iou_matrix(pred_boxes, true_boxes, "diou")
print("iou with torchvision:")
print(iou_torchvision)
print("iou with custom function:")
print(iou_custom)
assert torch.allclose(iou_torchvision, iou_custom, atol=1e-6), "custom function is wrong"
print("diou with custom func")
print(diou_custom)

# tests

def test_iou_identical_boxes():
    pred_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    true_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    expected_iou = torch.tensor([0.0, 0.0])
    iou_value = c.iou_loss(pred_boxes, true_boxes)
    # checks if tensors are element-wise equal within a certain precision
    assert torch.allclose(iou_value, expected_iou, atol=1e-6), f"Expected {expected_iou}, but got {iou_value}"

def test_iou_non_overlapping_boxes():
    pred_boxes = torch.tensor([[0, 0, 1, 1], [5, 5, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [8, 8, 2, 2]], dtype=torch.float32)
    expected_iou = torch.tensor([1.0, 1.0])
    expected_iou = expected_iou.mean()
    iou_value = c.iou_loss(pred_boxes, true_boxes)
    assert torch.allclose(iou_value, expected_iou, atol=1e-6), f"Expected {expected_iou}, but got {iou_value}"

def test_iou_partially_overlapping_boxes():
    pred_boxes = torch.tensor([[1, 1, 3, 3], [5, 5, 3, 3]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [6, 6, 3, 3]], dtype=torch.float32)
    expected_iou = torch.tensor([1 - ((2*2)/(3*3 + 3*3 - 2*2)), 1 - ((2*2)/(3*3 + 3*3 - 2*2))])
    expected_iou = expected_iou.mean()
    iou_value = c.iou_loss(pred_boxes, true_boxes)
    assert torch.allclose(iou_value, expected_iou, atol=1e-6), f"Expected {expected_iou}, but got {iou_value}"

def test_iou_one_box_inside_another():
    pred_boxes = torch.tensor([[1, 1, 4, 4], [5, 5, 4, 4]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [6, 6, 2, 2]], dtype=torch.float32)
    expected_iou = torch.tensor([1 - ((1*1)/(4*4)), 1 - ((2*2)/(4*4))])
    expected_iou = expected_iou.mean()
    iou_value = c.iou_loss(pred_boxes, true_boxes)
    assert torch.allclose(iou_value, expected_iou, atol=1e-6), f"Expected {expected_iou}, but got {iou_value}"

def test_iou_different_size_boxes_with_some_overlap():
    pred_boxes = torch.tensor([[1, 1, 5, 5], [3, 3, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [2, 2, 4, 4]], dtype=torch.float32)
    expected_iou = torch.tensor([1 - ((3*3)/(5*5 + 3*3 - 3*3)), 1 - ((2*2)/(2*2 + 4*4 - 2*2))])
    expected_iou = expected_iou.mean()
    iou_value = c.iou_loss(pred_boxes, true_boxes)
    assert torch.allclose(iou_value, expected_iou, atol=1e-6), f"Expected {expected_iou}, but got {iou_value}"

def test_diou_loss_identical_boxes():
    pred_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    true_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    expected_loss = 0.0
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, but got {loss}"

def test_diou_loss_non_overlapping_boxes():
    pred_boxes = torch.tensor([[0, 0, 1, 1], [5, 5, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [8, 8, 2, 2]], dtype=torch.float32)
    loss = c. diou_loss(pred_boxes, true_boxes).item()
    assert loss >= 1.0, f"Expected loss >= 1.0, but got {loss}"

def test_diou_loss_partially_overlapping_boxes():
    pred_boxes = torch.tensor([[1, 1, 3, 3], [5, 5, 3, 3]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [6, 6, 3, 3]], dtype=torch.float32)
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_diou_loss_one_box_inside_another():
    pred_boxes = torch.tensor([[1, 1, 4, 4], [5, 5, 4, 4]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [6, 6, 2, 2]], dtype=torch.float32)
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_diou_loss_different_size_boxes_with_some_overlap():
    pred_boxes = torch.tensor([[1, 1, 5, 5], [3, 3, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [2, 2, 4, 4]], dtype=torch.float32)
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_diou_loss_identical_boxes():
    pred_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    true_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    expected_loss = 0.0
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, but got {loss}"

def test_diou_loss_non_overlapping_boxes():
    pred_boxes = torch.tensor([[0, 0, 1, 1], [5, 5, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [8, 8, 2, 2]], dtype=torch.float32)
    loss = c. diou_loss(pred_boxes, true_boxes).item()
    assert loss >= 1.0, f"Expected loss >= 1.0, but got {loss}"

def test_diou_loss_partially_overlapping_boxes():
    pred_boxes = torch.tensor([[1, 1, 3, 3], [5, 5, 3, 3]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [6, 6, 3, 3]], dtype=torch.float32)
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_diou_loss_one_box_inside_another():
    pred_boxes = torch.tensor([[1, 1, 4, 4], [5, 5, 4, 4]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [6, 6, 2, 2]], dtype=torch.float32)
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_diou_loss_different_size_boxes_with_some_overlap():
    pred_boxes = torch.tensor([[1, 1, 5, 5], [3, 3, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [2, 2, 4, 4]], dtype=torch.float32)
    loss = c.diou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_giou_loss_identical_boxes():
    pred_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    true_boxes = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.float32)
    expected_loss = 0.0
    loss = c.giou_loss(pred_boxes, true_boxes).item()
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, but got {loss}"

def test_giou_loss_non_overlapping_boxes():
    pred_boxes = torch.tensor([[0, 0, 1, 1], [5, 5, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [8, 8, 2, 2]], dtype=torch.float32)
    loss = c.giou_loss(pred_boxes, true_boxes).item()
    expected_loss = 1.0
    assert loss >= expected_loss, f"Expected loss >= 1.0, but got {loss}"

def test_giou_loss_partially_overlapping_boxes():
    pred_boxes = torch.tensor([[1, 1, 3, 3], [5, 5, 3, 3]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [6, 6, 3, 3]], dtype=torch.float32)
    loss = c.giou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_giou_loss_one_box_inside_another():
    pred_boxes = torch.tensor([[1, 1, 4, 4], [5, 5, 4, 4]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 1, 1], [6, 6, 2, 2]], dtype=torch.float32)
    loss = c.giou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"

def test_giou_loss_different_size_boxes_with_some_overlap():
    pred_boxes = torch.tensor([[1, 1, 5, 5], [3, 3, 2, 2]], dtype=torch.float32)
    true_boxes = torch.tensor([[2, 2, 3, 3], [2, 2, 4, 4]], dtype=torch.float32)
    loss = c.giou_loss(pred_boxes, true_boxes).item()
    assert loss > 0 and loss < 1, f"Expected loss between 0 and 1, but got {loss}"


