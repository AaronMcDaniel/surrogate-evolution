
import torch
import numpy as np
import criterion as c
import pytest

if __name__ == "__main__":
    pytest.main()

pred_boxes = torch.tensor([[5, 5, 4, 4]], dtype=torch.float32)
true_boxes = torch.tensor([[6, 6, 2, 2]], dtype=torch.float32)
print(c.diou_loss(pred_boxes, true_boxes))
print(c.iou_loss(pred_boxes, true_boxes)) # iou = .25

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


