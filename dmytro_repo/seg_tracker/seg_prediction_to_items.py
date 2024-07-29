import numpy as np
import math
import common_utils
import config


def gaussian2D(shape, sigma_x, sigma_y):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-x * x / (2 * sigma_x * sigma_x) - y * y / (2 * sigma_y * sigma_y))
    h[h < 1e-4] = 0
    cy = (shape[0] - 1) // 2
    cx = (shape[1] - 1) // 2
    h[max(0, cy-2):cy+3, max(0, cx-2):cx+3] = 1.0
    return h, y + np.zeros_like(x), x + np.zeros_like(y)


def calc_iou(i1, i2):
    x1 = i1['cx'] + i1['offset'][0]
    y1 = i1['cy'] + i1['offset'][1]
    w1 = i1['w']
    h1 = i1['h']

    x2 = i2['cx'] + i2['offset'][0]
    y2 = i2['cy'] + i2['offset'][1]
    w2 = i2['w']
    h2 = i2['h']

    ix_min = max(x1-w1/2, x2-w2/2)
    iy_min = max(y1-h1/2, y2-h2/2)
    ix_max = min(x1+w1/2, x2+w2/2)
    iy_max = min(y1+h1/2, y2+h2/2)

    iw = max(ix_max - ix_min, 0.)
    ih = max(iy_max - iy_min, 0.)

    intersections = iw * ih
    unions = (i1['w'] * i1['h'] + i2['w'] * i2['h'] - intersections)

    iou = intersections / unions
    return iou


def pred_to_items(
        comb_pred,
        offset,
        size,
        tracking,
        distance,
        above_horizon,
        conf_threshold: float,
        pred_scale=8.0,
        x_offset=0,
        y_offset=0
):
    comb_pred = comb_pred.copy()
    res_items = []

    # pred_scale = 4.0

    h4, w4 = comb_pred.shape

    while True:
        y, x = common_utils.argmax2d(comb_pred)
        # conf = comb_pred[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].sum()
        conf = comb_pred[y, x]
        #if conf < conf_threshold:
        if len(res_items) >= 100:
            break

        w = 2 ** size[0, y, x]
        h = 2 ** size[1, y, x]
        d = 2 ** distance[y, x]

        w = min(512, max(8, w))
        h = min(512, max(8, h))
        # item_cls = cls[:, y, x]
        item_tracking = tracking[:, y, x] * config.OFFSET_SCALE

        cx_img = x
        cy_img = y
        cx = (x + 0.5) * pred_scale
        cy = (y + 0.5) * pred_scale

        new_item = dict(
            conf=conf,
            cx=cx+x_offset,
            cy=cy+y_offset,
            w=w,
            h=h,
            distance=d,
            tracking=list(item_tracking),
            offset=list(offset[:, y, x] * pred_scale),
            above_horizon=float(above_horizon[y, x])
        )

        overlaps = False
        for prev_item in res_items:
            if calc_iou(prev_item, new_item) > 0.025:
                overlaps = True
                # if conf > 0.5:
                #     print('overlaps:')
                #     print(prev_item)
                #     print(new_item)
                break
        if not overlaps:
            res_items.append(new_item)

        w = math.ceil(w * 2 / pred_scale)
        h = math.ceil(h * 2 / pred_scale)
        w = max(5, w // 2 * 2 + 1)
        h = max(5, h // 2 * 2 + 1)

        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 2, sigma_y=h / 2)

        # clip masks
        w2 = (w - 1) // 2
        h2 = (h - 1) // 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0

        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        comb_pred[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w] -= item_mask
    #print('res_items', len(res_items))
    return res_items


def test_pred_to_items():
    from dataset_tracking import  DetectionItem, render_y

    w = 256
    h = 200

    cx = 112
    cy = 133

    items = [
        DetectionItem(
            cls_name='Airplane',
            item_id=1,
            distance=500,
            cx=cx,
            cy=cy,
            w=32,
            h=32,
            above_horizon=1
        ),
    ]

    prev_items = [
        DetectionItem(
            cls_name='Airplane',
            item_id=1,
            distance=500,
            cx=cx + 64,
            cy=cy + 32,
            w=32,
            h=32,
            above_horizon=1
        ),
    ]

    y = render_y(items=items, prev_step_items=prev_items, pred_scale=4, w=w, h=h)
    pred = pred_to_items(
        comb_pred=1.0-y['cls'],
        offset=y['reg_offset'],
        size=y['reg_size'],
        tracking=y['reg_tracking'],
        distance=y['reg_distance'],
        above_horizon=y['above_horizon'],
        conf_threshold=0.1)

    print(pred)


if __name__ == '__main__':
    test_pred_to_items()

