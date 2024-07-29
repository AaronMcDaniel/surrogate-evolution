import math
import numpy as np
import scipy
import scipy.optimize
import common_utils
from sklearn.linear_model import LinearRegression


def offset_grid_to_transform_params(prev_frame_points, cur_frame_points, points_weight):
    """
    :param prev_frame_points: ndarray, shape 2xN, coordinates of points on the prev frame
    :param cur_frame_points: ndarray, shape 2xN, coordinates of points on the current frame
    :param points_weight: ndarray, shape N, weight of each point
    :return: dx, dy, angle, so
    cur_frame_points = T(dx,dy,angle) * prev_frame_points

    points have zero at the image center
    """
    points_weight = points_weight.astype(np.double)/points_weight.sum()

    dxy0 = -1*((cur_frame_points - prev_frame_points) * points_weight[None, :]).sum(axis=1)

    cx = cur_frame_points[0].astype(np.double)
    cy = cur_frame_points[1].astype(np.double)

    px = prev_frame_points[0].astype(np.double)
    py = prev_frame_points[1].astype(np.double)

    def cost(x):
        dx, dy, a = x
        a = a / 1000.0
        # ca = math.cos(a)
        # sa = math.sin(a)

        # pred_px = ca*cx - sa*cy + dx
        # pred_py = sa*cx + ca*cy + dy

        pred_px = cx - a * cy + dx
        pred_py = a * cx + cy + dy

        err = points_weight * ((px-pred_px) ** 2 + (py-pred_py) ** 2)
        return err.sum()

    def der(x):
        dx, dy, a = x
        a = a / 1000.0

        pred_px = cx - a * cy + dx
        pred_py = a * cx + cy + dy

        ddx = points_weight * 2 * (pred_px - px)
        ddy = points_weight * 2 * (pred_py - py)
        dda = points_weight * 2 * (cx * (pred_py - py) - cy * (pred_px - px)) / 1000.0
        return ddx.sum(), ddy.sum(), dda.sum()


    # check der
    # print(cost([1, 2, 3]), der([1, 2, 3]))
    # for i in range(3):
    #     x = np.array([1.0, 2.0, 3.0])
    #     x[i] += 0.001
    #     print((cost(x) - cost([1, 2, 3]))/0.001)

    x0 = np.array([dxy0[0], dxy0[1], 0.0])
    # x0 = np.array([0.0, 0.0, 0.0])
    res = scipy.optimize.minimize(cost, x0, jac=der, method='BFGS', options={'gtol': 1e-6, 'disp': False})
    # print(res)
    return res.x[0], res.x[1], res.x[2] / 1000.0 * 180 / math.pi, res.fun  # cost(res.x)


def offset_grid_to_transform(prev_frame_points, cur_frame_points, points_weight):
    """
    :param prev_frame_points: ndarray, shape 2xN, coordinates of points on the prev frame
    :param cur_frame_points: ndarray, shape 2xN, coordinates of points on the current frame
    :param points_weight: ndarray, shape N, weight of each point
    :return: T, so
    cur_frame_points = T * [prev_frame_points.T, 1].T
    """
    points_weight = points_weight.astype(np.double)/points_weight.sum()

    mx = LinearRegression()
    mx.fit(prev_frame_points.transpose(), cur_frame_points[0], sample_weight=points_weight)
    my = LinearRegression()
    my.fit(prev_frame_points.transpose(), cur_frame_points[1], sample_weight=points_weight)

    t = np.array([
        [mx.coef_[0], mx.coef_[1], mx.intercept_],
        [my.coef_[0], my.coef_[1], my.intercept_],
        [0, 0, 1]
    ])
    return t, 0




def test_offset_grid_to_transform_params():
    center = np.array([512.0, 512.0])[:, None]

    prev_points = np.zeros((2, 32, 32), dtype=np.float32)
    prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
    prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]

    prev_points_1d = prev_points.reshape((2, -1))

    dx = 64
    dy = 42
    angle = 0.5

    transform = common_utils.build_geom_transform(
        dst_w=1024,
        dst_h=1024,
        src_center_x=512.0 + dx,
        src_center_y=512.0 + dy,
        scale_x=1.0,
        scale_y=1.0,
        angle=angle,
        return_params=True
    )

    cur_points = ((transform[:2, :2] @ prev_points_1d).T + transform[:2, 2]).T
    # cur_points = cur_points.reshape((2, 32, 32))

    points_weight = np.ones((prev_points_1d.shape[1],), dtype=np.float32)

    with common_utils.timeit_context('estimate transformation'):
        dx_pred, dy_pred, angle_pred, err = offset_grid_to_transform_params(prev_points_1d-center, cur_points-center, points_weight)

    print(dx_pred, dy_pred, angle_pred, err)
    assert abs(dx_pred - dx) < 0.01
    assert abs(dy_pred - dy) < 0.01
    assert abs(angle_pred - angle) < 0.001
    assert err < 0.001

    cur_points = np.random.normal(cur_points, np.ones_like(cur_points))

    with common_utils.timeit_context('estimate transformation'):
        dx_pred, dy_pred, angle_pred, err = offset_grid_to_transform_params(prev_points_1d-center, cur_points-center, points_weight)

    print(dx_pred, dy_pred, angle_pred, err)
    assert abs(dx_pred - dx) < 0.1
    assert abs(dy_pred - dy) < 0.1
    assert abs(angle_pred - angle) < 0.01
    expected_err = 2.0
    assert abs(err - expected_err) < 0.2

    # test with the custom weight
    points_weight[:len(points_weight) // 2] = 2.0
    with common_utils.timeit_context('estimate transformation'):
        dx_pred, dy_pred, angle_pred, err = offset_grid_to_transform_params(prev_points_1d-center, cur_points-center, points_weight)

    print(dx_pred, dy_pred, angle_pred, err)
    assert abs(dx_pred - dx) < 0.1
    assert abs(dy_pred - dy) < 0.1
    assert abs(angle_pred - angle) < 0.01


def test_offset_grid_to_transform():
    center = np.array([512.0, 512.0])[:, None]

    prev_points = np.zeros((2, 32, 32), dtype=np.float32)
    prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
    prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]

    prev_points_1d = prev_points.reshape((2, -1))

    dx = 64
    dy = 42
    angle = 0.5

    transform = common_utils.build_geom_transform(
        dst_w=1024,
        dst_h=1024,
        src_center_x=512.0 + dx,
        src_center_y=512.0 + dy,
        scale_x=1.1,
        scale_y=0.9,
        angle=angle,
        return_params=True
    )

    cur_points = ((transform[:2, :2] @ prev_points_1d).T + transform[:2, 2]).T
    # cur_points = cur_points.reshape((2, 32, 32))

    points_weight = np.ones((prev_points_1d.shape[1],), dtype=np.float32)

    with common_utils.timeit_context('estimate transformation'):
        t, err = offset_grid_to_transform(prev_points_1d, cur_points, points_weight)

    cur_points_pred = ((t[:2, :2] @ prev_points_1d).T + t[:2, 2]).T

    print(transform)
    print(t)
    print(err, np.abs(t - transform).max())
    print(np.abs(cur_points_pred - cur_points).max())
    assert np.abs(t - transform).max() < 1e-3
    assert np.abs(cur_points_pred - cur_points).max() < 0.1

    print('Add noise')
    cur_points_orig = cur_points
    cur_points = np.random.normal(cur_points, np.ones_like(cur_points))

    with common_utils.timeit_context('estimate transformation'):
        t, err = offset_grid_to_transform(prev_points_1d, cur_points, points_weight)

    print(t)
    cur_points_pred = ((t[:2, :2] @ prev_points_1d).T + t[:2, 2]).T
    print(np.abs(t - transform).max())
    print(np.abs(cur_points_pred - cur_points_orig).max())
    assert np.abs(t - transform).max() < 0.2
    assert np.abs(cur_points_pred - cur_points_orig).max() < 1

    # test with the custom weight
    points_weight[:len(points_weight) // 2] = 2.0
    with common_utils.timeit_context('estimate transformation'):
        t, err = offset_grid_to_transform(prev_points_1d, cur_points, points_weight)

    print(t)
    cur_points_pred = ((t[:2, :2] @ prev_points_1d).T + t[:2, 2]).T
    print(np.abs(t - transform).max())
    print(np.abs(cur_points_pred - cur_points_orig).max())
    assert np.abs(t - transform).max() < 0.2
    assert np.abs(cur_points_pred - cur_points_orig).max() < 1


if __name__ == '__main__':
    # test_offset_grid_to_transform_params()
    test_offset_grid_to_transform()

