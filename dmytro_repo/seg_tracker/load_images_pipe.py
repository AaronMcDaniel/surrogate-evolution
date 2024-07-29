import numpy as np
import cv2
import mpipe


def load_img(fn):
    try:
        return fn, np.fromfile(fn, dtype=np.uint8)
    except FileNotFoundError:
        # print('file not found:', fn)
        return fn, None


def decode_img(args):
    fn, img_data = args
    if img_data is None:
        return fn, None

    try:
        img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
        return fn, img
    except cv2.error as e:
        # print('file not found:', fn)
        return fn, None


def limited_pipe(pipe, requests, max_queue=32):
    max_queue = min(max_queue, len(requests)-2)
    for i in range(max_queue):
        pipe.put(requests[i])

    for i in range(len(requests)):
        yield pipe.get()
        if i+max_queue < len(requests):
            pipe.put(requests[i+max_queue])


def test_limited_pipe():
    stage = mpipe.OrderedStage(lambda i: i, 1)
    pipe = mpipe.Pipeline(stage)

    requests = list(range(4096))
    res = [i for i in limited_pipe(pipe, requests, 16)]

    assert len(res) == len(requests)
    for request, result in zip(requests, res):
        if request != result:
            print(requests, result)
        assert request == result


if __name__ == '__main__':
    test_limited_pipe()
