#!/usr/bin/env python
# This file is the entrypoint for your submission.
# You can modify this file to include your code or directly call your functions/modules from here.
import random
from PIL import Image
from evaluator.airborne_detection import AirbornePredictor


class RandomPredictor(AirbornePredictor):
    """
    PARTICIPANT_TODO: You can name your implementation as you like. `RandomPredictor` is just an example.
    Below paths will be preloaded for you, you can read them as you like.
    """
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    """
    PARTICIPANT_TODO:
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def inference_setup(self):
        random.seed(42)
        pass

    """
    PARTICIPANT_TODO:
    During the evaluation all combinations for flight_id and flight_folder_path will be provided one by one.

    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def inference(self, flight_id):
        
        class_name = random.choice(["Airplane1", "Helicopter1"])
        track_id = random.randint(0, 3)
        bbox = [random.uniform(1300, 1500), random.uniform(1000, 1200)]
        bbox.append(bbox[0] + random.uniform(50, 100))
        bbox.append(bbox[1] + random.uniform(50, 100))

        i = random.randint(500, 900)
        j = random.randint(100, 200)
        
        for frame_image in self.get_all_frame_images(flight_id):
            # frame_image_path = self.get_frame_image_location(flight_id, frame_image)
            # img = Image.open(frame_image_path)
            # Do something...
            
            i -= 1
            if i > 0:
                continue

            j -= 1
            if j > 0:
                confidence = random.uniform(0.7, 1)
                # Please case your bbox & confidence values to json serializable class
                # Ex: np.float32 -> float, etc
                # bbox format is [x0, y0, x1, y1] (top, left, bottom, right)
                self.register_object_and_location(class_name, track_id, bbox, confidence, frame_image)


if __name__ == "__main__":
    submission = RandomPredictor()
    submission.run()
    print("Successfully generated predictions!")
