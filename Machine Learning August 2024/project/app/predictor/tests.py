from django.test import TestCase
import numpy as np
import os
import pickle
from ECEC.settings import PROJECT_ROOT
import pandas as pd
from predictor.content import columns

# Create your tests here.


class ModelTestCase(TestCase):
    def test_model_output_shape(self):
        sample_data = [
            [
                "M",
                np.int64(0),
                np.int64(1),
                np.int64(1),
                np.int64(112500),
                "Working",
                "Secondary / secondary special",
                "Married",
                "House / apartment",
                "Laborers",
                np.int64(3),
                np.int64(51),
                np.int64(15),
            ]
        ]
        df_to_test = pd.DataFrame(sample_data, columns=columns)
        loaded_model = pickle.load(
            open(os.path.join(PROJECT_ROOT, "best_model_pipeline.pkl"), "rb")
        )
        prediction = loaded_model.predict(df_to_test)

        self.assertEqual(
            df_to_test.shape,
            (1, 13),
            f"Model input shape is {df_to_test.shape}, expected {(1, 13)}",
        )

        self.assertEqual(
            prediction.shape,
            (1,),
            f"Model output shape is {prediction.shape}, expected {(1,)}",
        )
