import unittest

from utils as utils


class TestUtils(unittest.TestCase):
    def test_find_point_location_Old_Astoria(self):
        ex_latitude = 40.771042875660164
        ex_longitude = -73.92217124150466
        geom_point = utils.find_point_location(ex_latitude, ex_longitude)
        self.assertIsNotNone(geom_point)
        self.assertEqual(geom_point.iloc[0]['zone'], "Old Astoria")
        self.assertEqual(geom_point.iloc[0]['LocationID'], 179)

    def test_find_point_location_Bushwick_North(self):
        ex_latitude, ex_longitude = 40.70319418467109, -73.9238477573261
        geom_point = utils.find_point_location(ex_latitude, ex_longitude)
        self.assertIsNotNone(geom_point)
        self.assertEqual(geom_point.iloc[0]['zone'], "Bushwick North")
        self.assertEqual(geom_point.iloc[0]['LocationID'], 36)

    def test_get_taxi_zones(self):
        pickup_location_id, dropoff_location_id = utils.get_taxi_zones(
            40.771042875660164, 40.70319418467109, -73.92217124150466, -73.9238477573261
        )
        self.assertEqual(pickup_location_id, 179)
        self.assertEqual(dropoff_location_id, 36)

if __name__ == "__main__":
    unittest.main()
