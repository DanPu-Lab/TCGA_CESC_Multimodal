import numpy as np
import h5py
from PIL import Image
from preprocess.process_svs import create_tissue_mask, dynamic_stride, extract


class FakeSlide:
    dimensions = (3000, 3000)
    level_count = 3
    level_dimensions = [(3000, 3000), (1500, 1500), (750, 750)]
    level_downsamples = [1., 2., 4.]
    properties = {"openslide.objective-power": "20", "openslide.mpp-x": ".5"}
    def get_thumbnail(self, size):
        return Image.new("RGB", size, (130, 60, 120))
    def get_best_level_for_downsample(self, value):
        return min(range(3), key=lambda level: abs(self.level_downsamples[level]-value))
    def read_region(self, xy, level, size):
        return Image.new("RGBA", size, (130, 60, 120, 255))


def test_hsv_mask_and_stride():
    assert create_tissue_mask(np.full((20, 20, 3), 255, dtype=np.uint8)).sum() == 0
    assert create_tissue_mask(np.tile([130, 60, 120], (20, 20, 1)).astype(np.uint8)).all()
    assert dynamic_stride(.8) == int(224*.4)
    assert dynamic_stride(.5) == int(224*.6)
    assert dynamic_stride(.1) == int(224*.8)


def test_wsi_extraction_records_physical_resolution_gamma_and_coordinates(tmp_path):
    out = tmp_path / "case.h5"
    count = extract(FakeSlide(), out, "cgci")
    assert count >= 100
    with h5py.File(out) as handle:
        assert handle["patches"].shape == (count, 224, 224, 3)
        assert handle["coordinates"].shape == (count, 4)
        assert handle.attrs["pixel_encoding"] == "rgb_uint8_gamma_0.9"
        expected = np.rint(255*np.power(np.array([130, 60, 120])/255, .9)).astype(np.uint8)
        np.testing.assert_array_equal(handle["patches"][0, 0, 0], expected)
