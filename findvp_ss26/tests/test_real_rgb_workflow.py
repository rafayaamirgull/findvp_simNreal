from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from PIL import Image

from findvp_ros.geometry import transform_matrix
from findvp_ros.runtime import DeploymentFault, file_digest
from findvp_ros.survey_node import survey_viewpoints, run_survey, load_survey_anchor
from findvp_ros.target_selection import polygon_mask, load_reference_mask, validate_target_provenance, main


def test_reference_annotation_roundtrip_and_changed_image_rejected(tmp_path):
    reference = tmp_path / "target.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (100, 80), "orange").save(reference)
    main(["--image", str(reference), "--output", str(mask_path),
          "--polygon", "20", "10", "80", "10", "80", "70", "20", "70"])
    mask = load_reference_mask(mask_path, reference)
    assert mask.shape == (80, 100) and mask[50, 50] and not mask[0, 0]
    with pytest.raises(FileExistsError):
        main(["--image", str(reference), "--output", str(mask_path)])
    Image.new("RGB", (100, 80), "blue").save(reference)
    with pytest.raises(ValueError, match="image changed"):
        load_reference_mask(mask_path, reference)


@pytest.mark.parametrize("vertices", [[], [[0, 0], [4, 4]],
    [[0, 0], [4, 4], [float('nan'), 0]], [[0, 0], [400, 0], [0, 40]]])
def test_invalid_polygons_fail(vertices):
    with pytest.raises(ValueError):
        polygon_mask((100, 80), vertices)


def test_localization_mask_provenance_is_checked(tmp_path):
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    document = {"target_sha256": file_digest(image), "target_mask_sha256": file_digest(mask)}
    validate_target_provenance(document, image, mask)
    with pytest.raises(ValueError, match="target mask"):
        validate_target_provenance(document, image, None)
    mask.write_bytes(b"changed")
    with pytest.raises(ValueError, match="target mask"):
        validate_target_provenance(document, image, mask)


def test_survey_moves_in_camera_coordinates_and_keeps_orientation():
    anchor = transform_matrix((.3, -.1, .4), (0, 0, np.sqrt(.5), np.sqrt(.5)))
    views = survey_viewpoints(anchor, .04)
    np.testing.assert_allclose(views[0].position_xyz, [.3, -.1, .4])
    np.testing.assert_allclose(views[1].position_xyz, [.3, -.06, .4])
    np.testing.assert_allclose(views[2].position_xyz, [.3, -.14, .4])
    assert all(v.orientation_xyzw == views[0].orientation_xyzw for v in views)
    for distance in (.001, .2, float("nan")):
        with pytest.raises(ValueError):
            survey_viewpoints(anchor, distance)


@pytest.mark.parametrize("fault", ["plan", "execute", "capture", None])
def test_survey_stops_at_first_failure(fault):
    calls = []
    class FakeNode:
        def plan(self, view):
            calls.append(("plan", view.index))
            if fault == "plan": raise DeploymentFault("planning failed")
            return view, {}, {}
        def execute(self, trajectory, scene):
            calls.append(("execute", trajectory.index))
            if fault == "execute": raise DeploymentFault("motion unconfirmed")
            return {}
        def capture(self, view):
            calls.append(("capture", view.index))
            if fault == "capture": raise DeploymentFault("capture failed")
            return {"world_from_camera": transform_matrix(view.position_xyz, view.orientation_xyzw).tolist()}
    report = {}
    def run():
        run_survey(FakeNode(), SimpleNamespace(execute=True, survey_baseline=.04),
                   np.eye(4), report, lambda: None)
    if fault:
        with pytest.raises(DeploymentFault): run()
        assert not any(index == 2 for _, index in calls)
        assert "status" not in report
    else:
        run()
        assert calls == [("capture", 0), ("plan", 1), ("execute", 1), ("capture", 1),
                         ("plan", 2), ("execute", 2), ("capture", 2)]
        assert report["status"] == "survey_complete"


def test_reference_mask_bypasses_central_colour_selection(tmp_path, monkeypatch):
    from sample.target_guided_similarity import prepare_target_image
    import sample.target_guided_similarity as module
    reference = tmp_path / "image.png"
    Image.new("RGB", (80, 60), "orange").save(reference)
    mask = polygon_mask((80, 60), [[10, 10], [70, 10], [70, 50], [10, 50]])
    def unexpected(*args): raise AssertionError("must use explicit whole-object mask")
    monkeypatch.setattr(module, "_central_object_region", unexpected)
    monkeypatch.setattr(module, "_extract_object_features", lambda *args: "features")
    monkeypatch.setattr(module, "_extract_sift_features", lambda *args: "sift")
    prepared = prepare_target_image(reference, extractor=object(), target_mask=mask)
    np.testing.assert_array_equal(prepared.region.mask, mask)
    with pytest.raises(ValueError, match="dimensions"):
        prepare_target_image(reference, extractor=object(), target_mask=np.ones((3, 4)))


def test_failed_localization_retains_masks_and_pair_diagnostics(tmp_path, monkeypatch):
    import sample.rgb_object_localization as module
    from sample.rgb_object_localization import RGBObservation
    observations = [RGBObservation(Image.new("RGB", (40, 40)),
                    np.array([[100,0,20],[0,100,20],[0,0,1.]]), np.eye(4), str(i)) for i in range(2)]
    mask = np.ones((40, 40), dtype=bool)
    monkeypatch.setattr(module, "detect_target_region", lambda *a, **kw: {
        "mask": mask, "bbox_xyxy": [0,0,40,40], "similarity": .9, "diagnostics": {}})
    monkeypatch.setattr(module, "extract_masked_sift", lambda *a: (np.empty((0,2)), None))
    with pytest.raises(ValueError, match="triangulation"):
        module.localize_rgb_observations(observations, object(), object(), diagnostics_dir=tmp_path)
    assert (tmp_path / "view_000_selected_mask.png").exists()
    diagnostic = json.loads((tmp_path / "matching.json").read_text())
    assert diagnostic["pairs"][0]["mutual_matches"] == 0
    assert "center_world_m" not in diagnostic


def test_triangulation_rejection_reports_actual_failure_without_lowering_limits():
    from sample.rgb_localization_geometry import triangulate_matches, TriangulationRejected
    k = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    first, second = np.eye(4), np.eye(4)
    second[0,3] = .08
    a = np.array([[300+i, 210+i] for i in range(12)], dtype=float)
    b = a + [-60., 20.]  # large inconsistent vertical disparity
    with pytest.raises(TriangulationRejected) as rejected:
        triangulate_matches(a, b, k, k, first, second)
    details = rejected.value.diagnostics
    assert details["required_inliers"] == 8
    assert details["max_reprojection_error_px"] == 2.
    assert details["reprojection_pass"] == 0
    assert details["median_finite_reprojection_error_px"] > 2.
