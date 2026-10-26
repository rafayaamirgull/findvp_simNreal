# Viewpoint Sampling

## Fixture versus production input

`sample_fibonacci_viewpoints(...)` creates the deterministic 100-position
Level 6 fixture. The reachability filter itself does not require Fibonacci
sampling and accepts externally generated poses. See
[Using External Viewpoints](Using-External-Viewpoints).

## Defaults

| Parameter | Default |
| --- | ---: |
| Count | 100 |
| Requested radius | `0.30` to `0.50 m` |
| Elevation | `12` to `82 degrees` |
| Roll at sampling time | `0 degrees` |

Roll alternatives are expanded later by the reachability pipeline.

## Equal-area low-discrepancy shell

For sample `i` in a set of `N` samples:

```text
area_fraction = (i + 0.5) / N

sin(elevation_i) = sin(elevation_min)
                 + area_fraction
                 * (sin(elevation_max) - sin(elevation_min))

golden_angle = pi * (3 - sqrt(5))
azimuth_i = (i * golden_angle) mod (2*pi)

radius_fraction_i = radical_inverse_base_2(i + 1)
radius_i = radius_min
         + radius_fraction_i * (radius_max - radius_min)
```

When minimum and maximum radii are equal, the radius fraction is fixed at
`0.5`, which still produces the requested radius.

Sampling elevation uniformly in its sine is uniform in spherical surface area
over the requested elevation band. Golden-angle azimuths avoid aligned rings,
and the base-two radical inverse decorrelates radius from the other dimensions.
The result is deterministic and contains no random-number generator.

## Camera orientation

Each sampled optical center is oriented toward the object center:

```text
forward = normalize(object_center - camera_position)
right   = normalize(cross(forward, world_up))
down    = normalize(cross(forward, right))
R_world_camera = [right, down, forward]
```

This implements the repository optical convention:

- `+Z` points forward;
- `+X` points right;
- `-Y` points up.

If `forward` is nearly parallel to world `+Z` (`abs(dot) > 0.98`), world `+Y`
is used as the temporary up vector to avoid a singular cross product.

Optical-axis roll rotates the right/down basis but preserves position and
forward direction.

## Camera-frustum adjustment

The CLI fixture prevents the requested shell from placing the camera too close
to frame the complete workpiece bounding sphere.

```text
object_bounding_radius = 0.5 * norm(workpiece_extent)

usable_half_fov = 0.5 * min(vertical_fov, horizontal_fov)
                * (1 - 0.10)

framing_min_radius = object_bounding_radius / sin(usable_half_fov)
effective_radius_min = max(requested_radius_min, framing_min_radius)
```

The smaller horizontal/vertical FOV is used with a 10% angular margin. If the
required minimum exceeds `--view-radius-max`, the CLI stops with an explanatory
error instead of sampling a clipped object.

For the default sample object in the validated fixture:

| Quantity | Observed value |
| --- | ---: |
| Workpiece extent | `[0.191364, 0.108374, 0.219005] m` |
| Bounding radius | `0.155184 m` |
| Vertical FOV | `63.0374 degrees` |
| Horizontal FOV | `78.5434 degrees` |
| Requested minimum radius | `0.300000 m` |
| Effective minimum radius | `0.326624 m` |
| Actual sampled elevation range | `12.2292` to `80.5246 degrees` |

Direct calls to `filter_reachable_viewpoints(...)` do not perform this fixture
framing adjustment. An upstream viewpoint generator owns its own visibility
and framing policy.

## Input validation

The sampler rejects:

- an object center that is not exactly three finite values;
- non-positive or Boolean counts;
- non-finite radii or elevation bounds;
- `radius_min <= 0` or `radius_max < radius_min`;
- elevation bands outside `-90 < min <= max < 90 degrees`.

Next: [Level 6 Reachability Pipeline](Level-6-Reachability-Pipeline) or
[Using External Viewpoints](Using-External-Viewpoints).
