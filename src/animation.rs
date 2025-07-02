use splines::{Interpolate, Key};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Quaternion, Rad, VectorSpace};

use crate::{camera::PerspectiveCamera, PerspectiveProjection};

pub trait Lerp {
    fn lerp(&self, other: &Self, amount: f32) -> Self;
}

pub trait Sampler {
    type Sample;

    fn sample(&self, v: f32) -> Self::Sample;
}

pub struct Transition<T> {
    from: T,
    to: T,
    interp_fn: fn(f32) -> f32,
}
impl<T: Lerp + Clone> Transition<T> {
    pub fn new(from: T, to: T, interp_fn: fn(f32) -> f32) -> Self {
        Self {
            from,
            to,
            interp_fn,
        }
    }
}

impl<T: Lerp + Clone> Sampler for Transition<T> {
    type Sample = T;
    fn sample(&self, v: f32) -> Self::Sample {
        self.from.lerp(&self.to, (self.interp_fn)(v))
    }
}

pub struct TrackingShot {
    spline: splines::Spline<f32, PerspectiveCamera>,
}

impl TrackingShot {
    pub fn from_cameras<C>(cameras: Vec<C>) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();

        let last_two = cameras.iter().skip(cameras.len() - 2).take(2);
        let first_two = cameras.iter().take(2);
        let spline = splines::Spline::from_iter(
            last_two
                .chain(cameras.iter())
                .chain(first_two)
                .enumerate()
                .map(|(i, c)| {
                    let v = (i as f32 - 1.) / (cameras.len()) as f32;
                    Key::new(v, c.clone(), splines::Interpolation::CatmullRom)
                }),
        );

        Self { spline }
    }

    pub fn num_control_points(&self) -> usize {
        self.spline.len()
    }
}

impl Sampler for TrackingShot {
    type Sample = PerspectiveCamera;
    fn sample(&self, v: f32) -> Self::Sample {
        match self.spline.sample(v) {
            Some(p) => p,
            None => panic!("spline sample failed at {}", v),
        }
    }
}

impl Interpolate<f32> for PerspectiveCamera {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold {
            a
        } else {
            b
        }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        Self {
            position: Point3::from_vec(a.position.to_vec().lerp(b.position.to_vec(), t)),
            rotation: a.rotation.slerp(b.rotation, t),
            projection: a.projection.lerp(&b.projection, t),
        }
    }

    fn cosine(_t: f32, _a: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        // unroll quaternion rotations so that the animation always takes the shortest path
        // this is just a hack...
        let q_unrolled = unroll([x.1.rotation, a.1.rotation, b.1.rotation, y.1.rotation]);
        Self {
            position: Point3::from_vec(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.position.to_vec()),
                (a.0, a.1.position.to_vec()),
                (b.0, b.1.position.to_vec()),
                (y.0, y.1.position.to_vec()),
            )),
            rotation: Interpolate::cubic_hermite(
                t,
                (x.0, q_unrolled[0]),
                (a.0, q_unrolled[1]),
                (b.0, q_unrolled[2]),
                (y.0, q_unrolled[3]),
            )
            .normalize(),
            projection: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.projection),
                (a.0, a.1.projection),
                (b.0, b.1.projection),
                (y.0, y.1.projection),
            ),
        }
    }

    fn quadratic_bezier(_t: f32, _a: Self, _u: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }
}

impl Interpolate<f32> for PerspectiveProjection {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold {
            a
        } else {
            b
        }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        return a.lerp(&b, t);
    }

    fn cosine(_t: f32, _a: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        Self {
            fovx: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovx.0),
                (a.0, a.1.fovx.0),
                (b.0, b.1.fovx.0),
                (y.0, y.1.fovx.0),
            )),
            fovy: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovy.0),
                (a.0, a.1.fovy.0),
                (b.0, b.1.fovy.0),
                (y.0, y.1.fovy.0),
            )),
            znear: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.znear),
                (a.0, a.1.znear),
                (b.0, b.1.znear),
                (y.0, y.1.znear),
            ),
            zfar: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.zfar),
                (a.0, a.1.zfar),
                (b.0, b.1.zfar),
                (y.0, y.1.zfar),
            ),
            fov2view_ratio: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fov2view_ratio),
                (a.0, a.1.fov2view_ratio),
                (b.0, b.1.fov2view_ratio),
                (y.0, y.1.fov2view_ratio),
            ),
        }
    }

    fn quadratic_bezier(_t: f32, _a: Self, _u: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(_t: f32, _a: Self, _u: Self, _v: Self, _b: Self) -> Self {
        todo!()
    }
}

pub struct Animation<T> {
    duration: Duration,
    time_left: Duration,
    looping: bool,
    sampler: Box<dyn Sampler<Sample = T>>,
}

impl<T> Animation<T> {
    pub fn new(duration: Duration, looping: bool, sampler: Box<dyn Sampler<Sample = T>>) -> Self {
        Self {
            duration,
            time_left: duration,
            looping,
            sampler,
        }
    }

    pub fn done(&self) -> bool {
        if self.looping {
            false
        } else {
            self.time_left.is_zero()
        }
    }

    pub fn update(&mut self, dt: Duration) -> T {
        match self.time_left.checked_sub(dt) {
            Some(new_left) => {
                // set time left
                self.time_left = new_left;
            }
            None => {
                if self.looping {
                    // For looping: calculate how much time overflows and wrap around
                    let overflow = dt.saturating_sub(self.time_left);
                    self.time_left = self.duration.saturating_sub(overflow);
                } else {
                    self.time_left = Duration::ZERO;
                }
            }
        }
        return self.sampler.sample(self.progress());
    }

    pub fn progress(&self) -> f32 {
        if self.duration.is_zero() {
            return 1.0; // Animation is complete if duration is zero
        }
        return (1. - self.time_left.as_secs_f32() / self.duration.as_secs_f32()).clamp(0.0, 1.0);
    }

    pub fn set_progress(&mut self, v: f32) {
        let clamped_v = v.clamp(0.0, 1.0);
        self.time_left = self.duration.mul_f32(1. - clamped_v);
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn set_duration(&mut self, duration: Duration) {
        let progress = self.progress();
        self.duration = duration;
        self.set_progress(progress);
    }
}

/// unroll quaternion rotations so that the animation always takes the shortest path
fn unroll(rot: [Quaternion<f32>; 4]) -> [Quaternion<f32>; 4] {
    let mut rot = rot;
    if rot[0].s < 0. {
        rot[0] = -rot[0];
    }
    for i in 1..4 {
        if rot[i].dot(rot[i - 1]) < 0. {
            rot[i] = -rot[i];
        }
    }
    return rot;
}

pub struct NavigationSequence {
    cameras: Vec<PerspectiveCamera>,
    seconds_per_camera: f32,
}

impl NavigationSequence {
    pub fn new<C>(cameras: Vec<C>, seconds_per_camera: f32) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();

        Self {
            cameras,
            seconds_per_camera,
        }
    }
}

// Smooth step function for interpolation
fn smoothstep_local(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

impl Sampler for NavigationSequence {
    type Sample = PerspectiveCamera;

    fn sample(&self, v: f32) -> Self::Sample {
        if self.cameras.is_empty() {
            panic!("NavigationSequence has no cameras");
        }

        if self.cameras.len() == 1 {
            return self.cameras[0];
        }

        // v ranges from 0.0 to 1.0 over the entire animation
        // Map it to camera segments
        let total_segments = (self.cameras.len() - 1) as f32;
        let scaled_progress = v * total_segments;

        // Clamp to valid range
        let scaled_progress = scaled_progress.max(0.0).min(total_segments);

        // Find which camera segment we're in
        let segment_index = (scaled_progress.floor() as usize).min(self.cameras.len() - 2);
        let segment_progress = scaled_progress - segment_index as f32;

        // Interpolate between the two cameras in this segment
        let from_camera = self.cameras[segment_index];
        let to_camera = self.cameras[segment_index + 1];

        // Use smooth interpolation within each segment
        let smooth_progress = smoothstep_local(segment_progress);
        from_camera.lerp(&to_camera, smooth_progress)
    }
}

pub struct AdaptiveNavigationSequence {
    cameras: Vec<PerspectiveCamera>,
    forward_cameras: usize,
    pause_cameras: usize,
    forward_seconds_per_camera: f32,
    pause_seconds_per_camera: f32,
    return_seconds_per_camera: f32,
}

impl AdaptiveNavigationSequence {
    pub fn new<C>(
        cameras: Vec<C>,
        forward_cameras: usize,
        pause_cameras: usize,
        forward_seconds_per_camera: f32,
        pause_seconds_per_camera: f32,
        return_seconds_per_camera: f32,
    ) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();

        Self {
            cameras,
            forward_cameras,
            pause_cameras,
            forward_seconds_per_camera,
            pause_seconds_per_camera,
            return_seconds_per_camera,
        }
    }
}

impl Sampler for AdaptiveNavigationSequence {
    type Sample = PerspectiveCamera;

    fn sample(&self, v: f32) -> Self::Sample {
        if self.cameras.is_empty() {
            panic!("AdaptiveNavigationSequence has no cameras");
        }

        if self.cameras.len() == 1 {
            return self.cameras[0];
        }

        // Clamp v to valid range to prevent out-of-bounds issues
        let clamped_v = v.clamp(0.0, 1.0);

        // Calculate total duration for each phase with bounds checking
        let forward_duration = self.forward_cameras as f32 * self.forward_seconds_per_camera;
        let pause_duration = self.pause_cameras as f32 * self.pause_seconds_per_camera;

        // Ensure return cameras calculation doesn't underflow
        let return_cameras = if self.forward_cameras + self.pause_cameras <= self.cameras.len() {
            self.cameras.len() - self.forward_cameras - self.pause_cameras
        } else {
            0 // Fallback if camera counts are inconsistent
        };

        let return_duration = return_cameras as f32 * self.return_seconds_per_camera;
        let total_duration = (forward_duration + pause_duration + return_duration).max(0.001); // Avoid division by zero

        // Convert global v (0.0 to 1.0) to absolute time
        let absolute_time = clamped_v * total_duration;

        // Determine which phase we're in and calculate local time and camera indices
        let (camera_index, local_progress) = if absolute_time <= forward_duration {
            // FORWARD PHASE
            let local_time = absolute_time;
            let camera_index = (local_time / self.forward_seconds_per_camera).floor() as usize;
            let camera_index = camera_index.min(self.forward_cameras - 1);
            let local_progress =
                (local_time % self.forward_seconds_per_camera) / self.forward_seconds_per_camera;
            (camera_index, local_progress)
        } else if absolute_time <= forward_duration + pause_duration {
            // PAUSE PHASE
            let local_time = absolute_time - forward_duration;
            let camera_index = self.forward_cameras
                + (local_time / self.pause_seconds_per_camera).floor() as usize;
            let camera_index = camera_index.min(self.forward_cameras + self.pause_cameras - 1);
            let local_progress =
                (local_time % self.pause_seconds_per_camera) / self.pause_seconds_per_camera;
            (camera_index, local_progress)
        } else {
            // RETURN PHASE
            let local_time = absolute_time - forward_duration - pause_duration;
            let camera_index = self.forward_cameras
                + self.pause_cameras
                + (local_time / self.return_seconds_per_camera).floor() as usize;
            let camera_index = camera_index.min(self.cameras.len() - 1);
            let local_progress =
                (local_time % self.return_seconds_per_camera) / self.return_seconds_per_camera;
            (camera_index, local_progress)
        };

        // Handle edge case where we're at the very end
        if camera_index >= self.cameras.len() - 1 {
            return self.cameras[self.cameras.len() - 1];
        }

        // Interpolate between current camera and next camera
        let from_camera = &self.cameras[camera_index];
        let to_camera = &self.cameras[camera_index + 1];

        // Use smooth interpolation within each segment
        let smooth_progress = smoothstep_local(local_progress);
        from_camera.lerp(&to_camera, smooth_progress)
    }
}
