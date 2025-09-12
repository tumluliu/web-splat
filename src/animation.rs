use splines::{Interpolate, Key};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Quaternion, Rad, Vector3, VectorSpace};

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

/// Elegant object approach animation that creates a smooth path from current position to optimal object viewing position
/// Uses scene cameras as waypoints to ensure natural, forward-looking camera movement
pub struct ElegantObjectApproach {
    cameras: Vec<PerspectiveCamera>,
    total_duration: Duration,
    phase_durations: Vec<f32>, // Duration for each camera segment
}

impl ElegantObjectApproach {
    /// Create an elegant approach animation from current camera to optimal object viewing position
    /// Uses scene cameras to create natural waypoints and smooth transitions
    pub fn new(
        current_camera: PerspectiveCamera,
        target_object_center: Point3<f32>,
        optimal_camera: PerspectiveCamera,
        scene_cameras: &[crate::scene::SceneCamera],
        ground_up: Vector3<f32>,
        total_duration: Duration,
    ) -> Self {
        use cgmath::MetricSpace;

        log::info!("🎬 Creating elegant object approach animation");
        log::info!(
            "  Current position: ({:.3}, {:.3}, {:.3})",
            current_camera.position.x,
            current_camera.position.y,
            current_camera.position.z
        );
        log::info!(
            "  Target position: ({:.3}, {:.3}, {:.3})",
            optimal_camera.position.x,
            optimal_camera.position.y,
            optimal_camera.position.z
        );
        log::info!(
            "  Object center: ({:.3}, {:.3}, {:.3})",
            target_object_center.x,
            target_object_center.y,
            target_object_center.z
        );

        let mut animation_cameras = Vec::new();
        let mut phase_durations = Vec::new();

        // Always start with current camera
        animation_cameras.push(current_camera);

        // Find intermediate scene cameras that create a natural path
        let intermediate_cameras = Self::find_intermediate_cameras(
            current_camera.position,
            optimal_camera.position,
            target_object_center,
            scene_cameras,
            ground_up,
        );

        log::info!(
            "  Found {} intermediate cameras for smooth path",
            intermediate_cameras.len()
        );

        // If no intermediate cameras found, create a simple direct path with multiple steps
        if intermediate_cameras.is_empty() {
            log::info!("  No suitable intermediate cameras found, creating direct path with multiple steps");

            // Create 2-3 intermediate positions along the direct path for smooth motion
            let path_vector = optimal_camera.position - current_camera.position;
            let path_length = path_vector.magnitude();

            if path_length > 5.0 {
                // Only add intermediates for longer distances
                let num_steps = (path_length / 8.0).clamp(2.0, 3.0) as usize;
                for step in 1..num_steps {
                    let progress = step as f32 / num_steps as f32;
                    let intermediate_pos =
                        Point3::from_vec(current_camera.position.to_vec() + path_vector * progress);

                    // Create camera looking towards target with gradual orientation change
                    let to_target = (target_object_center - intermediate_pos).normalize();
                    let current_forward = {
                        let rot: cgmath::Matrix3<f32> = current_camera.rotation.into();
                        -Vector3::new(rot.z.x, rot.z.y, rot.z.z)
                    };

                    // Blend between current direction and target direction
                    let blend_factor = progress * progress; // Ease into target focus
                    let blended_forward = (current_forward * (1.0 - blend_factor)
                        + to_target * blend_factor)
                        .normalize();

                    let rotation = crate::camera::create_look_at_rotation(
                        intermediate_pos,
                        Point3::from_vec(intermediate_pos.to_vec() + blended_forward),
                        ground_up,
                    );

                    let intermediate_camera = PerspectiveCamera::new(
                        intermediate_pos,
                        rotation,
                        current_camera.projection, // Use same projection
                    );

                    animation_cameras.push(intermediate_camera);
                    log::info!(
                        "  Added synthetic intermediate camera at ({:.3}, {:.3}, {:.3})",
                        intermediate_pos.x,
                        intermediate_pos.y,
                        intermediate_pos.z
                    );
                }
            }
        } else {
            // Add intermediate cameras with forward-looking adjustments
            for (i, scene_camera) in intermediate_cameras.iter().enumerate() {
                let camera: PerspectiveCamera = scene_camera.clone().into();

                // Adjust camera to look towards the object while maintaining smooth motion
                let adjusted_camera = Self::adjust_camera_for_forward_looking(
                    camera,
                    target_object_center,
                    ground_up,
                    i as f32 / intermediate_cameras.len() as f32, // Progress through path
                );

                animation_cameras.push(adjusted_camera);
            }
        }

        // Always end with the optimal camera
        animation_cameras.push(optimal_camera);

        // Calculate phase durations based on distance and complexity
        let total_cameras = animation_cameras.len();
        if total_cameras > 1 {
            let base_duration = total_duration.as_secs_f32() / (total_cameras - 1) as f32;

            for i in 0..total_cameras - 1 {
                let distance = animation_cameras[i]
                    .position
                    .distance(animation_cameras[i + 1].position);
                // Longer durations for longer distances, shorter for close cameras
                let duration_multiplier = (distance / 10.0).clamp(0.5, 2.0);
                phase_durations.push(base_duration * duration_multiplier);
            }

            // Normalize durations to match total duration
            let total_phase_duration: f32 = phase_durations.iter().sum();
            let scale_factor = total_duration.as_secs_f32() / total_phase_duration;
            for duration in &mut phase_durations {
                *duration *= scale_factor;
            }
        }

        log::info!(
            "  Created animation with {} cameras over {:.2}s",
            animation_cameras.len(),
            total_duration.as_secs_f32()
        );

        Self {
            cameras: animation_cameras,
            total_duration,
            phase_durations,
        }
    }

    /// Find intermediate scene cameras that create a natural path from start to end
    fn find_intermediate_cameras(
        start_pos: Point3<f32>,
        end_pos: Point3<f32>,
        target_center: Point3<f32>,
        scene_cameras: &[crate::scene::SceneCamera],
        ground_up: Vector3<f32>,
    ) -> Vec<crate::scene::SceneCamera> {
        use cgmath::{InnerSpace, MetricSpace};

        let mut candidates: Vec<(crate::scene::SceneCamera, f32, f32)> = Vec::new();
        let path_vector = end_pos - start_pos;
        let path_length = path_vector.magnitude();

        if path_length < 0.1 {
            return Vec::new(); // Too short to need intermediates
        }

        let path_direction = path_vector.normalize();

        // Find cameras that are:
        // 1. Reasonably close to the direct path
        // 2. Have good visibility of the target object
        // 3. Are positioned to create smooth forward motion
        for scene_camera in scene_cameras {
            let cam_pos = Point3::from(scene_camera.position);
            let cam: PerspectiveCamera = scene_camera.clone().into();

            // Distance from camera to the direct path line
            let to_camera = cam_pos - start_pos;
            let projection_length = to_camera.dot(path_direction);
            let projection_point = start_pos + path_direction * projection_length;
            let distance_to_path = cam_pos.distance(projection_point);

            // Progress along the path (0.0 = start, 1.0 = end)
            let path_progress = (projection_length / path_length).clamp(0.0, 1.0);

            // Skip cameras too close to start/end or too far from path
            if path_progress < 0.1 || path_progress > 0.9 || distance_to_path > path_length * 0.5 {
                continue;
            }

            // Check visibility to target object
            let to_target = (target_center - cam_pos).normalize();
            let cam_rotation: cgmath::Matrix3<f32> = cam.rotation.into();
            let cam_forward = -Vector3::new(cam_rotation.z.x, cam_rotation.z.y, cam_rotation.z.z);
            let target_alignment = cam_forward.dot(to_target);

            // Score this camera based on multiple factors
            let path_score = 1.0 - (distance_to_path / (path_length * 0.5)).min(1.0);
            let visibility_score = (target_alignment + 1.0) / 2.0;
            let position_score = (0.5 - (path_progress - 0.5).abs()) * 2.0; // Prefer middle positions

            let total_score = path_score * 0.4 + visibility_score * 0.4 + position_score * 0.2;

            candidates.push((scene_camera.clone(), total_score, path_progress));
        }

        // Sort by score and select the best ones
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take up to 3-5 intermediate cameras, ensuring good spacing
        let mut selected: Vec<(crate::scene::SceneCamera, f32, f32)> = Vec::new();
        let max_intermediates = (path_length / 5.0).clamp(2.0, 5.0) as usize;

        for (camera, score, progress) in candidates.iter().take(max_intermediates * 2) {
            // Ensure good spacing between selected cameras
            let mut too_close = false;
            for (_, _, existing_progress) in &selected {
                if (progress - existing_progress).abs() < 0.2 {
                    too_close = true;
                    break;
                }
            }

            if !too_close && selected.len() < max_intermediates {
                selected.push((camera.clone(), *score, *progress));
            }
        }

        // Sort selected cameras by path progress
        selected.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        log::info!(
            "  Selected {} intermediate cameras with scores:",
            selected.len()
        );
        for (i, (_, score, progress)) in selected.iter().enumerate() {
            log::info!(
                "    Camera {}: score={:.3}, progress={:.3}",
                i,
                score,
                progress
            );
        }

        selected.into_iter().map(|(camera, _, _)| camera).collect()
    }

    /// Adjust camera to look towards object while maintaining smooth motion
    fn adjust_camera_for_forward_looking(
        mut camera: PerspectiveCamera,
        target_center: Point3<f32>,
        ground_up: Vector3<f32>,
        progress: f32,
    ) -> PerspectiveCamera {
        use cgmath::InnerSpace;

        // Calculate direction to target
        let to_target = (target_center - camera.position).normalize();

        // Get current camera forward direction
        let cam_rotation: cgmath::Matrix3<f32> = camera.rotation.into();
        let current_forward = -Vector3::new(cam_rotation.z.x, cam_rotation.z.y, cam_rotation.z.z);

        // Gradually blend from current direction to target direction
        // Early in the path, maintain more of the original direction
        // Later in the path, focus more on the target
        let blend_factor = smoothstep_local(progress * progress); // Ease into target focus
        let blended_forward =
            (current_forward * (1.0 - blend_factor) + to_target * blend_factor).normalize();

        // Create new rotation that looks in the blended direction
        let new_rotation = crate::camera::create_look_at_rotation(
            camera.position,
            Point3::from_vec(camera.position.to_vec() + blended_forward),
            ground_up,
        );

        camera.rotation = new_rotation;
        camera
    }
}

impl Sampler for ElegantObjectApproach {
    type Sample = PerspectiveCamera;

    fn sample(&self, v: f32) -> Self::Sample {
        if self.cameras.is_empty() {
            panic!("ElegantObjectApproach has no cameras");
        }

        if self.cameras.len() == 1 {
            return self.cameras[0];
        }

        let v = v.clamp(0.0, 1.0);

        // Find which phase we're in based on cumulative duration
        let mut cumulative_time = 0.0;
        let total_time = self.total_duration.as_secs_f32();
        let current_time = v * total_time;

        for (i, &phase_duration) in self.phase_durations.iter().enumerate() {
            if current_time <= cumulative_time + phase_duration
                || i == self.phase_durations.len() - 1
            {
                // We're in this phase
                let phase_progress = if phase_duration > 0.0 {
                    ((current_time - cumulative_time) / phase_duration).clamp(0.0, 1.0)
                } else {
                    1.0
                };

                let from_camera = self.cameras[i];
                let to_camera = self.cameras[i + 1];

                // Use smooth interpolation with easing
                let smooth_progress = smoothstep_local(phase_progress);
                return from_camera.lerp(&to_camera, smooth_progress);
            }
            cumulative_time += phase_duration;
        }

        // Fallback to last camera
        self.cameras[self.cameras.len() - 1]
    }
}

/// Closed-loop camera animation using scene camera poses
/// Cycles through scene cameras in sequence, creating a smooth loop
pub struct ClosedLoopSequence {
    cameras: Vec<PerspectiveCamera>,
    seconds_per_camera: f32,
    looping: bool,
}

impl ClosedLoopSequence {
    /// Create a new closed loop sequence from scene cameras
    /// The sequence will automatically close the loop by connecting the last camera back to the first
    pub fn from_scene_cameras<C>(cameras: Vec<C>, seconds_per_camera: f32) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let mut cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();

        // Ensure we have at least one camera
        if cameras.is_empty() {
            panic!("ClosedLoopSequence requires at least one camera");
        }

        // Add the first camera at the end to create a smooth loop
        if cameras.len() > 1 {
            cameras.push(cameras[0]);
        }

        Self {
            cameras,
            seconds_per_camera,
            looping: true,
        }
    }

    /// Create from a subset of scene cameras (e.g., every Nth camera for faster animation)
    pub fn from_scene_cameras_subset<C>(
        cameras: Vec<C>,
        step: usize,
        seconds_per_camera: f32,
    ) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let all_cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();
        let subset_cameras: Vec<PerspectiveCamera> = all_cameras
            .into_iter()
            .enumerate()
            .filter_map(|(i, camera)| if i % step == 0 { Some(camera) } else { None })
            .collect();

        Self::from_scene_cameras(subset_cameras, seconds_per_camera)
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }
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

impl Sampler for ClosedLoopSequence {
    type Sample = PerspectiveCamera;

    fn sample(&self, v: f32) -> Self::Sample {
        if self.cameras.is_empty() {
            panic!("ClosedLoopSequence has no cameras");
        }

        if self.cameras.len() == 1 {
            return self.cameras[0];
        }

        // For looping animation, v wraps around from 0.0 to 1.0 continuously
        // Map v to camera segments
        let total_segments = (self.cameras.len() - 1) as f32;
        let scaled_progress = (v % 1.0) * total_segments;

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
