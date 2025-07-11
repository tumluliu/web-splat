use std::{
    io::{Read, Seek},
    path::{Path, PathBuf},
    sync::Arc,
};

use image::Pixel;
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};
use renderer::Display;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::{util::DeviceExt, Backends, Extent3d};

use cgmath::{Deg, EuclideanSpace, InnerSpace, Point3, Quaternion, Rotation, UlpsEq, Vector2, Vector3};
use egui::FullOutput;
use num_traits::One;

use utils::key_to_num;
#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

mod animation;
mod ui;
pub use animation::{Animation, NavigationSequence, Sampler, TrackingShot, Transition};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pointcloud;
pub use pointcloud::PointCloud;
mod chat;
pub use chat::{ChatState, McpResponse, SceneObject, ScenePath};

pub mod io;

mod renderer;
pub use renderer::{GaussianRenderer, SplattingArgs};

mod scene;
use crate::utils::GPUStopwatch;

pub use self::scene::{Scene, SceneCamera, Split};

pub mod gpu_rs;
mod ui_renderer;
mod uniform;
mod utils;
mod highlighting;
pub use highlighting::HighlightRenderer;

pub struct RenderConfig {
    pub no_vsync: bool,
    pub skybox: Option<PathBuf>,
    pub hdr: bool,
    pub mcp_server_url: String,
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new_instance() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        return WGPUContext::new(&instance, None).await;
    }

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();
        log::info!("using {}", adapter.get_info().name);

        #[cfg(target_arch = "wasm32")]
        let required_features = wgpu::Features::default();
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    #[cfg(not(target_arch = "wasm32"))]
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: adapter_limits
                            .max_storage_buffer_binding_size,
                        max_storage_buffers_per_shader_stage: 12,
                        max_compute_workgroup_storage_size: 1 << 15,
                        ..adapter_limits
                    },

                    #[cfg(target_arch = "wasm32")]
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_storage_size: 1 << 15,
                        ..adapter_limits
                    },
                    label: None,
                    memory_hints: wgpu::MemoryHints::Performance
                },
                None,
            )
            .await
            .unwrap();

        Self {
            device,
            queue,
            adapter,
        }
    }
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    pc: PointCloud,
    pointcloud_file_path: Option<PathBuf>,
    renderer: GaussianRenderer,
    animation: Option<(Animation<PerspectiveCamera>, bool)>,
    controller: CameraController,
    scene: Option<Scene>,
    scene_file_path: Option<PathBuf>,
    current_view: Option<usize>,
    ui_renderer: ui_renderer::EguiWGPU,
    fps: f32,
    ui_visible: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,
    highlight_renderer: HighlightRenderer,

    splatting_args: SplattingArgs,

    saved_cameras: Vec<SceneCamera>,
    #[cfg(feature = "video")]
    cameras_save_path: String,
    stopwatch: Option<GPUStopwatch>,
    chat_state: ChatState,
    pending_chat_responses: Vec<(String, McpResponse)>,
    
    // Ground up direction derived from scene cameras for stable camera positioning
    ground_up_direction: Vector3<f32>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(
        window: Window,
        pc_file: R,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let window = Arc::new(window);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        log::info!("device: {:?}", wgpu_context.adapter.get_info().name);

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();

        let render_format = if render_config.hdr {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);

        let pc_raw = io::GenericGaussianPointCloud::load(pc_file)?;
        let pc = PointCloud::new(&device, pc_raw)?;
        log::info!("loaded point cloud with {:} points", pc.num_points());

        let renderer =
            GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed())
                .await;

        let aabb = pc.bbox();
        let aspect = size.width as f32 / size.height as f32;
        let view_camera = PerspectiveCamera::new(
            aabb.center() - Vector3::new(1., 1., 1.) * aabb.radius() * 0.5,
            Quaternion::one(),
            PerspectiveProjection::new(
                Vector2::new(size.width, size.height),
                Vector2::new(Deg(45.), Deg(45. / aspect)),
                0.01,
                1000.,
            ),
        );

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = pc.center();
        // controller.up = pc.up;
        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let display = Display::new(
            device,
            render_format,
            surface_format.remove_srgb_suffix(),
            size.width,
            size.height,
        );

        let highlight_renderer = HighlightRenderer::new(device, surface_format.remove_srgb_suffix());

        let stopwatch = if cfg!(not(target_arch = "wasm32")) {
            Some(GPUStopwatch::new(device, Some(3)))
        } else {
            None
        };

        Ok(Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            splatting_args: SplattingArgs {
                camera: view_camera,
                viewport: Vector2::new(size.width, size.height),
                gaussian_scaling: 1.,
                max_sh_deg: pc.sh_deg(),
                show_env_map: false,
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::ZERO,
                scene_center: None,
                scene_extend: None,
                background_color: wgpu::Color::BLACK,
                resolution: Vector2::new(size.width, size.height),
            },
            pc,
            // camera: view_camera,
            controller,
            ui_renderer,
            fps: 0.,
            #[cfg(not(target_arch = "wasm32"))]
            history: RingBuffer::new(512),
            ui_visible: true,
            display,
            highlight_renderer,
            saved_cameras: Vec::new(),
            #[cfg(feature = "video")]
            cameras_save_path: "cameras_saved.json".to_string(),
            animation: None,
            scene: None,
            current_view: None,
            pointcloud_file_path: None,
            scene_file_path: None,

            stopwatch,
            chat_state: ChatState {
                mcp_server_url: render_config.mcp_server_url.clone(),
                ..ChatState::default()
            },
            pending_chat_responses: Vec::new(),
            
            // Ground up direction derived from scene cameras for stable camera positioning
            ground_up_direction: Vector3::new(0.0, 1.0, 0.0),
        })
    }

    fn reload(&mut self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.pointcloud_file_path {
            log::info!("reloading volume from {:?}", file_path);
            let file = std::fs::File::open(file_path)?;
            let pc_raw = io::GenericGaussianPointCloud::load(file)?;
            self.pc = PointCloud::new(&self.wgpu_context.device, pc_raw)?;
        } else {
            return Err(anyhow::anyhow!("no pointcloud file path present"));
        }
        if let Some(scene_path) = &self.scene_file_path {
            log::info!("reloading scene from {:?}", scene_path);
            let file = std::fs::File::open(scene_path)?;

            self.set_scene(Scene::from_json(file)?);
        }
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
            self.display
                .resize(&self.wgpu_context.device, new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.splatting_args.viewport = Vector2::new(new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        self.ui_renderer.begin_frame(&self.window);
        let (request_redraw, chat_message) = ui::ui(self);

        // Handle chat message if present
        if let Some(message) = chat_message {
            log::info!("Chat message received: {}", message);
            self.handle_chat_message(message);
            log::info!("Chat message handling complete");
        }

        let shapes = self.ui_renderer.end_frame(&self.window);

        return (request_redraw, shapes);
    }

    fn handle_chat_message(&mut self, message: String) {
        log::info!("handle_chat_message called with: {}", message);
        
        // Add user message and set sending state
        self.chat_state.add_message(message.clone(), true);
        self.chat_state.is_sending = true;
        
        log::info!("User message added, making HTTP request to server");
        
        // Make HTTP request to the actual server
        let server_url = self.chat_state.mcp_server_url.clone();
        
        log::info!("Sending HTTP request to: {}/query with message: {}", server_url, message);
        
        // Spawn async task to make HTTP request
        #[cfg(not(target_arch = "wasm32"))]
        {
            let msg_clone = message.clone();
            let camera_pos = self.splatting_args.camera.position;
            let current_location = [camera_pos.x, camera_pos.y, camera_pos.z];
            let rt = tokio::runtime::Runtime::new().unwrap();
            match rt.block_on(crate::chat::send_chat_message(msg_clone, &server_url, current_location)) {
                Ok(response) => {
                    log::info!("Received HTTP response successfully");
                    self.pending_chat_responses.push((message, response));
                }
                Err(e) => {
                    log::warn!("HTTP request failed: {}, falling back to mock response", e);
                    let mock_response = ui::create_mock_response(&message);
                    self.pending_chat_responses.push((message, mock_response));
                }
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        {
            log::info!("WASM build: making HTTP request");
            
            // Store a reference to the pending responses that we can update from the async closure
            // We'll use a polling approach - the async task will store the response in a static location
            // and the main update loop will check for it
            
            let msg_clone = message.clone();
            let server_url_clone = server_url.clone();
            let camera_pos = self.splatting_args.camera.position;
            let current_location = [camera_pos.x, camera_pos.y, camera_pos.z];
            
            // Create a unique identifier for this request
            let request_id = format!("{}_{}", message.len(), chrono::Utc::now().timestamp_millis());
            
            // Store the request ID so we can match it with the response later
            self.chat_state.pending_request_id = Some(request_id.clone());
            
            wasm_bindgen_futures::spawn_local(async move {
                log::info!("Starting WASM HTTP request...");
                match crate::chat::send_chat_message(msg_clone.clone(), &server_url_clone, current_location).await {
                    Ok(response) => {
                        log::info!("WASM HTTP request successful!");
                        // Store the response in a global location that the main thread can access
                        crate::chat::store_async_response(request_id, msg_clone, response);
                    }
                    Err(e) => {
                        log::warn!("WASM HTTP request failed: {}, using fallback", e);
                        // Store a fallback response
                        let fallback = ui::create_mock_response(&msg_clone);
                        crate::chat::store_async_response(request_id, msg_clone, fallback);
                    }
                }
            });
        }
        
        log::info!("Response queued for processing");
    }

    fn process_pending_chat_responses(&mut self) {
        // Check for async responses in WASM
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(request_id) = &self.chat_state.pending_request_id.clone() {
                if let Some((message, response)) = crate::chat::check_async_response(request_id) {
                    log::info!("Found async response, processing it");
                    self.chat_state.pending_request_id = None;
                    self.pending_chat_responses.push((message, response));
                }
            }
        }
        
        if !self.pending_chat_responses.is_empty() {
            let responses = std::mem::take(&mut self.pending_chat_responses);
            for (_, response) in responses {
                // Get current camera position for response formatting
                let camera_pos = self.splatting_args.camera.position;
                let current_location = [camera_pos.x, camera_pos.y, camera_pos.z];
                
                let response_text = ui::format_response(&response, current_location);
                self.chat_state.add_message(response_text, false);
                self.chat_state.set_highlights(response.clone());
                
                // Update highlighting renderer and animate camera
                self.update_highlights_and_animate(response);
            }
            self.chat_state.is_sending = false;
        }
    }

    fn update_highlights_and_animate(&mut self, response: McpResponse) {
        // Log current camera position for debugging
        let camera_pos = self.splatting_args.camera.position;
        log::info!("Camera position: ({:.3}, {:.3}, {:.3})", camera_pos.x, camera_pos.y, camera_pos.z);
        
        // Check if this is a navigation response (has paths)
        if !response.paths.is_empty() {
            log::info!("🗺️ Navigation response detected - {} paths found", response.paths.len());
            
            let path_response = &response.paths[0]; // Use first path
            
            // Create ScenePath from PathResponse
            let scene_path = crate::chat::ScenePath {
                waypoints: path_response.path.clone(),
                description: Some(format!("Path to {}", path_response.object.name)),
            };
            
            // Set highlighting for both path and target object
            self.highlight_renderer.set_highlighted_objects(vec![path_response.object.clone()], &self.wgpu_context.device);
            self.highlight_renderer.set_highlighted_path(Some(scene_path.clone()), &self.wgpu_context.device);
            
            // Start camera animation along the path using target object's coordinate system
            log::info!("Starting camera animation along path with {} waypoints to '{}'", 
                       scene_path.waypoints.len(), path_response.object.name);
            self.animate_camera_along_path_with_object(scene_path, &path_response.object);
        }
        // Check if this is an object response (has objects but no paths)
        else if !response.answer.is_empty() {
            // Log object bounding boxes and sizes for debugging
            log::info!("📦 Found {} objects, analyzing sizes to select biggest for camera positioning:", response.answer.len());
            
            for (i, obj) in response.answer.iter().enumerate() {
                if obj.aligned_bbox.len() >= 8 {
                    // Calculate center and size from bounding box
                    let mut center = [0.0, 0.0, 0.0];
                    for point in &obj.aligned_bbox {
                        center[0] += point[0];
                        center[1] += point[1];
                        center[2] += point[2];
                    }
                    center[0] /= 8.0;
                    center[1] /= 8.0;
                    center[2] /= 8.0;
                    
                    // Calculate object volume for size comparison
                    let bbox_points: Vec<Vector3<f32>> = obj.aligned_bbox
                        .iter()
                        .map(|p| Vector3::new(p[0], p[1], p[2]))
                        .collect();
                    let width = (bbox_points[2] - bbox_points[3]).magnitude();
                    let height = (bbox_points[7] - bbox_points[3]).magnitude();
                    let depth = (bbox_points[0] - bbox_points[3]).magnitude();
                    let volume = width * height * depth;
                    
                    log::info!("  Object {}: '{}' at center ({:.3}, {:.3}, {:.3}) - Volume: {:.3} (W:{:.2} H:{:.2} D:{:.2})", 
                        i + 1, obj.name, center[0], center[1], center[2], volume, width, height, depth);
                }
            }
            
            // Update highlighting renderer
            self.highlight_renderer.set_highlighted_objects(response.answer.clone(), &self.wgpu_context.device);
            self.highlight_renderer.set_highlighted_path(None, &self.wgpu_context.device);
            
            // Animate camera to first object with scene ground-aligned positioning
            if let Some((target_center, optimal_camera_pos, object_size, ground_normal)) = self.highlight_renderer.get_first_object_viewing_info() {
                self.animate_camera_to_ground_aligned_position(target_center, optimal_camera_pos, object_size, ground_normal);
            }
        } else {
            // Clear highlights if no objects or paths found
            log::info!("No objects or paths found in response - clearing highlights");
            self.highlight_renderer.clear_highlights();
        }
    }

    fn animate_camera_to_ground_aligned_position(&mut self, target_center: Point3<f32>, optimal_camera_pos: Point3<f32>, object_size: f32, _ground_normal: Vector3<f32>) {
        use cgmath::{Deg, Rad};
        
        // Get the current camera position for debugging
        let current_pos = self.splatting_args.camera.position;
        log::info!("Current camera before animation: ({:.3}, {:.3}, {:.3})", current_pos.x, current_pos.y, current_pos.z);
        
        // Calculate look direction from camera position to target center
        let look_direction = (target_center - optimal_camera_pos).normalize();
        
        // Use the scene's ground up direction for stable camera positioning
        let ground_up = self.ground_up_direction;
        
        // Create camera rotation aligned with the scene's ground plane
        // The camera will look at the object with the scene's ground plane as reference
        let rotation = Quaternion::look_at(look_direction, ground_up);
        
        // No additional tilt needed - camera is already aligned with ground plane
        let final_rotation = rotation;
        
        // Set appropriate FOV based on object size (slightly wider for better framing)
        let base_fov = Deg(50.0); // Slightly wider than before
        let fov_adjustment = (object_size / 8.0).min(1.8).max(0.6); // More adaptive FOV
        let fovx = Rad::from(base_fov * fov_adjustment);
        let fovy = Rad::from(base_fov * fov_adjustment);
        
        // Create projection with calculated parameters
        let projection = crate::camera::PerspectiveProjection {
            fovx,
            fovy,
            znear: (object_size * 0.05).max(0.01),  // Closer near plane for better detail
            zfar: (object_size * 15.0).max(150.0),  // Extended far plane
            fov2view_ratio: fovx.0 / fovy.0,
        };
        
        // Set controller center to the object center for proper orbiting
        self.controller.center = target_center;
        
        // Update controller's up vector to match scene's ground up direction
        self.controller.up = Some(ground_up);
        
        // Create the ground-aligned camera
        let final_camera = PerspectiveCamera::new(
            optimal_camera_pos,
            final_rotation,
            projection,
        );
        
        // Log the calculated positioning for debugging
        log::info!("Scene ground-aligned camera positioning:");
        log::info!("  Object center: ({:.3}, {:.3}, {:.3})", target_center.x, target_center.y, target_center.z);
        log::info!("  Camera position: ({:.3}, {:.3}, {:.3})", optimal_camera_pos.x, optimal_camera_pos.y, optimal_camera_pos.z);
        log::info!("  Look direction: ({:.3}, {:.3}, {:.3})", look_direction.x, look_direction.y, look_direction.z);
        log::info!("  Scene ground up: ({:.3}, {:.3}, {:.3})", ground_up.x, ground_up.y, ground_up.z);
        log::info!("  Object size: {:.3}", object_size);
        log::info!("  FOV: {:.1}°", base_fov.0 * fov_adjustment);
        
        // Cancel any existing animation first
        self.animation.take();
        
        // Animate to the ground-aligned camera position with smooth transition
        self.set_camera(final_camera, Duration::from_millis(1500));
        log::info!("Camera animating to scene ground-aligned position");
    }

    fn animate_camera_along_path_with_object(&mut self, path: crate::chat::ScenePath, target_object: &crate::chat::SceneObject) {
        use cgmath::{Deg, Rad};
        
        if path.waypoints.is_empty() {
            log::warn!("Cannot animate along empty path");
            return;
        }
        
        log::info!("🎬 Starting elegant path animation with {} waypoints", path.waypoints.len());
        
        // Use the scene's ground up direction for stable camera positioning
        let ground_normal = self.ground_up_direction;
        
        log::info!("Scene ground up direction: ({:.3}, {:.3}, {:.3})", 
                   ground_normal.x, ground_normal.y, ground_normal.z);
        
        // Calculate object center from aligned_bbox
        let object_center = {
            let mut center = Vector3::new(0.0, 0.0, 0.0);
            for point in &target_object.aligned_bbox {
                center.x += point[0];
                center.y += point[1];
                center.z += point[2];
            }
            center / 8.0
        };
        
                // Calculate camera height relative to ground plane (increased for better visibility)
        let camera_height_above_ground = 4.0; // Increased height above ground plane
        
        let mut cameras = Vec::new();
        
        // Create cameras for each waypoint
        for (i, waypoint) in path.waypoints.iter().enumerate() {
            let waypoint_pos = Vector3::new(waypoint[0], waypoint[1], waypoint[2]);
            
            let (camera_pos, look_direction, projection) = if i == path.waypoints.len() - 1 {
                // For final waypoint, use optimal object viewing position (same as object search)
                if let Some((target_center, optimal_camera_pos, object_size, _)) = self.highlight_renderer.get_first_object_viewing_info() {
                    let look_dir = (target_center - optimal_camera_pos).normalize();
                    
                    // Project look direction onto ground plane
                    let projected_look = look_dir - look_dir.dot(ground_normal) * ground_normal;
                    let final_look_direction = if projected_look.magnitude() > 0.001 {
                        projected_look.normalize()
                    } else {
                        Vector3::new(1.0, 0.0, 0.0)
                    };
                    
                    // Use object-specific projection for better framing
                    let base_fov = Deg(50.0);
                    let fov_adjustment = (object_size / 8.0).min(1.8).max(0.6);
                    let fovx = Rad::from(base_fov * fov_adjustment);
                    let fovy = Rad::from(base_fov * fov_adjustment);
                    
                    let obj_projection = crate::camera::PerspectiveProjection {
                        fovx,
                        fovy,
                        znear: (object_size * 0.05).max(0.01),
                        zfar: (object_size * 15.0).max(150.0),
                        fov2view_ratio: fovx.0 / fovy.0,
                    };
                    
                    log::info!("  Final waypoint: using optimal object viewing position");
                    (Vector3::from(optimal_camera_pos.to_vec()), final_look_direction, obj_projection)
                } else {
                    // Fallback if no optimal position available
                    let camera_pos = waypoint_pos + ground_normal * camera_height_above_ground;
                    let look_dir = (object_center - waypoint_pos).normalize();
                    let projected_look = look_dir - look_dir.dot(ground_normal) * ground_normal;
                    let final_look_direction = if projected_look.magnitude() > 0.001 {
                        projected_look.normalize()
                    } else {
                        Vector3::new(1.0, 0.0, 0.0)
                    };
                    
                    let nav_projection = crate::camera::PerspectiveProjection {
                        fovx: Rad::from(Deg(70.0)),
                        fovy: Rad::from(Deg(50.0)),
                        znear: 0.1,
                        zfar: 500.0,
                        fov2view_ratio: Deg(70.0).0 / Deg(50.0).0,
                    };
                    
                    (camera_pos, final_look_direction, nav_projection)
                }
            } else {
                // For regular waypoints, position at height above ground plane
                let camera_pos = waypoint_pos + ground_normal * camera_height_above_ground;
                
                // Look towards next waypoint
                let next_waypoint = Vector3::new(
                    path.waypoints[i + 1][0],
                    path.waypoints[i + 1][1],
                    path.waypoints[i + 1][2]
                );
                let forward_direction = (next_waypoint - waypoint_pos).normalize();
                
                // Project forward direction onto ground plane to keep camera parallel
                let projected_forward = forward_direction - forward_direction.dot(ground_normal) * ground_normal;
                let look_direction = if projected_forward.magnitude() > 0.001 {
                    projected_forward.normalize()
                } else {
                    Vector3::new(1.0, 0.0, 0.0)
                };
                
                let nav_projection = crate::camera::PerspectiveProjection {
                    fovx: Rad::from(Deg(70.0)),
                    fovy: Rad::from(Deg(50.0)),
                    znear: 0.1,
                    zfar: 500.0,
                    fov2view_ratio: Deg(70.0).0 / Deg(50.0).0,
                };
                
                (camera_pos, look_direction, nav_projection)
            };
            
            // Create camera rotation using ground plane normal as up vector
            let rotation = Quaternion::look_at(look_direction, ground_normal);
            
            let camera = PerspectiveCamera::new(
                Point3::from_vec(camera_pos),
                rotation,
                projection,
            );
            
            cameras.push(camera);
            
            log::info!("  Waypoint {}: ground ({:.2}, {:.2}, {:.2}) -> camera ({:.2}, {:.2}, {:.2})",
                       i + 1, waypoint_pos.x, waypoint_pos.y, waypoint_pos.z,
                       camera_pos.x, camera_pos.y, camera_pos.z);
        }
        
        // Set controller to use ground plane for consistent controls
        self.controller.up = Some(ground_normal);
        
        // Create simple navigation sequence
        let seconds_per_waypoint = 1.0; // 1 second per waypoint for smooth motion
        let total_duration = Duration::from_secs_f32(cameras.len() as f32 * seconds_per_waypoint);
        
        let navigation = NavigationSequence::new(cameras, seconds_per_waypoint);
        let animation = crate::animation::Animation::new(
            total_duration,
            false, // No looping
            Box::new(navigation),
        );
        
        // Start the animation
        self.animation.take();
        self.animation = Some((animation, true));
        
        log::info!("✅ Started elegant path animation: {} waypoints, {:.1}s duration", 
                   path.waypoints.len(), total_duration.as_secs_f32());
        log::info!("   Camera maintains {:.1} units height above ground plane", camera_height_above_ground);
    }
    


    /// returns whether the sceen changed and we need a redraw
    fn update(&mut self, dt: Duration)  {
        // Process any pending chat responses
        self.process_pending_chat_responses();
        
        // ema fps update

        if self.splatting_args.walltime < Duration::from_secs(5) {
            self.splatting_args.walltime += dt;
        }
        if let Some((next_camera, playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.cancle_animation()
            } else {
                let dt = if *playing { dt } else { Duration::ZERO };
                self.splatting_args.camera = next_camera.update(dt);
                self.splatting_args
                    .camera
                    .projection
                    .resize(self.config.width, self.config.height);
                if next_camera.done() {
                    self.animation.take();
                    self.controller.reset_to_camera(self.splatting_args.camera);
                }
            }
        } else {
            self.controller
                .update_camera(&mut self.splatting_args.camera, dt);

            // check if camera moved out of selected view
            if let Some(idx) = self.current_view {
                if let Some(scene) = &self.scene {
                    if let Some(camera) = scene.camera(idx) {
                        let scene_camera: PerspectiveCamera = camera.into();
                        if !self.splatting_args.camera.position.ulps_eq(
                            &scene_camera.position,
                            1e-4,
                            f32::default_max_ulps(),
                        ) || !self.splatting_args.camera.rotation.ulps_eq(
                            &scene_camera.rotation,
                            1e-4,
                            f32::default_max_ulps(),
                        ) {
                            self.current_view.take();
                        }
                    }
                }
            }
        }

        let aabb = self.pc.bbox();
        self.splatting_args.camera.fit_near_far(aabb);
    }

    fn render(
        &mut self,
        redraw_scene: bool,
        shapes: Option<FullOutput>,
    ) -> Result<(), wgpu::SurfaceError> {
        self.stopwatch.as_mut().map(|s| s.reset());

        let output = self.surface.get_current_texture()?;
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let view_srgb = output.texture.create_view(&Default::default());
        // do prepare stuff

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        if redraw_scene {
            self.renderer.prepare(
                &mut encoder,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &self.pc,
                self.splatting_args,
                (&mut self.stopwatch).into(),
            );
        }

        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &mut encoder,
                shapes,
            )
        });

        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.start(&mut encoder, "rasterization").unwrap();
        }
        if redraw_scene {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.display.texture(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.splatting_args.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer.render(&mut render_pass, &self.pc);
        }
        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.stop(&mut encoder, "rasterization").unwrap();
        }

        self.display.render(
            &mut encoder,
            &view_rgb,
            self.splatting_args.background_color,
            self.renderer.camera(),
            &self.renderer.render_settings(),
        );

        // Render highlights on top of the scene
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("highlight render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_rgb,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.highlight_renderer.render(&mut render_pass, self.renderer.camera());
        }

        self.stopwatch.as_mut().map(|s| s.end(&mut encoder));

        if let Some(state) = &ui_state {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_srgb,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.ui_renderer.render(&mut render_pass, state);
        }


        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        self.wgpu_context.queue.submit([encoder.finish()]);

        output.present();
        self.splatting_args.resolution = Vector2::new(self.config.width, self.config.height);
        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.splatting_args.scene_extend = Some(scene.extend());
        let mut center = Point3::origin();
        for c in scene.cameras(None) {
            let z_axis: Vector3<f32> = c.rotation[2].into();
            center += Vector3::from(c.position) + z_axis * 2.;
        }
        center /= scene.num_cameras() as f32;

        self.controller.center = center;
        
        // Get the up vector from the first camera in the scene
        self.ground_up_direction = scene.get_first_camera_up();
        
        // Update controller's up vector to use the scene's ground up direction
        self.controller.up = Some(self.ground_up_direction);
        
        // Update highlighting renderer's ground up direction
        self.highlight_renderer.set_scene_ground_up(self.ground_up_direction);
        
        self.scene.replace(scene);
        if self.saved_cameras.is_empty() {
            self.saved_cameras = self
                .scene
                .as_ref()
                .unwrap()
                .cameras(Some(Split::Test))
                .clone();
        }
    }

    fn set_env_map<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let env_map_exr = image::open(path)?;
        let env_map_data: Vec<[f32; 4]> = env_map_exr
            .as_rgb32f()
            .ok_or(anyhow::anyhow!("env map must be rgb"))?
            .pixels()
            .map(|p| p.to_rgba().0)
            .collect();

        let env_texture = self.wgpu_context.device.create_texture_with_data(
            &self.wgpu_context.queue,
            &wgpu::TextureDescriptor {
                label: Some("env map texture"),
                size: Extent3d {
                    width: env_map_exr.width(),
                    height: env_map_exr.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&env_map_data.as_slice()),
        );
        self.display.set_env_map(
            &self.wgpu_context.device,
            Some(&env_texture.create_view(&Default::default())),
        );
        self.splatting_args.show_env_map = true;
        Ok(())
    }

    fn start_tracking_shot(&mut self) {
        if self.saved_cameras.len() > 1 {
            let shot = TrackingShot::from_cameras(self.saved_cameras.clone());
            let a = Animation::new(
                Duration::from_secs_f32(self.saved_cameras.len() as f32 * 2.),
                true,
                Box::new(shot),
            );
            self.animation = Some((a, true));
        }
    }

    fn cancle_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn stop_animation(&mut self) {
        if let Some((_animation, playing)) = &mut self.animation {
            *playing = false;
        }
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn set_scene_camera(&mut self, i: usize) {
        if let Some(scene) = &self.scene {
            self.current_view.replace(i);
            log::info!("view moved to camera {i}");
            if let Some(camera) = scene.camera(i) {
                // Ensure controller maintains the scene's ground up direction
                self.controller.up = Some(self.ground_up_direction);
                self.set_camera(camera, Duration::from_millis(200));
            } else {
                log::error!("camera {i} not found");
            }
        }
    }

    pub fn set_camera<C: Into<PerspectiveCamera>>(
        &mut self,
        camera: C,
        animation_duration: Duration,
    ) {
        let camera: PerspectiveCamera = camera.into();
        if animation_duration.is_zero() {
            self.update_camera(camera.into())
        } else {
            let target_camera = camera.into();
            let a = Animation::new(
                animation_duration,
                false,
                Box::new(Transition::new(
                    self.splatting_args.camera.clone(),
                    target_camera,
                    smoothstep,
                )),
            );
            self.animation = Some((a, true));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        self.splatting_args.camera = camera;
        self.splatting_args
            .camera
            .projection
            .resize(self.config.width, self.config.height);
    }

    fn save_view(&mut self) {
        let max_scene_id = if let Some(scene) = &self.scene {
            scene.cameras(None).iter().map(|c| c.id).max().unwrap_or(0)
        } else {
            0
        };
        let max_id = self.saved_cameras.iter().map(|c| c.id).max().unwrap_or(0);
        let id = max_id.max(max_scene_id) + 1;
        self.saved_cameras.push(SceneCamera::from_perspective(
            self.splatting_args.camera,
            id.to_string(),
            id,
            Vector2::new(self.config.width, self.config.height),
            Split::Test,
        ));
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let scene = scene_file.and_then(|f| match Scene::from_json(f) {
        Ok(s) => Some(s),
        Err(err) => {
            log::error!("cannot load scene: {:?}", err);
            None
        }
    });

    // let window_size = if let Some(scene) = &scene {
    //     let camera = scene.camera(0).unwrap();
    //     let factor = 1200. / camera.width as f32;
    //     LogicalSize::new(
    //         (camera.width as f32 * factor) as u32,
    //         (camera.height as f32 * factor) as u32,
    //     )
    // } else {
    //     LogicalSize::new(800, 600)
    // };
    let window_size = LogicalSize::new(800, 600);
    let window_attributes = Window::default_attributes()
        .with_inner_size( window_size)
        .with_title(format!(
            "{} ({})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ));
        
    #[allow(deprecated)]
    let window = event_loop.create_window(window_attributes).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.get_element_by_id("loading-display")
                    .unwrap()
                    .set_text_content(Some("Unpacking"));
                doc.body()
            })
            .and_then(|body| {
                let canvas = window.canvas().unwrap();
                canvas.set_id("window-canvas");
                canvas.set_width(body.client_width() as u32);
                canvas.set_height(body.client_height() as u32);
                let elm = web_sys::Element::from(canvas);
                elm.set_attribute("style", "width: 100%; height: 100%;")
                    .unwrap();
                body.append_child(&elm).ok()
            })
            .expect("couldn't append canvas to document body");
    }

    // limit the redraw rate to the monitor refresh rate
    let min_wait = window
        .current_monitor()
        .map(|m| {
            let hz = m.refresh_rate_millihertz().unwrap_or(60_000);
            Duration::from_millis(1000000 / hz as u64)
        })
        .unwrap_or(Duration::from_millis(17));

    let mut state = WindowContext::new(window, file, &config).await.unwrap();
    state.pointcloud_file_path = pointcloud_file_path;

    if let Some(scene) = scene {
        state.set_scene(scene);
        state.set_scene_camera(0);
        state.scene_file_path = scene_file_path;
    }

    if let Some(skybox) = &config.skybox {
        if let Err(e) = state.set_env_map(skybox.as_path()) {
            log::error!("failed do set skybox: {e}");
        }
    }

    #[cfg(target_arch = "wasm32")]
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            doc.get_element_by_id("spinner")
                .unwrap()
                .set_attribute("style", "display:none;")
                .unwrap();
            doc.body()
        });

    let mut last = Instant::now();

    #[allow(deprecated)]
    event_loop.run(move |event,target| 
        
        match event {
            Event::NewEvents(e) =>  match e{
                winit::event::StartCause::ResumeTimeReached { .. }=>{
                    state.window.request_redraw();
                }
                _=>{}
            },
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() && !state.ui_renderer.on_event(&state.window,event) => match event {
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size, None);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                ..
            } => {
                state.scale_factor = *scale_factor as f32;
            }
            WindowEvent::CloseRequested => {log::info!("close!");target.exit()},
            WindowEvent::ModifiersChanged(m)=>{
                state.controller.alt_pressed = m.state().alt_key();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key{
                if event.state == ElementState::Released{

                    if key == KeyCode::KeyT{
                        if state.animation.is_none(){
                            state.start_tracking_shot();
                        }else{
                            state.stop_animation()
                        }
                    }else if key == KeyCode::KeyU{
                        state.ui_visible = !state.ui_visible;
                        
                    }else if key == KeyCode::KeyC{
                        state.save_view();
                    } else  if key == KeyCode::KeyR && state.controller.alt_pressed{
                        if let Err(err) = state.reload(){
                            log::error!("failed to reload volume: {:?}", err);
                        }   
                    }else if let Some(scene) = &state.scene{

                        let new_camera = 
                        if let Some(num) = key_to_num(key){
                            Some(num as usize)
                        }
                        else if key == KeyCode::KeyR{
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                Some((rand::random::<u32>() as usize)%scene.num_cameras())
                            }
                            #[cfg(target_arch = "wasm32")]
                            {
                                // For WASM, just use the first camera instead of random
                                Some(0)
                            }
                        }else if key == KeyCode::KeyN{
                            scene.nearest_camera(state.splatting_args.camera.position,None)
                        }else if key == KeyCode::PageUp{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }else if key == KeyCode::KeyT{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }
                        else if key == KeyCode::PageDown{
                            Some(state.current_view.map_or(0, |v|v-1) % scene.num_cameras())
                        }else{None};

                        if let Some(new_camera) = new_camera{
                            state.set_scene_camera(new_camera);
                        }
                    }
                }
                state
                    .controller
                    .process_keyboard(key, event.state == ElementState::Pressed);
            }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    state.controller.process_scroll(*dy )
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    state.controller.process_scroll(p.y as f32 / 100.)
                }
            },
            WindowEvent::MouseInput { state:button_state, button, .. }=>{
                match button {
                    winit::event::MouseButton::Left =>                         state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                    winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                    _=>{}
                }
            }
            WindowEvent::RedrawRequested => {
                if !config.no_vsync{
                    // make sure the next redraw is called with a small delay
                    target.set_control_flow(ControlFlow::wait_duration(min_wait));
                }
                let now = Instant::now();
                let dt = now-last;
                last = now;

                let old_settings = state.splatting_args.clone();
                state.update(dt);

                let (redraw_ui,shapes) = state.ui();

                let resolution_change = state.splatting_args.resolution != Vector2::new(state.config.width, state.config.height);

                let request_redraw = old_settings != state.splatting_args || resolution_change;
    
                if request_redraw || redraw_ui{
                    state.fps = (1. / dt.as_secs_f32()) * 0.05 + state.fps * 0.95;
                    match state.render(request_redraw,state.ui_visible.then_some(shapes)) {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size(), None),
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) =>target.exit(),
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => println!("error: {:?}", e),
                    }
                }
                if config.no_vsync{
                    state.window.request_redraw();
                }
            }
            _ => {}
        },
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion{ delta, },
            .. // We're not using device_id currently
        } => {
            state.controller.process_mouse(delta.0 as f32, delta.1 as f32)
        }
        _ => {},
    }).unwrap();
}

#[cfg(target_arch = "wasm32")]
fn get_web_mcp_server_url() -> String {
    use web_sys::window;
    
    // Try to get URL from localStorage first
    if let Some(window) = window() {
        if let Ok(Some(storage)) = window.local_storage() {
            if let Ok(Some(stored_url)) = storage.get_item("mcp_server_url") {
                if !stored_url.is_empty() {
                    log::info!("Using MCP server URL from localStorage: {}", stored_url);
                    return stored_url;
                }
            }
        }
        
        // Try to get URL from URL parameters
        if let Some(location) = window.location().href().ok() {
            if let Ok(url) = web_sys::Url::new(&location) {
                let search_params = url.search_params();
                if let Some(mcp_url) = search_params.get("mcp_server_url") {
                    if !mcp_url.is_empty() {
                        log::info!("Using MCP server URL from URL parameter: {}", mcp_url);
                        return mcp_url;
                    }
                }
            }
        }
    }
    
    // Default fallback
    let default_url = "http://localhost:8080".to_string();
    log::info!("Using default MCP server URL: {}", default_url);
    default_url
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_wasm(
    pc: Vec<u8>,
    scene: Option<Vec<u8>>,
    pc_file: Option<String>,
    scene_file: Option<String>,
) {
    use std::{io::Cursor, str::FromStr};

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    let mcp_server_url = get_web_mcp_server_url();
    log::info!("🌐 Web app starting with MCP server URL: {}", mcp_server_url);

    wasm_bindgen_futures::spawn_local(open_window(
        pc_reader,
        scene_reader,
        RenderConfig {
            no_vsync: false,
            skybox: None,
            hdr: false,
            mcp_server_url,
        },
        pc_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        scene_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
    ));
}
