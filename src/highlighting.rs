use crate::chat::{SceneObject, ScenePath};
use crate::renderer::CameraUniform;
use crate::uniform::UniformBuffer;
use cgmath::{EuclideanSpace, InnerSpace, Matrix4, Point3, Vector3, Vector4};
use wgpu::{
    include_wgsl, util::DeviceExt, BufferUsages, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexStepMode,
};

pub struct HighlightRenderer {
    box_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    path_pipeline: wgpu::RenderPipeline,
    arrow_pipeline: wgpu::RenderPipeline,
    box_vertex_buffer: Option<wgpu::Buffer>,
    box_index_buffer: Option<wgpu::Buffer>,
    wireframe_vertex_buffer: Option<wgpu::Buffer>,
    wireframe_index_buffer: Option<wgpu::Buffer>,
    path_vertex_buffer: Option<wgpu::Buffer>,
    arrow_vertex_buffer: Option<wgpu::Buffer>,
    highlighted_objects: Vec<SceneObject>,
    highlighted_path: Option<ScenePath>,
    box_instances_buffer: Option<wgpu::Buffer>,
    wireframe_instances_buffer: Option<wgpu::Buffer>,
    box_vertex_count: u32,
    wireframe_vertex_count: u32,
    path_vertex_count: u32,
    arrow_vertex_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoxInstance {
    model_matrix: Matrix4<f32>,
    color: Vector4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColoredVertex {
    position: Vector3<f32>,
    color: Vector4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PathVertex {
    position: Vector3<f32>,
    color: Vector4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ArrowVertex {
    position: Vector3<f32>,
    color: Vector4<f32>,
}

impl HighlightRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let box_pipeline = Self::create_box_pipeline(device, target_format);
        let wireframe_pipeline = Self::create_wireframe_pipeline(device, target_format);
        let path_pipeline = Self::create_path_pipeline(device, target_format);
        let arrow_pipeline = Self::create_arrow_pipeline(device, target_format);

        // Create static box geometry (unit cube wireframe)
        let (box_vertex_buffer, box_index_buffer) = Self::create_box_geometry(device);
        let (wireframe_vertex_buffer, wireframe_index_buffer) =
            Self::create_wireframe_geometry(device);

        Self {
            box_pipeline,
            wireframe_pipeline,
            path_pipeline,
            arrow_pipeline,
            box_vertex_buffer: Some(box_vertex_buffer),
            box_index_buffer: Some(box_index_buffer),
            wireframe_vertex_buffer: Some(wireframe_vertex_buffer),
            wireframe_index_buffer: Some(wireframe_index_buffer),
            path_vertex_buffer: None,
            arrow_vertex_buffer: None,
            highlighted_objects: Vec::new(),
            highlighted_path: None,
            box_instances_buffer: None,
            wireframe_instances_buffer: None,
            box_vertex_count: 0,
            wireframe_vertex_count: 0,
            path_vertex_count: 0,
            arrow_vertex_count: 0,
        }
    }

    fn create_box_pipeline(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("highlight box pipeline layout"),
            bind_group_layouts: &[&UniformBuffer::<CameraUniform>::bind_group_layout(device)],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(include_wgsl!("shaders/highlight_direct.wgsl"));

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight box pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Direct colored vertices (position + color)
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<ColoredVertex>() as wgpu::BufferAddress,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &[
                            VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: VertexFormat::Float32x3,
                            },
                            VertexAttribute {
                                offset: std::mem::size_of::<Vector3<f32>>() as wgpu::BufferAddress,
                                shader_location: 1,
                                format: VertexFormat::Float32x4,
                            },
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_path_pipeline(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("highlight path pipeline layout"),
            bind_group_layouts: &[&UniformBuffer::<CameraUniform>::bind_group_layout(device)],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(include_wgsl!("shaders/highlight_path.wgsl"));

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight path pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[VertexBufferLayout {
                    array_stride: std::mem::size_of::<PathVertex>() as wgpu::BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: VertexFormat::Float32x3,
                        },
                        VertexAttribute {
                            offset: std::mem::size_of::<Vector3<f32>>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_arrow_pipeline(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("highlight arrow pipeline layout"),
            bind_group_layouts: &[&UniformBuffer::<CameraUniform>::bind_group_layout(device)],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(include_wgsl!("shaders/highlight_direct.wgsl"));

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight arrow pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[VertexBufferLayout {
                    array_stride: std::mem::size_of::<ArrowVertex>() as wgpu::BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: VertexFormat::Float32x3,
                        },
                        VertexAttribute {
                            offset: std::mem::size_of::<Vector3<f32>>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_box_geometry(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
        // Unit cube vertices
        let vertices: &[Vector3<f32>] = &[
            Vector3::new(-0.5, -0.5, -0.5), // 0
            Vector3::new(0.5, -0.5, -0.5),  // 1
            Vector3::new(0.5, 0.5, -0.5),   // 2
            Vector3::new(-0.5, 0.5, -0.5),  // 3
            Vector3::new(-0.5, -0.5, 0.5),  // 4
            Vector3::new(0.5, -0.5, 0.5),   // 5
            Vector3::new(0.5, 0.5, 0.5),    // 6
            Vector3::new(-0.5, 0.5, 0.5),   // 7
        ];

        // Solid cube indices (triangles for each face)
        let indices: &[u16] = &[
            // Front face
            0, 1, 2, 2, 3, 0, // Back face
            4, 6, 5, 6, 4, 7, // Left face
            4, 0, 3, 3, 7, 4, // Right face
            1, 5, 6, 6, 2, 1, // Bottom face
            4, 5, 1, 1, 0, 4, // Top face
            3, 2, 6, 6, 7, 3,
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("highlight box vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("highlight box indices"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer)
    }

    fn create_wireframe_pipeline(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("highlight wireframe pipeline layout"),
            bind_group_layouts: &[&UniformBuffer::<CameraUniform>::bind_group_layout(device)],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(include_wgsl!("shaders/highlight_direct.wgsl"));

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight wireframe pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Direct colored vertices (position + color)
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<ColoredVertex>() as wgpu::BufferAddress,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &[
                            VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: VertexFormat::Float32x3,
                            },
                            VertexAttribute {
                                offset: std::mem::size_of::<Vector3<f32>>() as wgpu::BufferAddress,
                                shader_location: 1,
                                format: VertexFormat::Float32x4,
                            },
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_wireframe_geometry(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
        // Same vertices as box
        let vertices: &[Vector3<f32>] = &[
            Vector3::new(-0.5, -0.5, -0.5), // 0
            Vector3::new(0.5, -0.5, -0.5),  // 1
            Vector3::new(0.5, 0.5, -0.5),   // 2
            Vector3::new(-0.5, 0.5, -0.5),  // 3
            Vector3::new(-0.5, -0.5, 0.5),  // 4
            Vector3::new(0.5, -0.5, 0.5),   // 5
            Vector3::new(0.5, 0.5, 0.5),    // 6
            Vector3::new(-0.5, 0.5, 0.5),   // 7
        ];

        // Wireframe edges
        let indices: &[u16] = &[
            // Bottom face edges
            0, 1, 1, 2, 2, 3, 3, 0, // Top face edges
            4, 5, 5, 6, 6, 7, 7, 4, // Vertical edges
            0, 4, 1, 5, 2, 6, 3, 7,
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("highlight wireframe vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("highlight wireframe indices"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer)
    }

    pub fn set_highlighted_objects(&mut self, objects: Vec<SceneObject>, device: &wgpu::Device) {
        log::info!(
            "🔥🔥🔥 HIGHLIGHT RENDERER: Setting {} highlighted objects",
            objects.len()
        );
        self.highlighted_objects = objects;
        self.update_box_instances(device);
    }

    pub fn set_highlighted_path(&mut self, path: Option<ScenePath>, device: &wgpu::Device) {
        self.highlighted_path = path;
        self.update_path_geometry(device);
    }

    pub fn clear_highlights(&mut self) {
        self.highlighted_objects.clear();
        self.highlighted_path = None;
        self.box_instances_buffer = None;
        self.wireframe_instances_buffer = None;
        self.path_vertex_buffer = None;
        self.arrow_vertex_buffer = None;
        self.box_vertex_count = 0;
        self.wireframe_vertex_count = 0;
        self.path_vertex_count = 0;
        self.arrow_vertex_count = 0;
    }

    fn update_box_instances(&mut self, device: &wgpu::Device) {
        if self.highlighted_objects.is_empty() {
            self.box_instances_buffer = None;
            self.wireframe_instances_buffer = None;
            self.arrow_vertex_buffer = None;
            return;
        }

        log::info!(
            "Creating exact bounding boxes for {} objects using MCP server coordinates",
            self.highlighted_objects.len()
        );

        // Create vertex buffers using exact coordinates from MCP server
        let mut all_box_vertices = Vec::new();
        let mut all_wireframe_vertices = Vec::new();
        let mut all_arrow_vertices: Vec<ArrowVertex> = Vec::new();

        for (i, obj) in self.highlighted_objects.iter().enumerate() {
            if obj.aligned_bbox.len() >= 8 {
                log::info!(
                    "Using exact 8 vertices from MCP server for object: {}",
                    obj.name
                );

                // Cycle through colors for different objects
                let solid_colors = [
                    Vector4::new(1.0, 0.0, 0.0, 0.25), // Red with moderate transparency
                    Vector4::new(0.0, 1.0, 0.0, 0.25), // Green with moderate transparency
                    Vector4::new(0.0, 0.5, 1.0, 0.25), // Blue with moderate transparency
                    Vector4::new(1.0, 1.0, 0.0, 0.25), // Yellow with moderate transparency
                    Vector4::new(1.0, 0.0, 1.0, 0.25), // Magenta with moderate transparency
                    Vector4::new(0.0, 1.0, 1.0, 0.25), // Cyan with moderate transparency
                ];

                let wireframe_colors = [
                    Vector4::new(1.0, 0.2, 0.2, 0.8), // Bright red, mostly opaque
                    Vector4::new(0.2, 1.0, 0.2, 0.8), // Bright green, mostly opaque
                    Vector4::new(0.2, 0.6, 1.0, 0.8), // Bright blue, mostly opaque
                    Vector4::new(1.0, 1.0, 0.2, 0.8), // Bright yellow, mostly opaque
                    Vector4::new(1.0, 0.2, 1.0, 0.8), // Bright magenta, mostly opaque
                    Vector4::new(0.2, 1.0, 1.0, 0.8), // Bright cyan, mostly opaque
                ];

                let solid_color = solid_colors[i % solid_colors.len()];
                let wireframe_color = wireframe_colors[i % wireframe_colors.len()];

                // Convert the 8 MCP vertices to our vertex format
                let vertices: Vec<ColoredVertex> = obj
                    .aligned_bbox
                    .iter()
                    .map(|point| ColoredVertex {
                        position: Vector3::new(point[0], point[1], point[2]),
                        color: solid_color,
                    })
                    .collect();

                let wireframe_vertices: Vec<ColoredVertex> = obj
                    .aligned_bbox
                    .iter()
                    .map(|point| ColoredVertex {
                        position: Vector3::new(point[0], point[1], point[2]),
                        color: wireframe_color,
                    })
                    .collect();

                // Add faces for solid box using MCP server vertex ordering
                // Face indices for solid cube rendering (6 faces, 2 triangles each)
                // MCP ordering: 0=bottom back left, 1=bottom back right, 2=bottom front right, 3=bottom front left
                //               4=top back left,    5=top back right,    6=top front right,    7=top front left
                let face_indices = [
                    // Front face (vertices 3,2,6,7) - front left, front right, top front right, top front left
                    3, 2, 6, 6, 7, 3,
                    // Back face (vertices 0,1,5,4) - back left, back right, top back right, top back left
                    0, 1, 5, 5, 4, 0,
                    // Left face (vertices 0,3,7,4) - back left, front left, top front left, top back left
                    0, 3, 7, 7, 4, 0,
                    // Right face (vertices 1,2,6,5) - back right, front right, top front right, top back right
                    1, 2, 6, 6, 5, 1,
                    // Bottom face (vertices 0,1,2,3) - all bottom vertices
                    0, 1, 2, 2, 3, 0, // Top face (vertices 4,5,6,7) - all top vertices
                    4, 5, 6, 6, 7, 4,
                ];

                // Add vertices for each triangle in the solid box
                for &index in &face_indices {
                    all_box_vertices.push(vertices[index]);
                }

                // Add edges for wireframe using MCP server vertex ordering
                let edge_indices = [
                    // Bottom face edges (0=back left, 1=back right, 2=front right, 3=front left)
                    0, 1, 1, 2, 2, 3, 3, 0,
                    // Top face edges (4=back left, 5=back right, 6=front right, 7=front left)
                    4, 5, 5, 6, 6, 7, 7, 4, // Vertical edges connecting bottom to top
                    0, 4, 1, 5, 2, 6, 3, 7,
                ];

                // Add vertices for each edge in the wireframe
                for &index in &edge_indices {
                    all_wireframe_vertices.push(wireframe_vertices[index]);
                }

                // Create arrow from center to front face center and extend outward
                // Calculate bbox center
                let mut center = Vector3::new(0.0, 0.0, 0.0);
                for point in &obj.aligned_bbox {
                    center.x += point[0];
                    center.y += point[1];
                    center.z += point[2];
                }
                center /= 8.0;

                // Convert bbox points to Vector3 for easier math
                let bbox_points: Vec<Vector3<f32>> = obj
                    .aligned_bbox
                    .iter()
                    .map(|p| Vector3::new(p[0], p[1], p[2]))
                    .collect();

                // Calculate centers of all 6 faces to find the best "front" direction
                let front_face_center =
                    (bbox_points[2] + bbox_points[3] + bbox_points[6] + bbox_points[7]) / 4.0; // front vertices
                let back_face_center =
                    (bbox_points[0] + bbox_points[1] + bbox_points[4] + bbox_points[5]) / 4.0; // back vertices
                let right_face_center =
                    (bbox_points[1] + bbox_points[2] + bbox_points[5] + bbox_points[6]) / 4.0; // right vertices
                let left_face_center =
                    (bbox_points[0] + bbox_points[3] + bbox_points[4] + bbox_points[7]) / 4.0; // left vertices
                let top_face_center =
                    (bbox_points[4] + bbox_points[5] + bbox_points[6] + bbox_points[7]) / 4.0; // top vertices
                let bottom_face_center =
                    (bbox_points[0] + bbox_points[1] + bbox_points[2] + bbox_points[3]) / 4.0; // bottom vertices

                // Calculate direction vectors for each face
                let face_directions = [
                    ("front", front_face_center - center),
                    ("back", back_face_center - center),
                    ("right", right_face_center - center),
                    ("left", left_face_center - center),
                    ("top", top_face_center - center),
                    ("bottom", bottom_face_center - center),
                ];

                // Find the face that gives the most reasonable viewing direction
                let mut best_direction = front_face_center - center;
                let mut best_face_name = "front";
                let mut best_score = 0.0;

                for (face_name, direction) in &face_directions {
                    let length = direction.magnitude();
                    if length > 0.001 {
                        let normalized = direction.normalize();
                        // Score based on: reasonable horizontal component and good distance
                        let horizontal_component =
                            (normalized.x.abs() + normalized.z.abs()).max(0.3);
                        let vertical_penalty = if normalized.y.abs() > 0.9 { 0.3 } else { 1.0 };
                        let score = length * horizontal_component * vertical_penalty;

                        if score > best_score {
                            best_score = score;
                            best_direction = *direction;
                            best_face_name = face_name;
                        }
                    }
                }

                log::info!("  Using face '{}' as front direction", best_face_name);
                let smart_direction_vec = best_direction;

                // Also get the MCP-provided normal vector if available
                let mcp_normal_vec = if let Some(normal) = &obj.normal_vector {
                    Vector3::new(normal[0], normal[1], normal[2])
                } else {
                    Vector3::new(0.0, 0.0, 0.0) // Zero vector if not provided
                };

                log::info!(
                    "  MCP provided normal vector: ({:.3}, {:.3}, {:.3})",
                    mcp_normal_vec.x,
                    mcp_normal_vec.y,
                    mcp_normal_vec.z
                );

                // Use the smart direction for the main logic
                let direction_vec = smart_direction_vec;
                log::info!(
                    "  Raw direction vector: ({:.3}, {:.3}, {:.3})",
                    direction_vec.x,
                    direction_vec.y,
                    direction_vec.z
                );

                // If the direction is too small, use a default direction
                let mut front_direction = if direction_vec.magnitude() < 0.001 {
                    log::info!("  Direction vector too small, using default +Z direction");
                    Vector3::new(0.0, 0.0, 1.0)
                } else {
                    direction_vec.normalize()
                };

                // Avoid camera singularity: if arrow points too vertically, add slight horizontal offset
                if front_direction.y.abs() > 0.95 {
                    // Add a small horizontal component to avoid pure vertical look
                    front_direction.x += 0.3;
                    front_direction.z += 0.3;
                    front_direction = front_direction.normalize();
                    log::info!(
                        "  Adjusted direction to avoid camera singularity: ({:.3}, {:.3}, {:.3})",
                        front_direction.x,
                        front_direction.y,
                        front_direction.z
                    );
                }

                // Calculate object dimensions for reasonable distance
                let width = (Vector3::new(
                    obj.aligned_bbox[2][0],
                    obj.aligned_bbox[2][1],
                    obj.aligned_bbox[2][2],
                ) - Vector3::new(
                    obj.aligned_bbox[3][0],
                    obj.aligned_bbox[3][1],
                    obj.aligned_bbox[3][2],
                ))
                .magnitude();
                let height = (Vector3::new(
                    obj.aligned_bbox[7][0],
                    obj.aligned_bbox[7][1],
                    obj.aligned_bbox[7][2],
                ) - Vector3::new(
                    obj.aligned_bbox[3][0],
                    obj.aligned_bbox[3][1],
                    obj.aligned_bbox[3][2],
                ))
                .magnitude();
                let depth = (Vector3::new(
                    obj.aligned_bbox[0][0],
                    obj.aligned_bbox[0][1],
                    obj.aligned_bbox[0][2],
                ) - Vector3::new(
                    obj.aligned_bbox[3][0],
                    obj.aligned_bbox[3][1],
                    obj.aligned_bbox[3][2],
                ))
                .magnitude();
                let max_dimension = width.max(depth).max(height);

                // Make arrow much longer and more visible for better camera positioning
                let arrow_length = (max_dimension * 6.5).max(10.0); // Optimal distance for camera positioning
                let arrow_end = center + front_direction * arrow_length;

                // Log the arrow information
                log::info!("ARROW INFO for object '{}' (index {}):", obj.name, i);
                log::info!(
                    "  Bbox center: ({:.3}, {:.3}, {:.3})",
                    center.x,
                    center.y,
                    center.z
                );
                log::info!(
                    "  Selected face center: ({:.3}, {:.3}, {:.3})",
                    (center + direction_vec).x,
                    (center + direction_vec).y,
                    (center + direction_vec).z
                );
                log::info!(
                    "  Front direction: ({:.3}, {:.3}, {:.3})",
                    front_direction.x,
                    front_direction.y,
                    front_direction.z
                );
                log::info!(
                    "  Dimensions: W={:.3}, D={:.3}, H={:.3}, Max={:.3}",
                    width,
                    depth,
                    height,
                    max_dimension
                );
                log::info!("  Arrow length: {:.3}", arrow_length);
                log::info!(
                    "  Arrow endpoint: ({:.3}, {:.3}, {:.3})",
                    arrow_end.x,
                    arrow_end.y,
                    arrow_end.z
                );
                log::info!(
                    "  Suggested camera position: ({:.3}, {:.3}, {:.3})",
                    arrow_end.x,
                    arrow_end.y,
                    arrow_end.z
                );
                log::info!(
                    "  Suggested camera target: ({:.3}, {:.3}, {:.3})",
                    center.x,
                    center.y,
                    center.z
                );

                // Add yellow arrow for MCP normal vector if provided
                if mcp_normal_vec.magnitude() > 0.001 {
                    // Define arrowhead size for yellow arrows
                    let arrowhead_size = (max_dimension * 0.5).max(1.0); // 50% of object size, minimum 1.0 unit
                    let mcp_arrow_color = Vector4::new(1.0, 1.0, 0.0, 1.0); // Bright yellow, fully opaque
                    let mcp_arrow_length = arrow_length * 0.8; // Slightly shorter than the green arrow
                    let mcp_normalized = mcp_normal_vec.normalize();
                    let mcp_arrow_end = center + mcp_normalized * mcp_arrow_length;

                    log::info!(
                        "  Adding yellow MCP normal arrow from center to ({:.3}, {:.3}, {:.3})",
                        mcp_arrow_end.x,
                        mcp_arrow_end.y,
                        mcp_arrow_end.z
                    );

                    // Main MCP arrow line
                    all_arrow_vertices.push(ArrowVertex {
                        position: center,
                        color: mcp_arrow_color,
                    });
                    all_arrow_vertices.push(ArrowVertex {
                        position: mcp_arrow_end,
                        color: mcp_arrow_color,
                    });

                    // MCP arrowhead
                    let mcp_arrowhead_size = arrowhead_size * 0.8; // Slightly smaller
                    let mcp_perp1 =
                        Vector3::new(-mcp_normalized.z, 0.0, mcp_normalized.x).normalize();
                    let mcp_perp2 = Vector3::new(0.0, 1.0, 0.0)
                        .cross(mcp_normalized)
                        .normalize();

                    // MCP arrowhead line 1
                    let mcp_arrowhead_point1 = mcp_arrow_end - mcp_normalized * mcp_arrowhead_size
                        + mcp_perp1 * mcp_arrowhead_size * 0.5;
                    all_arrow_vertices.push(ArrowVertex {
                        position: mcp_arrow_end,
                        color: mcp_arrow_color,
                    });
                    all_arrow_vertices.push(ArrowVertex {
                        position: mcp_arrowhead_point1,
                        color: mcp_arrow_color,
                    });

                    // MCP arrowhead line 2
                    let mcp_arrowhead_point2 = mcp_arrow_end - mcp_normalized * mcp_arrowhead_size
                        + mcp_perp2 * mcp_arrowhead_size * 0.5;
                    all_arrow_vertices.push(ArrowVertex {
                        position: mcp_arrow_end,
                        color: mcp_arrow_color,
                    });
                    all_arrow_vertices.push(ArrowVertex {
                        position: mcp_arrowhead_point2,
                        color: mcp_arrow_color,
                    });
                }
            } else {
                log::warn!(
                    "Object {} has {} vertices, expected 8. Skipping.",
                    obj.name,
                    obj.aligned_bbox.len()
                );
            }
        }

        // Create buffers with exact vertex data
        if !all_box_vertices.is_empty() {
            self.box_vertex_count = all_box_vertices.len() as u32;
            self.box_instances_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("highlight box vertices"),
                    contents: bytemuck::cast_slice(&all_box_vertices),
                    usage: BufferUsages::VERTEX,
                },
            ));
        } else {
            self.box_vertex_count = 0;
            self.box_instances_buffer = None;
        }

        if !all_wireframe_vertices.is_empty() {
            self.wireframe_vertex_count = all_wireframe_vertices.len() as u32;
            self.wireframe_instances_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("highlight wireframe vertices"),
                    contents: bytemuck::cast_slice(&all_wireframe_vertices),
                    usage: BufferUsages::VERTEX,
                },
            ));
        } else {
            self.wireframe_vertex_count = 0;
            self.wireframe_instances_buffer = None;
        }

        // Create arrow buffer
        if !all_arrow_vertices.is_empty() {
            self.arrow_vertex_count = all_arrow_vertices.len() as u32;
            log::info!(
                "Creating arrow buffer with {} vertices",
                self.arrow_vertex_count
            );
            self.arrow_vertex_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("highlight arrow vertices"),
                    contents: bytemuck::cast_slice(&all_arrow_vertices),
                    usage: BufferUsages::VERTEX,
                },
            ));
        } else {
            log::info!("No arrow vertices to create buffer");
            self.arrow_vertex_count = 0;
            self.arrow_vertex_buffer = None;
        }
    }

    fn update_path_geometry(&mut self, device: &wgpu::Device) {
        if let Some(path) = &self.highlighted_path {
            if path.waypoints.is_empty() {
                self.path_vertex_buffer = None;
                self.path_vertex_count = 0;
                return;
            }

            let mut vertices = Vec::new();
            let path_color = Vector4::new(0.9, 0.9, 0.3, 1.0); // Bright light yellow, fully opaque

            // Create line segments between waypoints
            for waypoint in &path.waypoints {
                vertices.push(PathVertex {
                    position: Vector3::new(waypoint[0], waypoint[1], waypoint[2]),
                    color: path_color,
                });
            }

            self.path_vertex_count = vertices.len() as u32;
            self.path_vertex_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("highlight path vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: BufferUsages::VERTEX,
                },
            ));
        } else {
            self.path_vertex_buffer = None;
            self.path_vertex_count = 0;
        }
    }

    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        camera: &'rpass UniformBuffer<CameraUniform>,
    ) {
        // Render solid semi-transparent bounding boxes first using exact vertices
        if let Some(vertex_buffer) = &self.box_instances_buffer {
            if self.box_vertex_count > 0 {
                render_pass.set_pipeline(&self.box_pipeline);
                render_pass.set_bind_group(0, camera.bind_group(), &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.draw(0..self.box_vertex_count, 0..1);
            }
        }

        // Render wireframe edges on top using exact vertices
        if let Some(wireframe_buffer) = &self.wireframe_instances_buffer {
            if self.wireframe_vertex_count > 0 {
                render_pass.set_pipeline(&self.wireframe_pipeline);
                render_pass.set_bind_group(0, camera.bind_group(), &[]);
                render_pass.set_vertex_buffer(0, wireframe_buffer.slice(..));
                render_pass.draw(0..self.wireframe_vertex_count, 0..1);
            }
        }

        // Render path
        if let Some(path_buffer) = &self.path_vertex_buffer {
            if self.path_vertex_count > 1 {
                render_pass.set_pipeline(&self.path_pipeline);
                render_pass.set_bind_group(0, camera.bind_group(), &[]);
                render_pass.set_vertex_buffer(0, path_buffer.slice(..));
                render_pass.draw(0..self.path_vertex_count, 0..1);
            }
        }

        // Render arrows (yellow normal vector indicators)
        if let Some(arrow_buffer) = &self.arrow_vertex_buffer {
            if self.arrow_vertex_count > 0 {
                render_pass.set_pipeline(&self.arrow_pipeline);
                render_pass.set_bind_group(0, camera.bind_group(), &[]);
                render_pass.set_vertex_buffer(0, arrow_buffer.slice(..));
                render_pass.draw(0..self.arrow_vertex_count, 0..1);
            }
        }
    }

    pub fn get_first_object_position(&self) -> Option<Point3<f32>> {
        self.highlighted_objects.first().map(|obj| {
            if obj.aligned_bbox.len() >= 8 {
                // Calculate center from all 8 points
                let mut center = [0.0, 0.0, 0.0];
                for point in &obj.aligned_bbox {
                    center[0] += point[0];
                    center[1] += point[1];
                    center[2] += point[2];
                }
                center[0] /= 8.0;
                center[1] /= 8.0;
                center[2] /= 8.0;
                Point3::new(center[0], center[1], center[2])
            } else {
                Point3::new(0.0, 0.0, 0.0)
            }
        })
    }

    /// Get optimal camera viewing position for the first object based on its orientation
    pub fn get_first_object_viewing_info(&self) -> Option<(Point3<f32>, Point3<f32>, f32)> {
        self.highlighted_objects.first().and_then(|obj| {
            if obj.aligned_bbox.len() >= 8 {
                // Calculate center
                let mut center = Vector3::new(0.0, 0.0, 0.0);
                for point in &obj.aligned_bbox {
                    center.x += point[0];
                    center.y += point[1];
                    center.z += point[2];
                }
                center /= 8.0;

                // Convert bbox points to Vector3 for easier math
                let bbox_points: Vec<Vector3<f32>> = obj
                    .aligned_bbox
                    .iter()
                    .map(|p| Vector3::new(p[0], p[1], p[2]))
                    .collect();

                // Calculate object's local coordinate system from the bounding box
                // MCP server vertex ordering (VERIFIED):
                // 0: bottom back left,  1: bottom back right,  2: bottom front right, 3: bottom front left
                // 4: top back left,     5: top back right,     6: top front right,    7: top front left

                // Calculate object dimensions first
                let width = (bbox_points[2] - bbox_points[3]).magnitude(); // front face width (front right - front left)
                let height = (bbox_points[7] - bbox_points[3]).magnitude(); // object height (top front left - bottom front left)
                let depth = (bbox_points[0] - bbox_points[3]).magnitude(); // depth (back left - front left)
                let max_dimension = width.max(depth).max(height);

                // Calculate centers of all 6 faces to find the best "front" direction
                let front_face_center =
                    (bbox_points[2] + bbox_points[3] + bbox_points[6] + bbox_points[7]) / 4.0; // front vertices
                let back_face_center =
                    (bbox_points[0] + bbox_points[1] + bbox_points[4] + bbox_points[5]) / 4.0; // back vertices
                let right_face_center =
                    (bbox_points[1] + bbox_points[2] + bbox_points[5] + bbox_points[6]) / 4.0; // right vertices
                let left_face_center =
                    (bbox_points[0] + bbox_points[3] + bbox_points[4] + bbox_points[7]) / 4.0; // left vertices
                let top_face_center =
                    (bbox_points[4] + bbox_points[5] + bbox_points[6] + bbox_points[7]) / 4.0; // top vertices
                let bottom_face_center =
                    (bbox_points[0] + bbox_points[1] + bbox_points[2] + bbox_points[3]) / 4.0; // bottom vertices

                // Calculate direction vectors for each face
                let face_directions = [
                    ("front", front_face_center - center),
                    ("back", back_face_center - center),
                    ("right", right_face_center - center),
                    ("left", left_face_center - center),
                    ("top", top_face_center - center),
                    ("bottom", bottom_face_center - center),
                ];

                // Find the face that gives the most reasonable viewing direction
                // Prefer faces that point outward and away from the center, and avoid purely vertical directions
                let mut best_direction = front_face_center - center;
                let mut best_face_name = "front";
                let mut best_score = 0.0;

                for (face_name, direction) in &face_directions {
                    let length = direction.magnitude();
                    if length > 0.001 {
                        let normalized = direction.normalize();
                        // Score based on: reasonable horizontal component (not too vertical) and good distance
                        let horizontal_component =
                            (normalized.x.abs() + normalized.z.abs()).max(0.3);
                        let vertical_penalty = if normalized.y.abs() > 0.9 { 0.3 } else { 1.0 };
                        let score = length * horizontal_component * vertical_penalty;

                        log::info!(
                            "  Face {}: direction ({:.3}, {:.3}, {:.3}), length {:.3}, score {:.3}",
                            face_name,
                            normalized.x,
                            normalized.y,
                            normalized.z,
                            length,
                            score
                        );

                        if score > best_score {
                            best_score = score;
                            best_direction = *direction;
                            best_face_name = face_name;
                        }
                    }
                }

                log::info!(
                    "  Selected best face: {} with score {:.3}",
                    best_face_name,
                    best_score
                );

                // Use MCP normal vector if available, otherwise use smart selection
                let direction_vec = if let Some(normal) = &obj.normal_vector {
                    let mcp_normal = Vector3::new(normal[0], normal[1], normal[2]);
                    if mcp_normal.magnitude() > 0.001 {
                        log::info!("  Using MCP-provided normal vector for camera positioning");
                        mcp_normal
                    } else {
                        log::info!("  MCP normal vector is zero, using smart-selected direction");
                        best_direction
                    }
                } else {
                    log::info!("  No MCP normal vector provided, using smart-selected direction");
                    best_direction
                };
                let mut front_direction = if direction_vec.magnitude() < 0.001 {
                    Vector3::new(0.0, 0.0, 1.0) // default direction
                } else {
                    direction_vec.normalize()
                };

                // Avoid camera singularity: if arrow points too vertically, add slight horizontal offset
                if front_direction.y.abs() > 0.95 {
                    // Add a small horizontal component to avoid pure vertical look
                    front_direction.x += 0.3;
                    front_direction.z += 0.3;
                    front_direction = front_direction.normalize();
                    log::info!(
                        "  Adjusted direction to avoid camera singularity: ({:.3}, {:.3}, {:.3})",
                        front_direction.x,
                        front_direction.y,
                        front_direction.z
                    );
                }

                // Calculate arrow endpoint (same as in arrow creation)
                let arrow_length = (max_dimension * 6.5).max(10.0); // Optimal distance for framing
                let arrow_end = center + front_direction * arrow_length;

                // Use the arrow endpoint directly as camera position
                let camera_position = Point3::from_vec(arrow_end);

                let is_vertical_arrow = front_direction.y.abs() > 0.8;
                // Apply target compensation based on observed offset patterns
                // Fine-tuned based on latest View Information results
                let target_offset = Vector3::new(4.4, -2.0, 3.7); // Refined offset pattern
                let compensated_target = center + target_offset;

                log::info!("Target-compensated camera positioning:");
                log::info!(
                    "  Original object center: ({:.3}, {:.3}, {:.3})",
                    center.x,
                    center.y,
                    center.z
                );
                log::info!(
                    "  Compensated target: ({:.3}, {:.3}, {:.3})",
                    compensated_target.x,
                    compensated_target.y,
                    compensated_target.z
                );
                log::info!(
                    "  Arrow direction: ({:.3}, {:.3}, {:.3})",
                    front_direction.x,
                    front_direction.y,
                    front_direction.z
                );
                log::info!(
                    "  Camera position: ({:.3}, {:.3}, {:.3})",
                    camera_position.x,
                    camera_position.y,
                    camera_position.z
                );
                log::info!(
                    "  Distance to object: {:.3}",
                    (Vector3::from(camera_position.to_vec()) - center).magnitude()
                );

                Some((
                    Point3::from_vec(compensated_target),
                    camera_position,
                    max_dimension,
                ))
            } else {
                None
            }
        })
    }

    pub fn get_path_start_position(&self) -> Option<Point3<f32>> {
        self.highlighted_path
            .as_ref()
            .and_then(|path| path.waypoints.first())
            .map(|waypoint| Point3::new(waypoint[0], waypoint[1], waypoint[2]))
    }
}
