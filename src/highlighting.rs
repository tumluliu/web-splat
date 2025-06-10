use crate::chat::{SceneObject, ScenePath};
use crate::renderer::CameraUniform;
use crate::uniform::UniformBuffer;
use cgmath::{Matrix4, Point3, Vector3, Vector4};
use wgpu::{
    include_wgsl, util::DeviceExt, BufferUsages, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexStepMode,
};

pub struct HighlightRenderer {
    box_pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    path_pipeline: wgpu::RenderPipeline,
    box_vertex_buffer: Option<wgpu::Buffer>,
    box_index_buffer: Option<wgpu::Buffer>,
    wireframe_vertex_buffer: Option<wgpu::Buffer>,
    wireframe_index_buffer: Option<wgpu::Buffer>,
    path_vertex_buffer: Option<wgpu::Buffer>,
    highlighted_objects: Vec<SceneObject>,
    highlighted_path: Option<ScenePath>,
    box_instances_buffer: Option<wgpu::Buffer>,
    wireframe_instances_buffer: Option<wgpu::Buffer>,
    path_vertex_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoxInstance {
    model_matrix: Matrix4<f32>,
    color: Vector4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PathVertex {
    position: Vector3<f32>,
    color: Vector4<f32>,
}

impl HighlightRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let box_pipeline = Self::create_box_pipeline(device, target_format);
        let wireframe_pipeline = Self::create_wireframe_pipeline(device, target_format);
        let path_pipeline = Self::create_path_pipeline(device, target_format);

        // Create static box geometry (unit cube wireframe)
        let (box_vertex_buffer, box_index_buffer) = Self::create_box_geometry(device);
        let (wireframe_vertex_buffer, wireframe_index_buffer) =
            Self::create_wireframe_geometry(device);

        Self {
            box_pipeline,
            wireframe_pipeline,
            path_pipeline,
            box_vertex_buffer: Some(box_vertex_buffer),
            box_index_buffer: Some(box_index_buffer),
            wireframe_vertex_buffer: Some(wireframe_vertex_buffer),
            wireframe_index_buffer: Some(wireframe_index_buffer),
            path_vertex_buffer: None,
            highlighted_objects: Vec::new(),
            highlighted_path: None,
            box_instances_buffer: None,
            wireframe_instances_buffer: None,
            path_vertex_count: 0,
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

        let shader = device.create_shader_module(include_wgsl!("shaders/highlight_box.wgsl"));

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight box pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Vertex positions (unit cube corners)
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vector3<f32>>() as wgpu::BufferAddress,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &[VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: VertexFormat::Float32x3,
                        }],
                    },
                    // Instance data (transformation matrix + color)
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<BoxInstance>() as wgpu::BufferAddress,
                        step_mode: VertexStepMode::Instance,
                        attributes: &[
                            // Matrix column 0
                            VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: VertexFormat::Float32x4,
                            },
                            // Matrix column 1
                            VertexAttribute {
                                offset: std::mem::size_of::<Vector4<f32>>() as wgpu::BufferAddress,
                                shader_location: 2,
                                format: VertexFormat::Float32x4,
                            },
                            // Matrix column 2
                            VertexAttribute {
                                offset: 2 * std::mem::size_of::<Vector4<f32>>()
                                    as wgpu::BufferAddress,
                                shader_location: 3,
                                format: VertexFormat::Float32x4,
                            },
                            // Matrix column 3
                            VertexAttribute {
                                offset: 3 * std::mem::size_of::<Vector4<f32>>()
                                    as wgpu::BufferAddress,
                                shader_location: 4,
                                format: VertexFormat::Float32x4,
                            },
                            // Color
                            VertexAttribute {
                                offset: 4 * std::mem::size_of::<Vector4<f32>>()
                                    as wgpu::BufferAddress,
                                shader_location: 5,
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

        let shader = device.create_shader_module(include_wgsl!("shaders/highlight_box.wgsl"));

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight wireframe pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vector3<f32>>() as wgpu::BufferAddress,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &[VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: VertexFormat::Float32x3,
                        }],
                    },
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<BoxInstance>() as wgpu::BufferAddress,
                        step_mode: VertexStepMode::Instance,
                        attributes: &[
                            VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: VertexFormat::Float32x4,
                            },
                            VertexAttribute {
                                offset: std::mem::size_of::<Vector4<f32>>() as wgpu::BufferAddress,
                                shader_location: 2,
                                format: VertexFormat::Float32x4,
                            },
                            VertexAttribute {
                                offset: 2 * std::mem::size_of::<Vector4<f32>>()
                                    as wgpu::BufferAddress,
                                shader_location: 3,
                                format: VertexFormat::Float32x4,
                            },
                            VertexAttribute {
                                offset: 3 * std::mem::size_of::<Vector4<f32>>()
                                    as wgpu::BufferAddress,
                                shader_location: 4,
                                format: VertexFormat::Float32x4,
                            },
                            VertexAttribute {
                                offset: 4 * std::mem::size_of::<Vector4<f32>>()
                                    as wgpu::BufferAddress,
                                shader_location: 5,
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
        log::info!("Setting {} highlighted objects", objects.len());
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
        self.path_vertex_count = 0;
    }

    fn update_box_instances(&mut self, device: &wgpu::Device) {
        if self.highlighted_objects.is_empty() {
            self.box_instances_buffer = None;
            self.wireframe_instances_buffer = None;
            return;
        }

        log::info!(
            "Creating bounding boxes for {} objects",
            self.highlighted_objects.len()
        );

        let (box_instances, wireframe_instances): (Vec<BoxInstance>, Vec<BoxInstance>) = self
            .highlighted_objects
            .iter()
            .enumerate()
            .map(|(i, obj)| {
                let position = Point3::new(obj.position[0], obj.position[1], obj.position[2]);

                // Get bounding box size from attributes or use defaults
                // Use very large default sizes to make bounding boxes highly visible
                let width = obj
                    .attributes
                    .get("width")
                    .and_then(|w| w.parse::<f32>().ok())
                    .unwrap_or(5.0); // Very large for visibility
                let height = obj
                    .attributes
                    .get("height")
                    .and_then(|h| h.parse::<f32>().ok())
                    .unwrap_or(3.0); // Very large for visibility
                let depth = obj
                    .attributes
                    .get("depth")
                    .and_then(|d| d.parse::<f32>().ok())
                    .unwrap_or(5.0); // Very large for visibility

                // Create transformation matrix
                let scale_matrix = Matrix4::from_nonuniform_scale(width, height, depth);
                let translation_matrix =
                    Matrix4::from_translation(Vector3::new(position.x, position.y, position.z));
                let model_matrix = translation_matrix * scale_matrix;

                // Cycle through colors for different objects - bright, moderately transparent colors for solid boxes
                let solid_colors = [
                    Vector4::new(1.0, 0.0, 0.0, 0.25), // Red with moderate transparency
                    Vector4::new(0.0, 1.0, 0.0, 0.25), // Green with moderate transparency
                    Vector4::new(0.0, 0.5, 1.0, 0.25), // Blue with moderate transparency
                    Vector4::new(1.0, 1.0, 0.0, 0.25), // Yellow with moderate transparency
                    Vector4::new(1.0, 0.0, 1.0, 0.25), // Magenta with moderate transparency
                    Vector4::new(0.0, 1.0, 1.0, 0.25), // Cyan with moderate transparency
                ];

                // Bright, highly opaque colors for wireframe edges to make them thick and prominent
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

                let box_instance = BoxInstance {
                    model_matrix,
                    color: solid_color,
                };

                let wireframe_instance = BoxInstance {
                    model_matrix,
                    color: wireframe_color,
                };

                (box_instance, wireframe_instance)
            })
            .unzip();

        self.box_instances_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("highlight box instances"),
                contents: bytemuck::cast_slice(&box_instances),
                usage: BufferUsages::VERTEX,
            },
        ));

        self.wireframe_instances_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("highlight wireframe instances"),
                contents: bytemuck::cast_slice(&wireframe_instances),
                usage: BufferUsages::VERTEX,
            },
        ));
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
        // Render solid semi-transparent bounding boxes first
        if let (Some(vertex_buffer), Some(index_buffer), Some(instances_buffer)) = (
            &self.box_vertex_buffer,
            &self.box_index_buffer,
            &self.box_instances_buffer,
        ) {
            render_pass.set_pipeline(&self.box_pipeline);
            render_pass.set_bind_group(0, camera.bind_group(), &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, instances_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..36, 0, 0..self.highlighted_objects.len() as u32);
            // 36 indices for solid cube (6 faces * 2 triangles * 3 vertices)
        }

        // Render wireframe edges on top for thicker edge definition
        if let (
            Some(wireframe_vertex_buffer),
            Some(wireframe_index_buffer),
            Some(instances_buffer),
        ) = (
            &self.wireframe_vertex_buffer,
            &self.wireframe_index_buffer,
            &self.wireframe_instances_buffer,
        ) {
            render_pass.set_pipeline(&self.wireframe_pipeline);
            render_pass.set_bind_group(0, camera.bind_group(), &[]);
            render_pass.set_vertex_buffer(0, wireframe_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, instances_buffer.slice(..));
            render_pass
                .set_index_buffer(wireframe_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..24, 0, 0..self.highlighted_objects.len() as u32);
            // 24 indices for wireframe edges
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
    }

    pub fn get_first_object_position(&self) -> Option<Point3<f32>> {
        self.highlighted_objects
            .first()
            .map(|obj| Point3::new(obj.position[0], obj.position[1], obj.position[2]))
    }

    pub fn get_path_start_position(&self) -> Option<Point3<f32>> {
        self.highlighted_path
            .as_ref()
            .and_then(|path| path.waypoints.first())
            .map(|waypoint| Point3::new(waypoint[0], waypoint[1], waypoint[2]))
    }
}
