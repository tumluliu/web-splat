struct CameraUniform {
    view_matrix: mat4x4<f32>,
    view_inv_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    proj_inv_matrix: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform vertex position directly (no model matrix needed)
    let world_position = vec4<f32>(input.position, 1.0);
    let view_position = camera.view_matrix * world_position;
    
    out.clip_position = camera.proj_matrix * view_position;
    out.color = input.color;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
} 