use std::ops::RangeInclusive;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use crate::chat::McpResponse;
#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::DEFAULT_KERNEL_SIZE;
use crate::{SceneCamera, Split, WindowContext};
use cgmath::{Euler, Matrix3, Quaternion};
#[cfg(not(target_arch = "wasm32"))]
use egui::Vec2b;

#[cfg(target_arch = "wasm32")]
use egui::{Align2, Vec2};

use egui::{emath::Numeric, Color32, RichText};

#[cfg(not(target_arch = "wasm32"))]
use egui_plot::{Legend, PlotPoints};

pub(crate) fn ui(state: &mut WindowContext) -> (bool, Option<String>) {
    let ctx = state.ui_renderer.winit.egui_ctx();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(stopwatch) = state.stopwatch.as_mut() {
        let durations = pollster::block_on(
            stopwatch.take_measurements(&state.wgpu_context.device, &state.wgpu_context.queue),
        );
        state.history.push((
            *durations.get("preprocess").unwrap_or(&Duration::ZERO),
            *durations.get("sorting").unwrap_or(&Duration::ZERO),
            *durations.get("rasterization").unwrap_or(&Duration::ZERO),
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    let num_drawn = pollster::block_on(
        state
            .renderer
            .num_visible_points(&state.wgpu_context.device, &state.wgpu_context.queue),
    );

    #[cfg(not(target_arch = "wasm32"))]
    egui::Window::new("Render Stats")
        .default_width(200.)
        .default_height(100.)
        .show(ctx, |ui| {
            use egui::TextStyle;
            egui::Grid::new("timing").num_columns(2).show(ui, |ui| {
                ui.colored_label(egui::Color32::WHITE, "FPS");
                ui.label(format!("{:}", state.fps as u32));
                ui.end_row();
                ui.colored_label(egui::Color32::WHITE, "Visible points");
                ui.label(format!(
                    "{:} ({:.2}%)",
                    format_thousands(num_drawn),
                    (num_drawn as f32 / state.pc.num_points() as f32) * 100.
                ));
            });
            let history = state.history.to_vec();
            let pre: Vec<f32> = history.iter().map(|v| v.0.as_secs_f32() * 1000.).collect();
            let sort: Vec<f32> = history.iter().map(|v| v.1.as_secs_f32() * 1000.).collect();
            let rast: Vec<f32> = history.iter().map(|v| v.2.as_secs_f32() * 1000.).collect();

            ui.label("Frame times (ms):");
            egui_plot::Plot::new("frame times")
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .y_axis_min_width(1.0)
                .y_axis_label("ms")
                .auto_bounds(Vec2b::TRUE)
                .show_axes([false, true])
                .legend(
                    Legend::default()
                        .text_style(TextStyle::Body)
                        .background_alpha(1.)
                        .position(egui_plot::Corner::LeftBottom),
                )
                .show(ui, |ui| {
                    let line = egui_plot::Line::new("preprocess", PlotPoints::from_ys_f32(&pre));
                    ui.line(line);
                    let line = egui_plot::Line::new("sorting", PlotPoints::from_ys_f32(&sort));
                    ui.line(line);
                    let line = egui_plot::Line::new("rasterize", PlotPoints::from_ys_f32(&rast));
                    ui.line(line);
                });
        });

    egui::Window::new("⚙ Render Settings").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Gaussian Scaling");
                ui.add(
                    egui::DragValue::new(&mut state.splatting_args.gaussian_scaling)
                        .range((1e-4)..=1.)
                        .clamp_existing_to_range(true)
                        .speed(1e-2),
                );
                ui.end_row();
                ui.label("Directional Color");
                let mut dir_color = state.splatting_args.max_sh_deg > 0;
                ui.add_enabled(
                    state.pc.sh_deg() > 0,
                    egui::Checkbox::new(&mut dir_color, ""),
                );
                state.splatting_args.max_sh_deg = if dir_color { state.pc.sh_deg() } else { 0 };

                ui.end_row();
                let enable_bg = !state.splatting_args.show_env_map && !state.display.has_env_map();
                ui.add_enabled(enable_bg, egui::Label::new("Background Color"));
                let mut color = egui::Color32::from_rgba_premultiplied(
                    (state.splatting_args.background_color.r * 255.) as u8,
                    (state.splatting_args.background_color.g * 255.) as u8,
                    (state.splatting_args.background_color.b * 255.) as u8,
                    (state.splatting_args.background_color.a * 255.) as u8,
                );
                ui.add_enabled_ui(enable_bg, |ui| {
                    egui::color_picker::color_edit_button_srgba(
                        ui,
                        &mut color,
                        egui::color_picker::Alpha::BlendOrAdditive,
                    )
                });

                let color32 = color.to_normalized_gamma_f32();
                state.splatting_args.background_color.r = color32[0] as f64;
                state.splatting_args.background_color.g = color32[1] as f64;
                state.splatting_args.background_color.b = color32[2] as f64;
                state.splatting_args.background_color.a = color32[3] as f64;

                ui.end_row();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.label("Dilation Kernel Size");
                    optional_drag(
                        ui,
                        &mut state.splatting_args.kernel_size,
                        Some(0.0..=10.0),
                        Some(0.1),
                        Some(
                            state
                                .pc
                                .dilation_kernel_size()
                                .unwrap_or(DEFAULT_KERNEL_SIZE),
                        ),
                    );
                    ui.end_row();
                    ui.label("Mip Splatting");
                    optional_checkbox(
                        ui,
                        &mut state.splatting_args.mip_splatting,
                        state.pc.mip_splatting().unwrap_or(false),
                    );
                    ui.end_row();
                }
            });
    });

    // Debug Information Window
    egui::Window::new("🐛 Debug Info")
        .default_width(300.)
        .default_height(400.)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let camera = &state.splatting_args.camera;
                let camera_pos = camera.position;
                let camera_rot = camera.rotation;
                let projection = &camera.projection;

                // Convert quaternion to Euler angles for display
                let euler: Euler<cgmath::Rad<f32>> = Euler::from(camera_rot);
                let euler_degrees = Euler::new(
                    cgmath::Deg::from(euler.x),
                    cgmath::Deg::from(euler.y),
                    cgmath::Deg::from(euler.z),
                );

                ui.heading("Camera Information");
                egui::Grid::new("camera_debug")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Position X:");
                        ui.label(format!("{:.3}", camera_pos.x));
                        ui.end_row();

                        ui.strong("Position Y:");
                        ui.label(format!("{:.3}", camera_pos.y));
                        ui.end_row();

                        ui.strong("Position Z:");
                        ui.label(format!("{:.3}", camera_pos.z));
                        ui.end_row();

                        ui.strong("Rotation X (deg):");
                        ui.label(format!("{:.1}°", euler_degrees.x.0));
                        ui.end_row();

                        ui.strong("Rotation Y (deg):");
                        ui.label(format!("{:.1}°", euler_degrees.y.0));
                        ui.end_row();

                        ui.strong("Rotation Z (deg):");
                        ui.label(format!("{:.1}°", euler_degrees.z.0));
                        ui.end_row();

                        ui.strong("FOV X (deg):");
                        ui.label(format!("{:.1}°", cgmath::Deg::from(projection.fovx).0));
                        ui.end_row();

                        ui.strong("FOV Y (deg):");
                        ui.label(format!("{:.1}°", cgmath::Deg::from(projection.fovy).0));
                        ui.end_row();

                        ui.strong("Near Plane:");
                        ui.label(format!("{:.3}", projection.znear));
                        ui.end_row();

                        ui.strong("Far Plane:");
                        ui.label(format!("{:.1}", projection.zfar));
                        ui.end_row();
                    });

                ui.separator();

                // View center (controller center)
                ui.heading("View Information");
                egui::Grid::new("view_debug")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        let center = state.controller.center;
                        ui.strong("View Center X:");
                        ui.label(format!("{:.3}", center.x));
                        ui.end_row();

                        ui.strong("View Center Y:");
                        ui.label(format!("{:.3}", center.y));
                        ui.end_row();

                        ui.strong("View Center Z:");
                        ui.label(format!("{:.3}", center.z));
                        ui.end_row();

                        ui.strong("Viewport Width:");
                        ui.label(format!("{}", state.splatting_args.viewport.x));
                        ui.end_row();

                        ui.strong("Viewport Height:");
                        ui.label(format!("{}", state.splatting_args.viewport.y));
                        ui.end_row();
                    });

                ui.separator();

                // Performance Information
                ui.heading("Performance");
                egui::Grid::new("performance_debug")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("FPS:");
                        ui.label(format!("{:.1}", state.fps));
                        ui.end_row();

                        ui.strong("Frame Time (ms):");
                        ui.label(format!("{:.1}", 1000.0 / state.fps.max(0.1)));
                        ui.end_row();

                        ui.strong("Walltime (s):");
                        ui.label(format!(
                            "{:.1}",
                            state.splatting_args.walltime.as_secs_f32()
                        ));
                        ui.end_row();

                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            ui.strong("Visible Gaussians:");
                            ui.label(format_thousands(num_drawn));
                            ui.end_row();

                            ui.strong("Visible %:");
                            ui.label(format!(
                                "{:.1}%",
                                (num_drawn as f32 / state.pc.num_points() as f32) * 100.
                            ));
                            ui.end_row();
                        }
                    });

                ui.separator();

                // Scene Information
                ui.heading("Scene");
                egui::Grid::new("scene_debug")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Total Gaussians:");
                        ui.label(format_thousands(state.pc.num_points()));
                        ui.end_row();

                        ui.strong("SH Degree:");
                        ui.label(state.pc.sh_deg().to_string());
                        ui.end_row();

                        ui.strong("Max SH Degree:");
                        ui.label(state.splatting_args.max_sh_deg.to_string());
                        ui.end_row();

                        ui.strong("Gaussian Scaling:");
                        ui.label(format!("{:.3}", state.splatting_args.gaussian_scaling));
                        ui.end_row();

                        let bbox = state.pc.bbox();
                        ui.strong("Scene Center:");
                        ui.label(format!(
                            "({:.2}, {:.2}, {:.2})",
                            bbox.center().x,
                            bbox.center().y,
                            bbox.center().z
                        ));
                        ui.end_row();

                        ui.strong("Scene Radius:");
                        ui.label(format!("{:.2}", bbox.radius()));
                        ui.end_row();

                        if let Some(scene) = &state.scene {
                            ui.strong("Dataset Cameras:");
                            ui.label(scene.num_cameras().to_string());
                            ui.end_row();

                            ui.strong("Scene Extend:");
                            ui.label(format!("{:.2}", scene.extend()));
                            ui.end_row();
                        }

                        if let Some(current_view) = state.current_view {
                            ui.strong("Current View ID:");
                            ui.label(current_view.to_string());
                            ui.end_row();
                        }
                    });

                ui.separator();

                // Rendering Information
                ui.heading("Rendering");
                egui::Grid::new("rendering_debug")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Resolution:");
                        ui.label(format!(
                            "{}x{}",
                            state.splatting_args.resolution.x, state.splatting_args.resolution.y
                        ));
                        ui.end_row();

                        ui.strong("Scale Factor:");
                        ui.label(format!("{:.2}", state.scale_factor));
                        ui.end_row();

                        ui.strong("Show Env Map:");
                        ui.label(state.splatting_args.show_env_map.to_string());
                        ui.end_row();

                        ui.strong("Compressed:");
                        ui.label(state.pc.compressed().to_string());
                        ui.end_row();

                        if let Some(mip_splatting) = state.splatting_args.mip_splatting {
                            ui.strong("Mip Splatting:");
                            ui.label(mip_splatting.to_string());
                            ui.end_row();
                        }

                        if let Some(kernel_size) = state.splatting_args.kernel_size {
                            ui.strong("Kernel Size:");
                            ui.label(format!("{:.2}", kernel_size));
                            ui.end_row();
                        }
                    });
            });
        });

    let mut new_camera: Option<SetCamera> = None;
    #[allow(unused_mut)]
    let mut toggle_tracking_shot = false;
    egui::Window::new("ℹ Scene")
        .default_width(200.)
        .resizable(true)
        .default_height(100.)
        .show(ctx, |ui| {
            egui::Grid::new("scene info")
                .num_columns(2)
                .striped(false)
                .show(ui, |ui| {
                    ui.strong("Gaussians:");
                    ui.label(format_thousands(state.pc.num_points()));
                    ui.end_row();
                    ui.strong("SH Degree:");
                    ui.label(state.pc.sh_deg().to_string());
                    ui.end_row();
                    ui.strong("Compressed:");
                    ui.label(state.pc.compressed().to_string());
                    ui.end_row();
                    ui.strong("Mip Splatting:");
                    ui.label(
                        state
                            .pc
                            .mip_splatting()
                            .map(|v| v.to_string())
                            .unwrap_or("-".to_string()),
                    );
                    ui.end_row();
                    ui.strong("Dilation Kernel Size:");
                    ui.label(
                        state
                            .pc
                            .dilation_kernel_size()
                            .map(|v| v.to_string())
                            .unwrap_or("-".to_string()),
                    );
                    ui.end_row();
                    if let Some(path) = &state.pointcloud_file_path {
                        ui.strong("File:");
                        let text = path.to_string_lossy().to_string();

                        ui.add(egui::Label::new(
                            path.file_name().unwrap().to_string_lossy().to_string(),
                        ))
                        .on_hover_text(text);
                        ui.end_row();
                    }
                    ui.end_row();
                });

            if let Some(scene) = &state.scene {
                let nearest = scene.nearest_camera(state.splatting_args.camera.position, None);
                ui.separator();
                ui.collapsing("Dataset Images", |ui| {
                    egui::Grid::new("image info")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Images");
                            ui.label(scene.num_cameras().to_string());
                            ui.end_row();

                            ui.strong("Current View");

                            if let Some(c) = &mut state.current_view {
                                ui.horizontal(|ui| {
                                    let drag = ui.add(
                                        egui::DragValue::new(c)
                                            .range(0..=(scene.num_cameras().saturating_sub(1)))
                                            .clamp_existing_to_range(true),
                                    );
                                    if drag.changed() {
                                        new_camera = Some(SetCamera::ID(*c));
                                    }
                                    ui.label(scene.camera(*c as usize).unwrap().split.to_string());
                                });
                            } else {
                                ui.label("-");
                            }
                            if let Some(path) = &state.scene_file_path {
                                ui.end_row();
                                ui.strong("File:");
                                let text = path.to_string_lossy().to_string();

                                ui.add(egui::Label::new(
                                    path.file_name().unwrap().to_string_lossy().to_string(),
                                ))
                                .on_hover_text(text);
                            }
                        });

                    egui::ScrollArea::vertical()
                        .max_height(300.)
                        .show(ui, |ui| {
                            let cameras = scene.cameras(None);
                            let cameras2 = cameras.clone();
                            let curr_view = state.current_view;
                            egui::Grid::new("scene views grid")
                                .num_columns(4)
                                .striped(true)
                                .with_row_color(move |idx, _| {
                                    if let Some(view_id) = curr_view {
                                        if idx < cameras.len() && (&cameras)[idx].id == view_id {
                                            return Some(Color32::from_gray(64));
                                        }
                                    }
                                    return None;
                                })
                                .min_col_width(50.)
                                .show(ui, |ui| {
                                    let style = ui.style().clone();
                                    for c in cameras2 {
                                        ui.colored_label(
                                            style.visuals.strong_text_color(),
                                            c.id.to_string(),
                                        );
                                        ui.colored_label(
                                            match c.split {
                                                Split::Train => Color32::DARK_GREEN,
                                                Split::Test => Color32::LIGHT_GREEN,
                                            },
                                            c.split.to_string(),
                                        )
                                        .on_hover_text(
                                            RichText::new(format!(
                                                "{:#?}",
                                                Euler::from(Quaternion::from(Matrix3::from(
                                                    c.rotation
                                                )))
                                            )),
                                        );

                                        let resp =
                                            ui.add(egui::Label::new(c.img_name.clone()).truncate());
                                        if let Some(view_id) = curr_view {
                                            if c.id == view_id {
                                                resp.scroll_to_me(None);
                                            }
                                        }
                                        if ui.button("🎥").clicked() {
                                            new_camera = Some(SetCamera::ID(c.id));
                                        }
                                        ui.end_row();
                                    }
                                });
                        });
                    if let Some(nearest) = nearest {
                        ui.separator();
                        if ui.button(format!("Snap to closest ({nearest})")).clicked() {
                            new_camera = Some(SetCamera::ID(nearest));
                        }
                    }
                });
            }
        });

    #[cfg(target_arch = "wasm32")]
    egui::Window::new("🎮")
        .default_width(200.)
        .resizable(false)
        .default_height(100.)
        .default_open(false)
        .movable(false)
        .anchor(Align2::LEFT_BOTTOM, Vec2::new(10., -10.))
        .show(ctx, |ui| {
            egui::Grid::new("controls")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Camera");
                    ui.end_row();
                    ui.label("Rotate Camera");
                    ui.label("Left click + drag");
                    ui.end_row();

                    ui.label("Move Target/Center");
                    ui.label("Right click + drag");
                    ui.end_row();

                    ui.label("Tilt Camera");
                    ui.label("Alt + drag mouse");
                    ui.end_row();

                    ui.label("Zoom");
                    ui.label("Mouse wheel");
                    ui.end_row();

                    ui.label("Toggle UI");
                    ui.label("U");
                    ui.end_row();

                    ui.strong("Scene Views");
                    ui.end_row();
                    ui.label("Views 0-9");
                    ui.label("0-9");
                    ui.end_row();
                    ui.label("Random view");
                    ui.label("R");
                    ui.end_row();
                    ui.label("Next View");
                    ui.label("Page Up");
                    ui.end_row();
                    ui.label("Previous View");
                    ui.label("Page Down");
                    ui.end_row();
                    ui.label("Snap to nearest view");
                    ui.label("N");
                    ui.end_row();
                    ui.label("Start/Pause Tracking shot");
                    ui.label("T");
                    ui.end_row();
                });
        });

    // Chat UI - handle separately to avoid borrowing conflicts
    let (chat_message, new_input) = chat_ui(state, ctx);

    // Update chat input state
    state.chat_state.current_input = new_input;

    let requested_repaint = ctx.has_requested_repaint();

    if let Some(c) = new_camera {
        match c {
            SetCamera::ID(id) => state.set_scene_camera(id),
            SetCamera::Camera(c) => state.set_camera(c, Duration::from_millis(200)),
        }
    }
    if toggle_tracking_shot {
        if let Some((_animation, playing)) = &mut state.animation {
            *playing = !*playing;
        } else {
            state.start_tracking_shot();
        }
    }
    return (requested_repaint, chat_message);
}

enum SetCamera {
    ID(usize),
    #[allow(dead_code)]
    Camera(SceneCamera),
}

/// 212312321 -> 212.312.321
fn format_thousands(n: u32) -> String {
    let mut n = n;
    let mut result = String::new();
    while n > 0 {
        let rem = n % 1000;
        n /= 1000;
        if n > 0 {
            result = format!(".{:03}", rem) + &result;
        } else {
            result = rem.to_string() + &result;
        }
    }
    result
}

#[allow(unused)]
fn optional_drag<T: Numeric>(
    ui: &mut egui::Ui,
    opt: &mut Option<T>,
    range: Option<RangeInclusive<T>>,
    speed: Option<impl Into<f64>>,
    default: Option<T>,
) {
    let mut placeholder = default.unwrap_or(T::from_f64(0.));
    let mut drag = if let Some(ref mut val) = opt {
        egui_winit::egui::DragValue::new(val)
    } else {
        egui_winit::egui::DragValue::new(&mut placeholder).custom_formatter(|_, _| {
            if let Some(v) = default {
                format!("{:.2}", v.to_f64())
            } else {
                "—".into()
            }
        })
    };
    if let Some(range) = range {
        drag = drag.range(range).clamp_existing_to_range(true);
    }
    if let Some(speed) = speed {
        drag = drag.speed(speed);
    }
    let changed = ui.add(drag).changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(placeholder);
    }
}

#[allow(unused)]
fn optional_checkbox(ui: &mut egui::Ui, opt: &mut Option<bool>, default: bool) {
    let mut val = default;
    let checkbox = if let Some(ref mut val) = opt {
        egui::Checkbox::new(val, "")
    } else {
        egui::Checkbox::new(&mut val, "")
    };
    let changed = ui.add(checkbox).changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(val);
    }
}

/// Chat UI for 3D scene understanding
pub fn chat_ui(state: &WindowContext, ctx: &egui::Context) -> (Option<String>, String) {
    let mut message_to_send = None;
    let mut current_input = state.chat_state.current_input.clone();
    let mut clear_highlights = false;
    let mut server_url = state.chat_state.mcp_server_url.clone();

    egui::Window::new("💬 3D Scene Chat")
        .default_width(400.)
        .default_height(500.)
        .resizable(true)
        .show(ctx, |ui| {
            // Chat messages area
            ui.heading("Ask about the 3D scene");
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(300.)
                .show(ui, |ui| {
                    for message in &state.chat_state.messages {
                        let color = if message.is_user {
                            egui::Color32::LIGHT_BLUE
                        } else {
                            egui::Color32::LIGHT_GREEN
                        };

                        let prefix = if message.is_user { "You: " } else { "AI: " };

                        ui.horizontal(|ui| {
                            ui.colored_label(color, prefix);
                            ui.label(&message.content);
                        });
                        ui.separator();
                    }
                });

            ui.separator();

            // Input area
            ui.horizontal(|ui| {
                let text_edit = egui::TextEdit::singleline(&mut current_input)
                    .hint_text("Ask about objects, locations, or navigation...")
                    .desired_width(ui.available_width() - 80.)
                    .id(egui::Id::new("chat_input")); // Give it a unique ID

                let response = ui.add_enabled(!state.chat_state.is_sending, text_edit);

                let send_button = ui.add_enabled(
                    !state.chat_state.is_sending && !current_input.trim().is_empty(),
                    egui::Button::new("Send"),
                );

                // Send message on button click or Enter key
                let should_send = send_button.clicked()
                    || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));

                if should_send && !current_input.trim().is_empty() {
                    message_to_send = Some(current_input.trim().to_string());
                    current_input.clear(); // Clear the input field when sending
                }
            });

            if state.chat_state.is_sending {
                ui.spinner();
                ui.label("Thinking...");
            }

            ui.separator();

            // Highlight controls
            ui.horizontal(|ui| {
                if ui.button("Clear Highlights").clicked() {
                    clear_highlights = true;
                }

                ui.label(format!(
                    "Objects: {}",
                    state.chat_state.highlighted_objects.len()
                ));
                if let Some(_path) = &state.chat_state.highlighted_path {
                    ui.label("Path: Active");
                }
            });

            // Server settings
            ui.collapsing("Settings", |ui| {
                ui.horizontal(|ui| {
                    ui.label("MCP Server URL:");
                    ui.text_edit_singleline(&mut server_url);
                });
            });
        });

    (message_to_send, current_input)
}

/// Create mock response for testing - replace with real async handling
pub fn create_mock_response(message: &str) -> McpResponse {
    use std::collections::HashMap;

    // Create mock response based on the message content
    if message.to_lowercase().contains("table") {
        let mut attributes = HashMap::new();
        attributes.insert("type".to_string(), "dining_table".to_string());
        attributes.insert("material".to_string(), "wood".to_string());
        attributes.insert("size".to_string(), "large".to_string());

        let object = crate::chat::SceneObject {
            name: "Dining Table".to_string(),
            position: [2.5, 0.0, 1.0],
            attributes,
            confidence: Some(0.95),
        };

        McpResponse::Objects {
            objects: vec![object],
            description: Some("Found a large wooden dining table in the scene".to_string()),
        }
    } else if message.to_lowercase().contains("chair") {
        let mut attributes = HashMap::new();
        attributes.insert("type".to_string(), "chair".to_string());
        attributes.insert("material".to_string(), "wood".to_string());

        let objects = vec![
            crate::chat::SceneObject {
                name: "Chair 1".to_string(),
                position: [1.0, 0.0, 1.0],
                attributes: attributes.clone(),
                confidence: Some(0.90),
            },
            crate::chat::SceneObject {
                name: "Chair 2".to_string(),
                position: [4.0, 0.0, 1.0],
                attributes,
                confidence: Some(0.85),
            },
        ];

        McpResponse::Objects {
            objects,
            description: Some("Found 2 wooden chairs in the scene".to_string()),
        }
    } else if message.to_lowercase().contains("path") || message.to_lowercase().contains("navigate")
    {
        let path = crate::chat::ScenePath {
            waypoints: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.5],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
            ],
            description: Some("Path from entrance to dining area".to_string()),
        };

        McpResponse::Path {
            path,
            description: Some("Navigation path calculated".to_string()),
        }
    } else {
        McpResponse::Error {
            message: "I can help you find objects like tables, chairs, or navigate between locations. Try asking 'Where is the table?' or 'Show me a path to the kitchen'".to_string(),
        }
    }
}

/// Format MCP response for display
pub fn format_response(response: &McpResponse) -> String {
    match response {
        McpResponse::Objects {
            objects,
            description,
        } => {
            let desc = description.as_deref().unwrap_or("Found objects");
            format!(
                "{}: {}",
                desc,
                objects
                    .iter()
                    .map(|o| o.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        McpResponse::Path { description, .. } => description
            .as_deref()
            .unwrap_or("Path calculated")
            .to_string(),
        McpResponse::Error { message } => message.clone(),
    }
}
