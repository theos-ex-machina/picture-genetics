use eframe::egui;
use std::fs;

use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use crate::blocks::{BlockCandidate, ConstructedImage, calculate_total_sum_of_differences};

pub struct PictureGeneticsApp {
    target_image_path: String,
    output_path: String,
    output_filename: String,
    num_blocks: usize,
    blocks_per_iteration: usize,
    num_iterations: usize,
    blocks_to_keep: usize,
    random_added_blocks: usize,
    render_from_scratch: bool,
    base_image_path: String,
    status_message: String,
    is_running: bool,
    available_images: Vec<String>,
    progress: f32,
    current_block: usize,
    estimated_time_remaining: Option<Duration>,
    processing_thread: Option<thread::JoinHandle<()>>,
    progress_receiver: Option<std::sync::mpsc::Receiver<ProcessingUpdate>>,
    intermediate_image_texture: Option<egui::TextureHandle>,
    show_intermediate_image: bool,
}

#[derive(Debug, Clone)]
struct ProcessingUpdate {
    status: String,
    progress: f32,
    current_block: usize,
    estimated_time_remaining: Option<Duration>,
    image_updated: bool,
}

impl Default for PictureGeneticsApp {
    fn default() -> Self {
        let mut app = Self {
            target_image_path: String::new(),
            output_path: "output".to_string(),
            output_filename: "generated-image".to_string(),
            num_blocks: crate::NUM_BLOCKS,
            blocks_per_iteration: crate::BLOCKS_PER_GENETIC_ITERATION,
            num_iterations: crate::NUM_GENETIC_ITERATIONS,
            blocks_to_keep: crate::BLOCKS_TO_KEEP,
            random_added_blocks: crate::RANDOM_ADDED_BLOCKS,
            render_from_scratch: true,
            base_image_path: String::new(),
            status_message: "Ready to start".to_string(),
            is_running: false,
            available_images: Vec::new(),
            progress: 0.0,
            current_block: 0,
            estimated_time_remaining: None,
            processing_thread: None,
            progress_receiver: None,
            intermediate_image_texture: None,
            show_intermediate_image: true,
        };
        app.load_available_images();
        app
    }
}

impl PictureGeneticsApp {
    fn load_available_images(&mut self) {
        self.available_images.clear();

        // Load input images
        if let Ok(entries) = fs::read_dir("input") {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.ends_with(".png")
                        || filename.ends_with(".jpg")
                        || filename.ends_with(".jpeg")
                        || filename.ends_with(".webp")
                    {
                        self.available_images.push(filename.to_string());
                    }
                }
            }
        }
    }

    fn start_processing(&mut self) {
        if self.target_image_path.is_empty() {
            self.status_message = "Please select a target image".to_string();
            return;
        }

        self.is_running = true;
        self.status_message = "Loading textures and initializing...".to_string();
        self.progress = 0.0;
        self.current_block = 0;
        self.estimated_time_remaining = None;

        let (tx, rx) = std::sync::mpsc::channel();
        self.progress_receiver = Some(rx);

        // Clone parameters for the thread
        let target_image_path = format!("input\\{}", self.target_image_path);
        let render_from_scratch = self.render_from_scratch;
        let base_image_path = if !self.render_from_scratch && !self.base_image_path.is_empty() {
            Some(format!("output\\{}.png", self.base_image_path))
        } else {
            None
        };
        let output_filename = format!("{}\\{}.png", self.output_path, self.output_filename);
        let num_blocks = self.num_blocks;
        let blocks_per_iteration = self.blocks_per_iteration;
        let num_iterations = self.num_iterations;
        let blocks_to_keep = self.blocks_to_keep;
        let random_added_blocks = self.random_added_blocks;

        let handle = thread::spawn(move || {
            Self::run_processing(
                tx,
                target_image_path,
                render_from_scratch,
                base_image_path,
                output_filename,
                num_blocks,
                blocks_per_iteration,
                num_iterations,
                blocks_to_keep,
                random_added_blocks,
            );
        });

        self.processing_thread = Some(handle);
    }

    fn run_processing(
        tx: std::sync::mpsc::Sender<ProcessingUpdate>,
        target_image_path: String,
        render_from_scratch: bool,
        base_image_path: Option<String>,
        output_filename: String,
        num_blocks: usize,
        blocks_per_iteration: usize,
        num_iterations: usize,
        blocks_to_keep: usize,
        random_added_blocks: usize,
    ) {
        // Try to initialize GPU (infrastructure ready for future use)
        let _gpu_scorer = crate::gpu::GpuScorer::new();
        let _use_gpu = _gpu_scorer.is_some();

        // Send initial status
        let _ = tx.send(ProcessingUpdate {
            status: if _use_gpu {
                "GPU acceleration enabled! Loading target image...".to_string()
            } else {
                "GPU not available, using CPU. Loading target image...".to_string()
            },
            progress: 0.0,
            current_block: 0,
            estimated_time_remaining: None,
            image_updated: false,
        });

        // Load target image
        let target_image: RgbaImage = match image::open(&target_image_path) {
            Ok(img) => img.to_rgba8(),
            Err(e) => {
                let _ = tx.send(ProcessingUpdate {
                    status: format!("Error loading target image: {}", e),
                    progress: 0.0,
                    current_block: 0,
                    estimated_time_remaining: None,
                    image_updated: false,
                });
                return;
            }
        };

        let _ = tx.send(ProcessingUpdate {
            status: "Loading minecraft textures...".to_string(),
            progress: 0.0,
            current_block: 0,
            estimated_time_remaining: None,
            image_updated: false,
        });

        // Load textures
        let mut possible_textures: Vec<std::sync::Arc<RgbaImage>> = Vec::new();
        if let Ok(entries) = fs::read_dir("./textures/blocks") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Ok(image) = image::open(path) {
                        let rgba_image = image.to_rgba8();
                        if rgba_image.dimensions() == (16, 16) {
                            possible_textures.push(std::sync::Arc::new(rgba_image));
                        }
                    }
                }
            }
        }

        if possible_textures.is_empty() {
            let _ = tx.send(ProcessingUpdate {
                status: "Error: No valid textures found in ./textures/blocks".to_string(),
                progress: 0.0,
                current_block: 0,
                estimated_time_remaining: None,
                image_updated: false,
            });
            return;
        }

        let _ = tx.send(ProcessingUpdate {
            status: format!("{} minecraft textures loaded!", possible_textures.len()),
            progress: 0.0,
            current_block: 0,
            estimated_time_remaining: None,
            image_updated: false,
        });

        let mut constructed_image = ConstructedImage::new();
        let target_width = target_image.width();
        let target_height = target_image.height();

        let mut rendered_canvas: RgbaImage;
        let mut base_image: Option<RgbaImage> = None;
        let average_pixel: Rgba<u8>;

        if render_from_scratch {
            average_pixel = Self::get_average_pixel(&target_image);
            rendered_canvas =
                image::RgbaImage::from_pixel(target_width, target_height, average_pixel);
        } else if let Some(base_path) = base_image_path {
            match image::open(&base_path) {
                Ok(img) => {
                    rendered_canvas = img.to_rgba8();
                    base_image = Some(rendered_canvas.clone());
                    average_pixel = Self::get_average_pixel(&target_image);
                }
                Err(e) => {
                    let _ = tx.send(ProcessingUpdate {
                        status: format!("Error loading base image: {}", e),
                        progress: 0.0,
                        current_block: 0,
                        estimated_time_remaining: None,
                        image_updated: false,
                    });
                    return;
                }
            }
        } else {
            average_pixel = Self::get_average_pixel(&target_image);
            rendered_canvas =
                image::RgbaImage::from_pixel(target_width, target_height, average_pixel);
        }

        let mut current_total_ssd =
            calculate_total_sum_of_differences(&target_image, &rendered_canvas);
        let mut times: Vec<Duration> = Vec::with_capacity(num_blocks);

        let _ = tx.send(ProcessingUpdate {
            status: "Starting natural selection algorithm...".to_string(),
            progress: 0.0,
            current_block: 0,
            estimated_time_remaining: None,
            image_updated: false,
        });

        for i in 0..num_blocks {
            let _ = tx.send(ProcessingUpdate {
                status: format!("Block #{}, starting initial block generation...", i + 1),
                progress: (i as f32) / (num_blocks as f32),
                current_block: i + 1,
                estimated_time_remaining: if !times.is_empty() {
                    let average_time =
                        times.iter().map(|d| d.as_millis()).sum::<u128>() / times.len() as u128;
                    Some(Duration::from_millis(
                        average_time as u64 * (num_blocks - i) as u64,
                    ))
                } else {
                    None
                },
                image_updated: false,
            });

            let time = Instant::now();
            let mut current_population: Vec<BlockCandidate> = (0..blocks_per_iteration)
                .into_par_iter()
                .map(|_| {
                    let mut local_rand = rand::rng();
                    BlockCandidate::new(
                        &possible_textures,
                        &mut local_rand,
                        target_width as f32,
                        target_height as f32,
                    )
                })
                .collect();

            let _ = tx.send(ProcessingUpdate {
                status: format!("Block #{}, starting reproduction algorithms...", i + 1),
                progress: (i as f32) / (num_blocks as f32),
                current_block: i + 1,
                estimated_time_remaining: if !times.is_empty() {
                    let average_time =
                        times.iter().map(|d| d.as_millis()).sum::<u128>() / times.len() as u128;
                    Some(Duration::from_millis(
                        average_time as u64 * (num_blocks - i) as u64,
                    ))
                } else {
                    None
                },
                image_updated: false,
            });

            for _ in 0..num_iterations {
                let mut use_cpu = true;
                if let Some(ref gpu) = _gpu_scorer {
                    // Extract block data for GPU
                    let mut block_data: Vec<(f32, f32, f32)> = current_population
                        .iter()
                        .map(|b| (b.x(), b.y(), b.scale()))
                        .collect();

                    // Assuming all blocks use the same texture (first one)
                    let texture = current_population[0].texture();

                    // Try to get scores from GPU
                    if let Some(scores) = gpu.score_blocks(
                        &target_image,
                        &rendered_canvas,
                        texture,
                        &mut block_data,
                        current_total_ssd,
                    ) {
                        // Apply scores back
                        for (block, score) in current_population.iter_mut().zip(scores.iter()) {
                            block.score = *score;
                        }
                        use_cpu = false;
                    }
                }

                if use_cpu {
                    // CPU fallback (GPU unavailable or images too large)
                    current_population.par_iter_mut().for_each(|block| {
                        block.calculate_score_incrementally(
                            &target_image,
                            &rendered_canvas,
                            current_total_ssd,
                        );
                    });
                }

                current_population.sort_unstable_by(|a, b| {
                    a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal)
                });

                let mut parents = current_population;
                parents.truncate(blocks_to_keep);

                let mut children: Vec<BlockCandidate> =
                    Vec::with_capacity(blocks_per_iteration - blocks_to_keep);

                let base_children_per_parent =
                    (blocks_per_iteration - blocks_to_keep) / blocks_to_keep;
                let mut remaining_children_to_assign =
                    (blocks_per_iteration - blocks_to_keep) % blocks_to_keep;

                for parent_block in parents.iter() {
                    let num_to_create = base_children_per_parent
                        + if remaining_children_to_assign > 0 {
                            remaining_children_to_assign -= 1;
                            1
                        } else {
                            0
                        };
                    if num_to_create > 0 {
                        let mut local_rand = rand::rng();
                        children.append(
                            &mut parent_block.create_children(num_to_create, &mut local_rand),
                        );
                    }
                }

                current_population = parents;
                current_population.append(&mut children);

                for _ in 0..random_added_blocks {
                    let mut local_rand = rand::rng();
                    current_population.push(BlockCandidate::new(
                        &possible_textures,
                        &mut local_rand,
                        target_width as f32,
                        target_height as f32,
                    ));
                }
            }

            let mut use_cpu = true;
            if let Some(ref gpu) = _gpu_scorer {
                // Extract block data for GPU
                let mut block_data: Vec<(f32, f32, f32)> = current_population
                    .iter()
                    .map(|b| (b.x(), b.y(), b.scale()))
                    .collect();

                // Assuming all blocks use the same texture (first one)
                let texture = current_population[0].texture();

                // Try to get scores from GPU
                if let Some(scores) = gpu.score_blocks(
                    &target_image,
                    &rendered_canvas,
                    texture,
                    &mut block_data,
                    current_total_ssd,
                ) {
                    // Apply scores back
                    for (block, score) in current_population.iter_mut().zip(scores.iter()) {
                        block.score = *score;
                    }
                    use_cpu = false;
                }
            }

            if use_cpu {
                // CPU fallback (GPU unavailable or images too large)
                current_population.par_iter_mut().for_each(|block| {
                    block.calculate_score_incrementally(
                        &target_image,
                        &rendered_canvas,
                        current_total_ssd,
                    );
                });
            }

            current_population
                .sort_unstable_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));

            if !current_population.is_empty() && current_population[0].score < current_total_ssd {
                let elapsed = time.elapsed();
                times.push(elapsed);

                let _ = tx.send(ProcessingUpdate {
                    status: format!(
                        "Winning block found in {:?}, score: {}, adding to image",
                        elapsed, current_population[0].score
                    ),
                    progress: (i as f32) / (num_blocks as f32),
                    current_block: i + 1,
                    estimated_time_remaining: if !times.is_empty() {
                        let average_time =
                            times.iter().map(|d| d.as_millis()).sum::<u128>() / times.len() as u128;
                        Some(Duration::from_millis(
                            average_time as u64 * (num_blocks - i - 1) as u64,
                        ))
                    } else {
                        None
                    },
                    image_updated: true,
                });

                constructed_image.blocks.push(current_population[0].clone());

                rendered_canvas = constructed_image.construct_image(
                    target_width,
                    target_height,
                    &average_pixel,
                    &base_image,
                );

                current_total_ssd =
                    calculate_total_sum_of_differences(&target_image, &rendered_canvas);

                // Save intermediate image
                let _ = constructed_image
                    .construct_image(target_width, target_height, &average_pixel, &base_image)
                    .save("intermitent-image.png");
            } else {
                let _ = tx.send(ProcessingUpdate {
                    status: format!(
                        "Warning: No improvement found for block {}. Best score: {}, Current SSD: {}",
                        i + 1,
                        if current_population.is_empty() {
                            "N/A".to_string()
                        } else {
                            current_population[0].score.to_string()
                        },
                        current_total_ssd
                    ),
                    progress: (i as f32) / (num_blocks as f32),
                    current_block: i + 1,
                    estimated_time_remaining: if !times.is_empty() {
                        let average_time = times.iter().map(|d| d.as_millis()).sum::<u128>() / times.len() as u128;
                        Some(Duration::from_millis(average_time as u64 * (num_blocks - i - 1) as u64))
                    } else {
                        None
                    },
                    image_updated: false,
                });
            }
        }

        // Save final image
        let _ = tx.send(ProcessingUpdate {
            status: "Saving final image...".to_string(),
            progress: 1.0,
            current_block: num_blocks,
            estimated_time_remaining: Some(Duration::ZERO),
            image_updated: false,
        });

        match constructed_image
            .construct_image(target_width, target_height, &average_pixel, &base_image)
            .save(&output_filename)
        {
            Ok(_) => {
                let _ = tx.send(ProcessingUpdate {
                    status: format!("Generation complete! Image saved as {}", output_filename),
                    progress: 1.0,
                    current_block: num_blocks,
                    estimated_time_remaining: Some(Duration::ZERO),
                    image_updated: false,
                });
            }
            Err(e) => {
                let _ = tx.send(ProcessingUpdate {
                    status: format!("Error saving image: {}", e),
                    progress: 1.0,
                    current_block: num_blocks,
                    estimated_time_remaining: Some(Duration::ZERO),
                    image_updated: false,
                });
            }
        }
    }

    fn get_average_pixel(image: &RgbaImage) -> Rgba<u8> {
        let pixel_count = (image.width() * image.height()) as u64;
        let (r_sum, g_sum, b_sum, a_sum) = image
            .pixels()
            .map(|p| (p[0] as u64, p[1] as u64, p[2] as u64, p[3] as u64))
            .fold((0, 0, 0, 0), |(sr, sg, sb, sa), (r, g, b, a)| {
                (sr + r, sg + g, sb + b, sa + a)
            });

        Rgba([
            (r_sum / pixel_count) as u8,
            (g_sum / pixel_count) as u8,
            (b_sum / pixel_count) as u8,
            (a_sum / pixel_count) as u8,
        ])
    }

    fn load_intermediate_image(&mut self, ctx: &egui::Context) {
        if let Ok(image) = image::open("intermitent-image.png") {
            let rgba_image = image.to_rgba8();
            let size = [rgba_image.width() as usize, rgba_image.height() as usize];
            let pixels: Vec<egui::Color32> = rgba_image
                .pixels()
                .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
                .collect();

            let color_image = egui::ColorImage { size, pixels };

            self.intermediate_image_texture = Some(ctx.load_texture(
                "intermediate_image",
                color_image,
                egui::TextureOptions::default(),
            ));
        }
    }

    fn check_progress(&mut self, ctx: &egui::Context) {
        let mut should_finish = false;
        let mut should_update_image = false;

        if let Some(receiver) = &self.progress_receiver {
            while let Ok(update) = receiver.try_recv() {
                self.status_message = update.status;
                self.progress = update.progress;
                self.current_block = update.current_block;
                self.estimated_time_remaining = update.estimated_time_remaining;

                if update.image_updated {
                    should_update_image = true;
                }

                if update.progress >= 1.0 {
                    should_finish = true;
                }
            }
        }

        if should_update_image {
            self.load_intermediate_image(ctx);
        }

        if should_finish {
            self.is_running = false;
            self.progress_receiver = None;
            if let Some(handle) = self.processing_thread.take() {
                let _ = handle.join();
            }
        }
    }
}

impl eframe::App for PictureGeneticsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for progress updates
        self.check_progress(ctx);

        // Request repaint if processing
        if self.is_running {
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Picture Genetics - GUI");

            ui.separator();

            // Control buttons - always visible
            ui.horizontal(|ui| {
                if ui.button("▶ Start Processing").clicked() && !self.is_running {
                    self.start_processing();
                }

                if ui.button("⏹ Stop").clicked() && self.is_running {
                    self.is_running = false;
                    self.status_message = "Processing stopped".to_string();
                }

                if ui.button("📁 Open Input Folder").clicked() {
                    #[cfg(target_os = "windows")]
                    {
                        std::process::Command::new("explorer")
                            .arg("input")
                            .spawn()
                            .ok();
                    }
                }

                if ui.button("📁 Open Output Folder").clicked() {
                    #[cfg(target_os = "windows")]
                    {
                        std::process::Command::new("explorer")
                            .arg("output")
                            .spawn()
                            .ok();
                    }
                }
            });

            ui.separator();

            // Status and Progress - always visible
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(&self.status_message);
            });

            if self.is_running {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Processing...");
                });

                // Progress bar
                ui.horizontal(|ui| {
                    ui.label("Progress:");
                    ui.add(egui::ProgressBar::new(self.progress).show_percentage());
                    ui.label(format!("Block: {}/{}", self.current_block, self.num_blocks));
                });

                // Estimated time remaining
                if let Some(eta) = self.estimated_time_remaining {
                    ui.horizontal(|ui| {
                        ui.label("Estimated time remaining:");
                        ui.label(format!("{:.1} minutes", eta.as_secs_f64() / 60.0));
                    });
                }
            }

            ui.separator();

            // Collapsible Settings Section
            egui::CollapsingHeader::new("⚙ Settings")
                .default_open(!self.is_running)
                .show(ui, |ui| {
                    // Target Image Selection
                    ui.horizontal(|ui| {
                        ui.label("Target Image:");
                        egui::ComboBox::from_label("")
                            .selected_text(if self.target_image_path.is_empty() {
                                "Select an image..."
                            } else {
                                &self.target_image_path
                            })
                            .show_ui(ui, |ui| {
                                for image_name in &self.available_images {
                                    ui.selectable_value(
                                        &mut self.target_image_path,
                                        image_name.clone(),
                                        image_name,
                                    );
                                }
                            });

                        if ui.button("Refresh").clicked() {
                            self.load_available_images();
                        }
                    });

                    ui.separator();

                    // Parameters
                    ui.horizontal(|ui| {
                        ui.label("Number of blocks:");
                        ui.add(egui::Slider::new(&mut self.num_blocks, 100..=5000));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Blocks per iteration:");
                        ui.add(egui::Slider::new(
                            &mut self.blocks_per_iteration,
                            100..=2000,
                        ));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Number of iterations:");
                        ui.add(egui::Slider::new(&mut self.num_iterations, 10..=1000));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Blocks to keep per iteration:");
                        ui.add(egui::Slider::new(&mut self.blocks_to_keep, 10..=200));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Random blocks to add:");
                        ui.add(egui::Slider::new(&mut self.random_added_blocks, 5..=100));
                    });

                    ui.separator();

                    // Render options
                    ui.checkbox(&mut self.render_from_scratch, "Render from scratch");

                    if !self.render_from_scratch {
                        ui.horizontal(|ui| {
                            ui.label("Base image:");
                            ui.text_edit_singleline(&mut self.base_image_path);
                        });
                    }

                    ui.separator();

                    // Output settings
                    ui.horizontal(|ui| {
                        ui.label("Output folder:");
                        ui.text_edit_singleline(&mut self.output_path);
                    });

                    ui.horizontal(|ui| {
                        ui.label("Output filename:");
                        ui.text_edit_singleline(&mut self.output_filename);
                        ui.label(".png");
                    });
                });

            ui.separator();

            // Large Intermediate Image Display at Bottom
            if self.show_intermediate_image {
                ui.checkbox(
                    &mut self.show_intermediate_image,
                    "🖼 Show intermediate image",
                );

                if let Some(texture) = &self.intermediate_image_texture {
                    ui.separator();
                    ui.label("Current Progress:");

                    // Use most of the available space for the image
                    let available_size = ui.available_size();
                    let max_size = egui::Vec2::new(
                        available_size.x - 20.0,     // Leave some padding
                        available_size.y.max(400.0), // At least 400px height
                    );

                    let image_size = texture.size_vec2();
                    let scale = (max_size.x / image_size.x)
                        .min(max_size.y / image_size.y)
                        .min(1.0);
                    let display_size = image_size * scale;

                    // Center the image
                    ui.vertical_centered(|ui| {
                        ui.add(egui::Image::from_texture(texture).fit_to_exact_size(display_size));
                    });
                }
            } else {
                ui.checkbox(
                    &mut self.show_intermediate_image,
                    "🖼 Show intermediate image",
                );
            }
        });
    }
}
