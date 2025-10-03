mod blocks;
mod gpu;
mod gui;

pub const NUM_BLOCKS: usize = 2000;
pub const BLOCKS_PER_GENETIC_ITERATION: usize = 1000;
pub const NUM_GENETIC_ITERATIONS: usize = 500;

pub const BLOCKS_TO_KEEP: usize = 75;
pub const RANDOM_ADDED_BLOCKS: usize = 25;

pub const BLOCK_MIN_SCALE: f32 = 10.0;
pub const BLOCK_MAX_SCALE: f32 = 300.0;

pub const CHILDREN_MUTATION_VARIANCE_POS: f32 = 0.25;
pub const CHILDREN_MUTATION_VARIANCE_SCALE: f32 = 0.35;

pub const SCORE_SAMPLING_RATE: usize = 4;

fn main() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_min_inner_size([700.0, 500.0]),
        ..Default::default()
    };

    let _ = eframe::run_native(
        "Picture Genetics",
        options,
        Box::new(|_cc| Ok(Box::new(gui::PictureGeneticsApp::default()))),
    );
}
