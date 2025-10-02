use std::sync::Arc;

use image::{Pixel, Rgba, RgbaImage};
use rand::Rng;
use rand::rngs::ThreadRng;
use rand::seq::IndexedRandom;

use crate::{
    BLOCK_MAX_SCALE, BLOCK_MIN_SCALE, CHILDREN_MUTATION_VARIANCE_POS,
    CHILDREN_MUTATION_VARIANCE_SCALE, SCORE_SAMPLING_RATE,
};

#[derive(Clone)]
pub struct ConstructedImage {
    pub blocks: Vec<BlockCandidate>,
}

impl ConstructedImage {
    pub fn new() -> Self {
        ConstructedImage { blocks: Vec::new() }
    }

    pub fn construct_image(
        &self,
        target_image_width: u32,
        target_image_height: u32,
        average_target_image_pixel: &image::Rgba<u8>,
        base_image: &Option<RgbaImage>,
    ) -> RgbaImage {
        let mut constructed_image: RgbaImage;
        if let Some(image) = base_image {
            constructed_image = image.clone();
        } else {
            constructed_image = RgbaImage::from_pixel(
                target_image_width,
                target_image_height,
                *average_target_image_pixel,
            );
        }

        let img_width = constructed_image.width();
        let img_height = constructed_image.height();

        for block in &self.blocks {
            match block.for_each_relevant_pixel(
                img_width,
                img_height,
                1, // Step is 1 for constructing the image (no sampling)
                |cx_canvas, cy_canvas, block_pixel_to_apply| {
                    let canvas_pixel_mut = constructed_image.get_pixel_mut(cx_canvas, cy_canvas);
                    canvas_pixel_mut.blend(block_pixel_to_apply);
                },
            ) {
                Ok(_) => { /* Successfully processed block pixels */ }
                Err(_) => {
                    continue; /* Mimics original behavior for invalid block/texture */
                }
            }
        }

        constructed_image
    }
}

pub fn calculate_total_sum_of_differences(target: &RgbaImage, actual: &RgbaImage) -> f32 {
    let mut total_squared_difference: f64 = 0.0;
    for (t_pixel, a_pixel) in target.pixels().zip(actual.pixels()) {
        for k in 0..4 {
            // R, G, B, A components
            let diff = t_pixel[k] as f64 - a_pixel[k] as f64;
            total_squared_difference += diff * diff;
        }
    }

    total_squared_difference as f32
}

#[derive(Clone)]
pub struct BlockCandidate {
    texture: Arc<RgbaImage>,
    x: f32,
    y: f32,
    scale: f32,
    pub score: f32,
}

impl BlockCandidate {
    pub fn new(
        possible_textures: &[Arc<RgbaImage>],
        rand: &mut ThreadRng,
        target_width: f32,
        target_height: f32,
    ) -> Self {
        let texture: Arc<RgbaImage> = possible_textures
            .choose(rand)
            .expect("Texture list is empty")
            .clone();

        let x = rand.random_range(0.0..target_width);
        let y = rand.random_range(0.0..target_height);
        let scale = rand.random_range(
            BLOCK_MIN_SCALE..(BLOCK_MAX_SCALE.min(target_width.min(target_height) * 0.5)),
        );

        BlockCandidate {
            texture,
            x,
            y,
            scale,
            score: -1.0,
        }
    }

    fn for_each_relevant_pixel<F>(
        &self,
        canvas_width: u32,
        canvas_height: u32,
        step: usize,
        mut action: F,
    ) -> Result<(u32, u32, u32, u32, u64), &'static str>
    where
        F: FnMut(u32, u32, &Rgba<u8>),
    {
        let texture_width_f = self.texture.width() as f32;
        let texture_height_f = self.texture.height() as f32;

        if texture_width_f < 1.0 || texture_height_f < 1.0 {
            return Err("Texture dimensions are too small.");
        }
        if self.scale <= 1e-6 {
            return Err("Block scale is too small or zero.");
        }

        let block_on_canvas_x_start_f = self.x - self.scale / 2.0;
        let block_on_canvas_y_start_f = self.y - self.scale / 2.0;
        let block_on_canvas_x_end_f = self.x + self.scale / 2.0;
        let block_on_canvas_y_end_f = self.y + self.scale / 2.0;

        let loop_start_x = (block_on_canvas_x_start_f.max(0.0).floor() as u32).min(canvas_width);
        let loop_end_x = (block_on_canvas_x_end_f.max(0.0).ceil() as u32).min(canvas_width);
        let loop_start_y = (block_on_canvas_y_start_f.max(0.0).floor() as u32).min(canvas_height);
        let loop_end_y = (block_on_canvas_y_end_f.max(0.0).ceil() as u32).min(canvas_height);

        let mut pixels_iterated_count: u64 = 0;

        if loop_start_x >= loop_end_x || loop_start_y >= loop_end_y {
            // No pixels to iterate, block is likely off-canvas or has no area.
            return Ok((loop_start_x, loop_end_x, loop_start_y, loop_end_y, 0));
        }

        for cy_canvas_base in (loop_start_y..loop_end_y).step_by(step) {
            for cx_canvas_base in (loop_start_x..loop_end_x).step_by(step) {
                let cx_canvas = cx_canvas_base;
                let cy_canvas = cy_canvas_base;

                let x_in_block_scaled_f = cx_canvas as f32 - block_on_canvas_x_start_f;
                let y_in_block_scaled_f = cy_canvas as f32 - block_on_canvas_y_start_f;

                let norm_x = x_in_block_scaled_f / self.scale;
                let norm_y = y_in_block_scaled_f / self.scale;

                let tex_x_f = norm_x * texture_width_f;
                let tex_y_f = norm_y * texture_height_f;

                let tex_bx = (tex_x_f.floor() as u32).min(self.texture.width() - 1);
                let tex_by = (tex_y_f.floor() as u32).min(self.texture.height() - 1);

                let block_pixel_value = *self.texture.get_pixel(tex_bx, tex_by);
                action(cx_canvas, cy_canvas, &block_pixel_value);
                pixels_iterated_count += 1;
            }
        }
        Ok((
            loop_start_x,
            loop_end_x,
            loop_start_y,
            loop_end_y,
            pixels_iterated_count,
        ))
    }

    pub fn calculate_score_incrementally(
        &mut self,
        target_image: &RgbaImage,
        current_canvas_rendered: &RgbaImage,
        base_ssd: f32,
    ) {
        let mut delta_ssd: f64 = 0.0;

        let canvas_width = target_image.width();
        let canvas_height = target_image.height();

        let result = self.for_each_relevant_pixel(
            canvas_width,
            canvas_height,
            SCORE_SAMPLING_RATE,
            |cx_canvas, cy_canvas, new_block_pixel_ref| {
                let target_pixel = *target_image.get_pixel(cx_canvas, cy_canvas);
                let current_canvas_pixel = *current_canvas_rendered.get_pixel(cx_canvas, cy_canvas);

                let mut pixel_after_blend = current_canvas_pixel;
                pixel_after_blend.blend(new_block_pixel_ref);

                for k in 0..4 {
                    // R, G, B, A components
                    let t_k = target_pixel[k] as f64;
                    let c_old_k = current_canvas_pixel[k] as f64;
                    let c_after_blend_k = pixel_after_blend[k] as f64;

                    delta_ssd += (t_k - c_after_blend_k).powi(2) - (t_k - c_old_k).powi(2);
                }
            },
        );

        match result {
            Ok((loop_start_x, loop_end_x, loop_start_y, loop_end_y, pixels_sampled)) => {
                let effective_delta_ssd = if SCORE_SAMPLING_RATE > 1 && pixels_sampled > 0 {
                    let width_in_loop = if loop_end_x > loop_start_x {
                        loop_end_x - loop_start_x
                    } else {
                        0
                    };
                    let height_in_loop = if loop_end_y > loop_start_y {
                        loop_end_y - loop_start_y
                    } else {
                        0
                    };
                    let total_pixels_in_coverage = width_in_loop as f64 * height_in_loop as f64;

                    if total_pixels_in_coverage > 0.0 {
                        delta_ssd * (total_pixels_in_coverage / pixels_sampled as f64)
                    } else {
                        delta_ssd
                    }
                } else {
                    delta_ssd
                };

                let new_score = base_ssd as f64 + effective_delta_ssd;
                self.score = if new_score < 0.0 {
                    0.0
                } else {
                    new_score as f32
                };
            }
            Err(_) => {
                self.score = f32::MAX;
            }
        }
    }

    pub fn create_children(
        &self,
        num_children: usize,
        rand: &mut ThreadRng,
    ) -> Vec<BlockCandidate> {
        (0..num_children)
            .map(|_| {
                let mut child = self.clone();
                child.x += rand
                    .random_range(-CHILDREN_MUTATION_VARIANCE_POS..CHILDREN_MUTATION_VARIANCE_POS)
                    * self.x;
                child.y += rand
                    .random_range(-CHILDREN_MUTATION_VARIANCE_POS..CHILDREN_MUTATION_VARIANCE_POS)
                    * self.y;
                child.scale += rand.random_range(
                    -CHILDREN_MUTATION_VARIANCE_SCALE..CHILDREN_MUTATION_VARIANCE_SCALE,
                ) * self.scale;
                child.score = -1.0;
                child
            })
            .collect()
    }
}
