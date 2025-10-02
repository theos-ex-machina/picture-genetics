use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs;
use std::io::stdin;
use std::sync::Arc;
use std::time::{self, Duration};

mod blocks;
use blocks::{BlockCandidate, ConstructedImage, calculate_total_sum_of_differences};

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
    println!("Enter the name of the target image");
    let mut target_image_path = String::new();
    stdin()
        .read_line(&mut target_image_path)
        .expect("Couldn't read user input");

    target_image_path = format!("input\\{}", target_image_path.trim());

    println!("Using {}...", target_image_path);
    let target_image: RgbaImage = image::open(target_image_path)
        .expect("bro the image aint there")
        .to_rgba8();
    println!("Target image loaded!");

    println!("Loading minecraft textures...");
    let mut possible_textures: Vec<Arc<RgbaImage>> = Vec::new();
    for entry in fs::read_dir("./textures/blocks").expect("could not read directory") {
        let path = entry.expect("couldn't load entry").path();

        if path.is_file() {
            let image = image::open(path).expect("Couldn't open image").to_rgba8();
            if image.dimensions() == (16, 16) {
                possible_textures.push(Arc::new(image));
            }
        }
    }
    println!("{:?} minecraft textures loaded!", possible_textures.len());

    let mut constructed_image = ConstructedImage::new();
    let mut rand = rand::rng();

    let target_width = target_image.width();
    let target_height = target_image.height();

    let mut rendered_canvas: RgbaImage;
    let mut base_image: Option<RgbaImage> = None;

    let mut average_pixel: Rgba<u8> = Rgba([0, 0, 0, 0]);

    println!("Render from scratch?");
    let mut answer = String::new();
    stdin()
        .read_line(&mut answer)
        .expect("Couldn't read user input");
    let answer = answer.trim();

    if answer.contains('y') || answer.contains('Y') {
        average_pixel = get_average_pixel(&target_image);
        rendered_canvas = image::RgbaImage::from_pixel(target_width, target_height, average_pixel);
    } else {
        println!("Enter the name of the starting image:");
        let mut base_image_path = String::new();
        stdin()
            .read_line(&mut base_image_path)
            .expect("Couldn't read user input");

        base_image_path = format!("output\\{}.png", base_image_path.trim());
        println!("Using {}...", base_image_path);

        rendered_canvas = image::open(base_image_path)
            .expect("Couldn't open provided image")
            .to_rgba8();

        base_image = Some(rendered_canvas.clone());
    }

    let num_blocks: usize = parse_input("How many blocks to add?", Some(NUM_BLOCKS))
        .expect("Couldn't parse user input");

    let num_blocks_per_genetic_iteration = parse_input(
        "How many block candidates per genetic iteration",
        Some(BLOCKS_PER_GENETIC_ITERATION),
    )
    .expect("Couldn't parse user input");

    let num_genetic_iterations: usize = parse_input(
        "How many genetic iterations per block?",
        Some(NUM_GENETIC_ITERATIONS),
    )
    .expect("Couldn't parse user input");

    let blocks_to_keep: usize = parse_input(
        "How many blocks to keep per iteration?",
        Some(BLOCKS_TO_KEEP),
    )
    .expect("Couldn't parse user input");

    let random_added_blocks: usize = parse_input(
        "How many random blocks to add per iteration?",
        Some(RANDOM_ADDED_BLOCKS),
    )
    .expect("Couldn't parse user input");

    let mut current_total_ssd = calculate_total_sum_of_differences(&target_image, &rendered_canvas);
    let mut times: Vec<Duration> = Vec::with_capacity(num_blocks);

    println!("Starting natural selection algorithm...");
    for i in 0..num_blocks {
        println!("Block #{}, starting initial block generation...", i);
        let time = time::Instant::now();
        let mut current_population: Vec<BlockCandidate> = (0..num_blocks_per_genetic_iteration)
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
        println!("block generation done!");

        println!("Starting reproduction algorithms...");
        for _ in 0..num_genetic_iterations {
            current_population.par_iter_mut().for_each(|block| {
                block.calculate_score_incrementally(
                    &target_image,
                    &rendered_canvas,
                    current_total_ssd,
                );
            });

            current_population
                .sort_unstable_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));

            let mut parents = current_population;
            parents.truncate(blocks_to_keep);

            let mut children: Vec<BlockCandidate> =
                Vec::with_capacity(num_blocks_per_genetic_iteration - blocks_to_keep);

            let base_children_per_parent =
                (num_blocks_per_genetic_iteration - blocks_to_keep) / blocks_to_keep;
            let mut remaining_children_to_assign =
                (num_blocks_per_genetic_iteration - blocks_to_keep) % blocks_to_keep;

            for parent_block in parents.iter() {
                let num_to_create = base_children_per_parent
                    + if remaining_children_to_assign > 0 {
                        remaining_children_to_assign -= 1;
                        1
                    } else {
                        0
                    };
                if num_to_create > 0 {
                    children.append(&mut parent_block.create_children(num_to_create, &mut rand));
                }
            }

            current_population = parents;
            current_population.append(&mut children);

            // add random extra blocks in there so it doesnt find some random minimum
            for _ in 0..random_added_blocks {
                current_population.push(BlockCandidate::new(
                    &possible_textures,
                    &mut rand,
                    target_width as f32,
                    target_height as f32,
                ));
            }
        }

        current_population.par_iter_mut().for_each(|block| {
            block.calculate_score_incrementally(&target_image, &rendered_canvas, current_total_ssd);
        });

        current_population
            .sort_unstable_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal)); // Use unstable sort

        if !current_population.is_empty() && current_population[0].score < current_total_ssd {
            let elapsed = time.elapsed();

            println!(
                "Winning block found in {:?}, score is: {} adding to image",
                elapsed, current_population[0].score
            );

            times.push(elapsed);

            let average_time = times
                .iter()
                .map(|duration| duration.as_millis())
                .sum::<u128>()
                / times.len() as u128;

            println!(
                "Predicted time remaining: {:?}",
                Duration::from_millis(average_time as u64 * (num_blocks - i) as u64)
            );

            constructed_image.blocks.push(current_population[0].clone());

            rendered_canvas = constructed_image.construct_image(
                target_width,
                target_height,
                &average_pixel,
                &base_image,
            );

            current_total_ssd = calculate_total_sum_of_differences(&target_image, &rendered_canvas);

            constructed_image
                .construct_image(target_width, target_height, &average_pixel, &base_image)
                .save("intermitent-image.png")
                .expect("Couldn't save image");
        } else {
            println!(
                "Warning: Population was empty or no improvement, no block added. Best score: {}, Current SSD: {}",
                if current_population.is_empty() {
                    "N/A".to_string()
                } else {
                    current_population[0].score.to_string()
                },
                current_total_ssd
            );
        }
    }

    println!("Generation done! What should the filename be?");
    let mut name = String::new();
    stdin()
        .read_line(&mut name)
        .expect("Couldn't read user input");
    name = format!("output\\{}.png", name.trim());

    println!("Saving constructed image as {}", name);

    constructed_image
        .construct_image(target_width, target_height, &average_pixel, &base_image)
        .save(name)
        .expect("Couldn't save image");

    println!("Yippee!");
}

fn parse_input<T: std::str::FromStr + std::fmt::Display>(
    question: &str,
    default: Option<T>,
) -> Result<T, T::Err> {
    match &default {
        Some(val) => println!("{} (Enter for default [{}])", question, val),
        None => println!("{} (Enter for default [None])", question),
    }

    let mut answer = String::new();
    stdin().read_line(&mut answer).expect("Couldn't read line");

    match answer.trim().parse::<T>() {
        Ok(val) => Ok(val),
        Err(e) => {
            if let Some(default_value) = default {
                Ok(default_value)
            } else {
                Err(e)
            }
        }
    }
}

fn get_average_pixel(image: &RgbaImage) -> Rgba<u8> {
    // Calculate average pixel of the target image
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
