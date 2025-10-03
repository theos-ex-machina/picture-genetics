// Compute shader for calculating block candidate scores
// This replaces the CPU-based score calculation with GPU acceleration

struct BlockCandidate {
    x: f32,
    y: f32,
    scale: f32,
    texture_index: u32,
    score: f32,
    _padding: vec3<f32>,
}

struct Uniforms {
    canvas_width: u32,
    canvas_height: u32,
    texture_width: u32,
    texture_height: u32,
    base_ssd: f32,
    sampling_rate: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> target_image: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> current_canvas: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> texture_data: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> blocks: array<BlockCandidate>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// Alpha blending function
fn blend_pixels(base: vec4<f32>, overlay: vec4<f32>) -> vec4<f32> {
    let alpha = overlay.a;
    let one_minus_alpha = 1.0 - alpha;
    
    return vec4<f32>(
        base.r * one_minus_alpha + overlay.r * alpha,
        base.g * one_minus_alpha + overlay.g * alpha,
        base.b * one_minus_alpha + overlay.b * alpha,
        base.a * one_minus_alpha + overlay.a * alpha
    );
}

// Calculate squared difference between two colors
fn color_difference_squared(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let diff_r = a.r - b.r;
    let diff_g = a.g - b.g;
    let diff_b = a.b - b.b;
    let diff_a = a.a - b.a;
    
    return diff_r * diff_r + diff_g * diff_g + diff_b * diff_b + diff_a * diff_a;
}

// Sample texture with bilinear filtering
fn sample_texture(tex_x: f32, tex_y: f32) -> vec4<f32> {
    let x_clamped = clamp(u32(tex_x), 0u, uniforms.texture_width - 1u);
    let y_clamped = clamp(u32(tex_y), 0u, uniforms.texture_height - 1u);
    let index = y_clamped * uniforms.texture_width + x_clamped;
    return texture_data[index];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_index = global_id.x;
    
    // Check if this thread should process a block
    if (block_index >= arrayLength(&blocks)) {
        return;
    }
    
    let block = blocks[block_index];
    
    // Calculate block bounds
    let half_scale = block.scale * 0.5;
    let block_x_start = block.x - half_scale;
    let block_y_start = block.y - half_scale;
    let block_x_end = block.x + half_scale;
    let block_y_end = block.y + half_scale;
    
    // Clamp to canvas bounds
    let loop_start_x = u32(max(floor(block_x_start), 0.0));
    let loop_end_x = u32(min(ceil(block_x_end), f32(uniforms.canvas_width)));
    let loop_start_y = u32(max(floor(block_y_start), 0.0));
    let loop_end_y = u32(min(ceil(block_y_end), f32(uniforms.canvas_height)));
    
    // Check if block is valid
    if (loop_start_x >= loop_end_x || loop_start_y >= loop_end_y) {
        blocks[block_index].score = 999999999.0;
        return;
    }
    
    var delta_ssd: f32 = 0.0;
    var pixels_sampled: u32 = 0u;
    let step = uniforms.sampling_rate;
    
    // Calculate texture dimensions
    let tex_width_f = f32(uniforms.texture_width);
    let tex_height_f = f32(uniforms.texture_height);
    
    // Iterate through pixels in the block area
    for (var cy = loop_start_y; cy < loop_end_y; cy += step) {
        for (var cx = loop_start_x; cx < loop_end_x; cx += step) {
            // Calculate position within the block
            let x_in_block = f32(cx) - block_x_start;
            let y_in_block = f32(cy) - block_y_start;
            
            // Normalize to [0, 1]
            let norm_x = x_in_block / block.scale;
            let norm_y = y_in_block / block.scale;
            
            // Map to texture coordinates
            let tex_x = norm_x * tex_width_f;
            let tex_y = norm_y * tex_height_f;
            
            // Sample texture
            let block_pixel = sample_texture(tex_x, tex_y);
            
            // Get canvas index
            let canvas_idx = cy * uniforms.canvas_width + cx;
            
            // Get pixels from target and current canvas
            let target_pixel = target_image[canvas_idx];
            let current_pixel = current_canvas[canvas_idx];
            
            // Blend the block pixel with current canvas pixel
            let blended_pixel = blend_pixels(current_pixel, block_pixel);
            
            // Calculate the change in error
            let old_diff = color_difference_squared(target_pixel, current_pixel);
            let new_diff = color_difference_squared(target_pixel, blended_pixel);
            
            delta_ssd += (new_diff - old_diff);
            pixels_sampled += 1u;
        }
    }
    
    // Scale up delta_ssd if we used sampling
    var effective_delta_ssd = delta_ssd;
    if (step > 1u && pixels_sampled > 0u) {
        let width_in_loop = loop_end_x - loop_start_x;
        let height_in_loop = loop_end_y - loop_start_y;
        let total_pixels = f32(width_in_loop * height_in_loop);
        
        if (total_pixels > 0.0) {
            effective_delta_ssd = delta_ssd * (total_pixels / f32(pixels_sampled));
        }
    }
    
    // Calculate final score
    let new_score = uniforms.base_ssd + effective_delta_ssd;
    blocks[block_index].score = max(new_score, 0.0);
}
