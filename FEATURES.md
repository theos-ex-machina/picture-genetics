# Picture Genetics - Features

## Image Selection

### Target Image
- **📁 Browse Button**: Click to open a file picker dialog and select any image from your computer
- **Quick Select Dropdown**: Choose from images in the `input/` folder for convenience
- Supports: PNG, JPG, JPEG, WebP, BMP formats

### Base Image (Optional)
- When "Render from scratch" is unchecked, you can select a base image to start from
- **📁 Browse Button**: Opens file picker to select any image from your drive
- Useful for iteratively improving existing generated images

## GPU Acceleration

- Automatically detects and uses GPU when available
- Smart fallback to CPU for:
  - Large images (>128MB buffer size)
  - Systems without compatible GPU
- Status message shows whether GPU is enabled or using CPU

## Processing Controls

- **▶ Start Processing**: Begin the genetic algorithm
- **⏹ Stop Processing**: Cancel current processing
- **📁 Open Output Folder**: Quick access to generated images

## Settings (Collapsible)

All parameters are adjustable:
- Number of blocks (100-5000)
- Blocks per iteration (100-2000)
- Number of iterations (10-1000)
- Blocks to keep per iteration (10-200)
- Random blocks added (10-100)

## Real-Time Preview

- Large image preview at bottom of window
- Updates automatically during processing
- Shows current progress with percentage and estimated time remaining

## Output

- Configurable output folder and filename
- Saves intermediate image (`intermitent-image.png`) during processing
- Final image saved with timestamp or custom name
