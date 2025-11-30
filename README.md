# Stable3DGen ComfyUI Custom Node

This custom node allows you to use the Stable3DGen pipeline directly within ComfyUI to generate 3D meshes from images.

## Installation

1.  **Place the Node File:**
    *   Download the `comfyui_stable3dgen.py` file.
    *   Place it in your `ComfyUI/custom_nodes/` directory.

2.  **Install Dependencies:**
    *   This node has several dependencies that need to be installed in your ComfyUI's Python environment.
    *   Navigate to your ComfyUI installation directory.
    *   Activate your virtual environment if you are using one.
    *   Install the required packages by running:
        ```bash
        pip install diffusers accelerate triton kornia==0.8.0 timm==0.6.7 transformers pillow tqdm scipy trimesh numpy==1.26.4 scikit-image opencv-python einops huggingface_hub
        ```
    *   **Note:** You also need a compatible version of PyTorch with CUDA support installed.

3.  **Restart ComfyUI:**
    *   After installing the dependencies, restart ComfyUI.

## Usage

1.  **Find the Node:**
    *   In the ComfyUI menu, you can find the new node under the **Stable3DGen** category. Add the "Stable3DGen Generation" node to your workflow.

2.  **Connect Inputs:**
    *   **image:** Connect a standard ComfyUI `IMAGE` output (e.g., from a LoadImage node).
    *   **seed:** An integer to control the randomness of the generation.
    *   **ss_guidance_strength / ss_sampling_steps:** Parameters for the Sparse Structure Generation stage.
    *   **slat_guidance_strength / slat_sampling_steps:** Parameters for the Structured Latent Generation stage.

3.  **Get Outputs:**
    *   **mesh_path:** A string containing the absolute path to the generated `.glb` 3D mesh file. The file is saved in your ComfyUI `output` directory.
    *   **normal_map_image:** An `IMAGE` output of the intermediate normal map, which can be previewed or used in other nodes.

## Model Caching

*   The first time you run the node, it will automatically download the required models from Hugging Face (approximately several GB).
*   The models will be cached in your `ComfyUI/models/stable3dgen_weights/` directory for future use.
