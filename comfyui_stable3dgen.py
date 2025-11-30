import torch
import numpy as np
import os
from PIL import Image
import trimesh
from huggingface_hub import snapshot_download
from hi3dgen.pipelines import Hi3DGenPipeline
import folder_paths

# A dictionary to hold the loaded models
LOADED_MODELS = {}
MAX_SEED = np.iinfo(np.int32).max

# Define the directory for caching weights
WEIGHTS_DIR = os.path.join(folder_paths.get_folder_path('models'), 'stable3dgen_weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def tensor_to_pil(tensor):
    """Converts a torch tensor to a PIL Image."""
    return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil_to_tensor(image):
    """Converts a PIL Image to a torch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Stable3DGenNode:
    """
    A custom node for ComfyUI that generates a 3D mesh from a single image
    using the Stable3DGen pipeline.
    """
    def __init__(self):
        self._load_models()

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "ss_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 6, "min": 1, "max": 50}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("mesh_path", "normal_map_image")
    FUNCTION = "generate"
    CATEGORY = "Stable3DGen"

    def _cache_weights(self):
        """Downloads and caches the model weights from Hugging Face."""
        model_ids = [
            "Stable-X/trellis-normal-v0-1",
            "Stable-X/yoso-normal-v1-8-1",
            "ZhengPeng7/BiRefNet",
        ]
        for model_id in model_ids:
            local_path = os.path.join(WEIGHTS_DIR, model_id.split("/")[-1])
            if not os.path.exists(local_path):
                print(f"Downloading and caching model: {model_id}")
                snapshot_download(repo_id=model_id, local_dir=local_path, force_download=False)
            else:
                print(f"Model already cached at: {local_path}")

    def _load_models(self):
        """Loads the required models into memory."""
        self._cache_weights()

        if "hi3dgen_pipeline" not in LOADED_MODELS:
            print("Loading Hi3DGenPipeline model...")
            pipeline = Hi3DGenPipeline.from_pretrained(os.path.join(WEIGHTS_DIR, "trellis-normal-v0-1"))
            pipeline.cuda()
            LOADED_MODELS["hi3dgen_pipeline"] = pipeline
            print("Hi3DGenPipeline model loaded.")

        if "normal_predictor" not in LOADED_MODELS:
            print("Loading StableNormal_turbo model...")
            try:
                # This assumes the model is structured correctly in the weights dir
                normal_predictor = torch.hub.load(
                    "hugoycj/StableNormal",
                    "StableNormal_turbo",
                    trust_repo=True,
                    yoso_version='yoso-normal-v1-8-1',
                    local_cache_dir=WEIGHTS_DIR
                )
                LOADED_MODELS["normal_predictor"] = normal_predictor
                print("StableNormal_turbo model loaded.")
            except Exception as e:
                print(f"Failed to load StableNormal_turbo model: {e}")
                raise e


    def generate(self, image, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps):
        """The main execution function for the node."""
        hi3dgen_pipeline = LOADED_MODELS["hi3dgen_pipeline"]
        normal_predictor = LOADED_MODELS["normal_predictor"]

        # Convert input tensor to PIL image
        pil_image = tensor_to_pil(image)

        # Pre-process image and generate normal map
        processed_image = hi3dgen_pipeline.preprocess_image(pil_image, resolution=1024)
        normal_image = normal_predictor(processed_image, resolution=768, match_input_resolution=True, data_type='object')

        # Run the 3D generation pipeline
        outputs = hi3dgen_pipeline.run(
            normal_image,
            seed=seed,
            formats=["mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        generated_mesh = outputs['mesh'][0]

        # Save the generated mesh to the ComfyUI output directory
        output_dir = folder_paths.get_output_directory()
        mesh_filename = f"stable3dgen_{seed}.glb"
        mesh_path = os.path.join(output_dir, mesh_filename)

        trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
        trimesh_mesh.export(mesh_path)

        # Convert the normal map (PIL Image) to a tensor for output
        normal_map_tensor = pil_to_tensor(normal_image)

        return (mesh_path, normal_map_tensor)

# A dictionary to map node class names to their respective classes
NODE_CLASS_MAPPINGS = {
    "Stable3DGenNode": Stable3DGenNode
}

# A dictionary to map node display names to their respective classes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Stable3DGen Generation": Stable3DGenNode
}
