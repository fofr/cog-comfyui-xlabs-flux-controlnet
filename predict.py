import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]
mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def preprocessor_map(self, preprocessor_label):
        return {
            "Canny": "CannyEdgePreprocessor",
            "Midas": "MiDaS-DepthMapPreprocessor",
            "Zoe": "Zoe-DepthMapPreprocessor",
            "DepthAnything": "DepthAnythingPreprocessor",
            "Zoe-DepthAnything": "Zoe_DepthAnythingPreprocessor",
            "HED": "HEDPreprocessor",
            "TEED": "TEEDPreprocessor",
            "PiDiNet": "PiDiNetPreprocessor",
        }[preprocessor_label]

    def control_weights_map(self, control_type):
        return {
            "canny": "flux-canny-controlnet-v3.safetensors",
            "soft_edge": "flux-hed-controlnet-v3.safetensors",
            "depth": "flux-depth-controlnet-v3.safetensors",
        }[control_type]

    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["53"]["inputs"]
        positive_prompt["clip_l"] = kwargs["prompt"]
        positive_prompt["t5xxl"] = kwargs["prompt"]
        positive_prompt["guidance"] = kwargs["guidance_scale"]

        negative_prompt = workflow["57"]["inputs"]
        negative_prompt["clip_l"] = f"nsfw, {kwargs['negative_prompt']}"
        negative_prompt["t5xxl"] = f"nsfw, {kwargs['negative_prompt']}"
        negative_prompt["guidance"] = kwargs["guidance_scale"]

        control_image = workflow["16"]["inputs"]
        control_image["image"] = kwargs["control_image_filename"]

        control_strength = workflow["14"]["inputs"]
        control_strength["strength"] = kwargs["control_strength"]

        sampler = workflow["3"]["inputs"]
        sampler["steps"] = kwargs["steps"]
        sampler["noise_seed"] = kwargs["seed"]
        sampler["seed"] = kwargs["seed"]
        sampler["true_gs"] = kwargs["guidance_scale"]
        sampler["image_to_image_strength"] = kwargs["image_to_image_strength"]

        control_weights = workflow["13"]["inputs"]
        control_weights["controlnet_path"] = self.control_weights_map(
            kwargs["control_type"]
        )

        if kwargs["control_type"] == "depth":
            preprop = kwargs["depth_preprocessor"]
        elif kwargs["control_type"] == "soft_edge":
            preprop = kwargs["soft_edge_preprocessor"]
        else:
            preprop = "Canny"

        preprocessor = workflow["51"]["inputs"]
        preprocessor["preprocessor"] = self.preprocessor_map(preprop)

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        guidance_scale: float = Input(
            description="Guidance scale",
            default=3.5,
            le=5,
            ge=0,
        ),
        steps: int = Input(
            description="Number of steps",
            default=28,
            le=50,
            ge=1,
        ),
        control_type: str = Input(
            description="Type of control net",
            choices=["canny", "soft_edge", "depth"],
            default="depth",
        ),
        control_strength: float = Input(
            description="Strength of control net. Different controls work better with different strengths. Canny works best with 0.5, soft edge works best with 0.4, and depth works best between 0.5 and 0.75. If images are low quality, try reducing the strength and try reducing the guidance scale.",
            default=0.5,
            le=3,
            ge=0,
        ),
        control_image: Path = Input(
            description="Image to use with control net",
        ),
        image_to_image_strength: float = Input(
            description="Strength of image to image control. 0 means none of the control image is used. 1 means the control image is returned used as is. Try values between 0 and 0.25 for best results.",
            default=0,
            le=1,
            ge=0,
        ),
        depth_preprocessor: str = Input(
            description="Preprocessor to use with depth control net",
            choices=["Midas", "Zoe", "DepthAnything", "Zoe-DepthAnything"],
            default="DepthAnything",
        ),
        soft_edge_preprocessor: str = Input(
            description="Preprocessor to use with soft edge control net",
            choices=["HED", "TEED", "PiDiNet"],
            default="HED",
        ),
        return_preprocessed_image: bool = Input(
            description="Return the preprocessed image used to control the generation process. Useful for debugging.",
            default=False,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        control_image_filename = None
        if control_image:
            control_image_filename = self.filename_with_extension(
                control_image, "control_image"
            )
            self.handle_input_file(control_image, control_image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_type=control_type,
            control_image_filename=control_image_filename,
            control_strength=control_strength,
            guidance_scale=guidance_scale,
            steps=steps,
            soft_edge_preprocessor=soft_edge_preprocessor,
            depth_preprocessor=depth_preprocessor,
            image_to_image_strength=image_to_image_strength,
            return_preprocessed_image=return_preprocessed_image,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        output_directories = [OUTPUT_DIR]
        if return_preprocessed_image:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(output_directories)
        )
