import os
import urllib.parse
import shutil
import time
import subprocess
import tarfile
from huggingface_hub import hf_hub_download

HF_TEMP_DIR = "TEMP_HF"
COMFYUI_LORA_DIR = "ComfyUI/models/loras"


class DownloadExternalLora:
    def __init__(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    def download(self, url: str) -> str:
        if url.startswith("https://huggingface.co"):
            return self.download_from_huggingface(url)
        elif url.startswith("https://civitai.com"):
            return self.download_from_civitai(url)
        elif url.startswith("https://replicate.delivery"):
            return self.download_from_replicate(url)
        else:
            raise ValueError(
                "URL must be from 'huggingface.co', 'civitai.com', or 'replicate.delivery'"
            )

    def download_from_huggingface(self, url: str) -> str:
        repo_id, revision, filename_and_path, original_filename = (
            self.extract_parts_from_huggingface_url(url)
        )

        print(f"Downloading LoRA from HuggingFace: {url}")
        print(f"repo_id: {repo_id}")
        print(f"revision: {revision}")

        filename = f"{repo_id.replace('/', '_')}_{original_filename}"
        dest_path = os.path.join(COMFYUI_LORA_DIR, filename)

        if os.path.exists(dest_path):
            print(f"File {filename} already exists. Skipping download.")
            return filename

        file_path = hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename="/".join(filename_and_path),
            local_dir=HF_TEMP_DIR,
        )

        os.makedirs(COMFYUI_LORA_DIR, exist_ok=True)
        shutil.move(file_path, dest_path)

        print(f"Successfully downloaded {filename}")
        return filename

    def download_from_civitai(self, url: str) -> str:
        filename = self.get_civitai_filename(url)
        dest_path = os.path.join(COMFYUI_LORA_DIR, filename)

        if os.path.exists(dest_path):
            print(f"File {filename} already exists. Skipping download.")
            return filename

        print(f"Downloading LoRA from Civitai: {url} to {filename}")

        start_time = time.time()
        try:
            result = subprocess.run(["pget", "-f", url, dest_path], timeout=600)
            if result.returncode != 0:
                raise RuntimeError("Download failed.")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Download failed due to timeout")

        print(f"Successfully downloaded {filename}")
        end_time = time.time()
        print(f"Downloaded in: {end_time - start_time:.2f} seconds")

        return filename

    def download_from_replicate(self, url: str) -> str:
        print(f"Downloading LoRA from Replicate: {url}")
        filename = self.get_replicate_filename(url)
        dest_path = os.path.join(COMFYUI_LORA_DIR, filename)
        temp_tar_path = os.path.join(HF_TEMP_DIR, filename)

        if os.path.exists(dest_path):
            print(f"File {filename} already exists. Skipping download.")
            return filename

        subprocess.run(["pget", "-f", url, temp_tar_path], timeout=600)

        # Extract the safetensors file from the tar
        with tarfile.open(temp_tar_path, "r") as tar:
            lora_file = tar.extractfile("output/flux_train_replicate/lora.safetensors")
            if lora_file is None:
                raise ValueError("LoRA file not found in the downloaded tar")

            os.makedirs(COMFYUI_LORA_DIR, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(lora_file.read())

        os.unlink(temp_tar_path)
        print(f"Successfully downloaded and extracted {filename}")
        return filename

    @staticmethod
    def get_replicate_filename(url: str) -> str:
        unique_id = url.split("/")[-2]
        return f"replicate_lora_{unique_id}.safetensors"

    @staticmethod
    def get_civitai_filename(url: str) -> str:
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        model_id = parsed_url.path.split("/")[-1]

        filename_parts = [f"civitai_{model_id}"]
        for param in ["type", "format", "size", "fp"]:
            if value := query_params.get(param, [""])[0]:
                filename_parts.append(value.lower())

        return "_".join(filename_parts) + ".safetensors"

    @staticmethod
    def extract_parts_from_huggingface_url(url: str):
        parsed_url = urllib.parse.urlparse(url)
        path_parts = parsed_url.path.split("/")

        if len(path_parts) < 5:
            raise ValueError(f"HuggingFace URL does not contain enough parts: {url}")

        repo_id = f"{path_parts[1]}/{path_parts[2]}"
        revision = path_parts[4]
        filename_and_path = path_parts[5:]
        filename = filename_and_path[-1]

        return repo_id, revision, filename_and_path, filename
