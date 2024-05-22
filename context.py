import os
import datetime

from openai import OpenAI
import ollama
from datasets import load_dataset
from dotenv import load_dotenv
from slugify import slugify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()


class IVANContext:
    """
    APP-wide context.
    Initialized on app boot-up and passed down to all commands by Click.
    """

    openai_client: OpenAI = None
    """ Reference to current OpenAI API client. """

    ollama_client: ollama = None
    """ Reference to current Ollama API client. """

    rlhf_dataset_name = "Anthropic/hh-rlhf"
    """ Name of the RLHF dataset analyzed by IVAN. """

    rlhf_dataset = None
    """ Reference to RLHF dataset loaded in memory. """

    datetime_slug = slugify(
        datetime.datetime.utcnow().isoformat(sep=" ", timespec="minutes"),
    )
    """ Slug for "now": YYYY-MM-DD-HH-MM. """

    data_dir = os.path.join(os.getcwd(), "data")
    """ Path to the "data" dir. """

    transformers_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    """ Device to be used by Transformers / Sentence Transformers """

    reward_models_cache = {}
    """ Cache for "warm" reward models and their associated tokenizers. """

    text_similarity_models_cache = {}
    """ Cache for "warm" text similarity models """

    def __init__(self):
        # Check that required env vars are set
        for env_var_name in [
            "OLLAMA_API_HOST",
            "OLLAMA_API_TIMEOUT",
            "OPENAI_API_KEY",
            "OPENAI_ORG_ID",
        ]:
            try:
                assert os.environ.get(env_var_name, None)
            except Exception:
                raise KeyError(f"Required environment variable {env_var_name} not found.")

        # Load shared API clients and datasets
        self.openai_client = OpenAI()

        self.ollama_client = ollama.Client(
            host=os.environ.get("OLLAMA_API_HOST", "http://localhost:11434"),
            timeout=int(os.environ.get("OLLAMA_API_TIMEOUT", 60)),
        )

        self.rlhf_dataset = load_dataset(self.rlhf_dataset_name, split="train")

        # Add "row_id" to rlhf_dataset
        self.rlhf_dataset = self.rlhf_dataset.add_column(
            "row_id", [i for i in range(0, len(self.rlhf_dataset))]
        )

    def load_reward_model(self, model_name: str) -> None:
        """
        Loads a given reward model and associated tokenizer and keep them "warm" in cache.
        Uses HuggingFace's transformers library to load models.
        """
        rank_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        rank_model = rank_model.to(self.transformers_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.reward_models_cache[model_name] = {"model": rank_model, "tokenizer": tokenizer}

    def load_text_similarity_model(self, model_name: str) -> None:
        """
        Loads a given text similarity model and keeps it "warm" in cache.
        Uses Sentence Transformers to load models.
        """
        model = SentenceTransformer(model_name)
        self.text_similarity_models_cache[model_name] = model

    def infer_with_reward_model(self, model_name: str, question: str, response: str) -> float:
        """
        Runs inference on a question + response with a given reward model.
        Will attempt to load model if not ready.
        """
        if model_name not in self.reward_models_cache:
            self.load_reward_model(model_name)

        model = self.reward_models_cache[model_name]["model"]
        tokenizer = self.reward_models_cache[model_name]["tokenizer"]

        inputs = tokenizer(question, response, return_tensors="pt").to(self.transformers_device)
        score = model(**inputs).logits[0].detach()

        return float(score[0])

    def infer_with_text_similarity_model(
        self,
        model_name: str,
        texts: list[str],
        convert_to_tensor=True,
        normalize_embeddings=True,
    ) -> list:
        """
        Runs inference on text(s) using a given text similarity model and returns resulting embeddings.
        Will attempt to load model if not ready.
        """
        if model_name not in self.text_similarity_models_cache:
            self.load_text_similarity_model(model_name)

        model = self.text_similarity_models_cache[model_name]

        embeddings = model.encode(
            texts,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
        )

        return embeddings
