import io
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Parameter
from tqdm import tqdm


def _tqdm_options(config):
    return {
        "file": getattr(config, "_tqdm_file", None),
        "mininterval": float(getattr(config, "tqdm_mininterval", 60.0)),
        "disable": bool(getattr(config, "tqdm_disable", False)),
        "dynamic_ncols": True,
    }

class BiomarkerEmbedder(nn.Module):
    """Sum biomarker embeddings weighted by cell intensities."""

    def __init__(self, config, precomputed_esm=False):
        super().__init__()
        self.config = config
        self.device = config.device
        self.method = getattr(config, 'embedding_method', 'learnable')
        self.precomputed_esm = precomputed_esm
        self.biomarker_info = self._load_biomarker_info()
        self.biomarker_mapping = self._load_biomarker_mapping()
        self.valid_biomarkers = list(self.biomarker_info.keys())
        self.biomarker_embeddings = None
        self.biomarker_to_idx = {}
        self.idx_to_biomarker = {}
        self.embedding_dim = None
        self._initialize_method()

    def _load_biomarker_info(self):
        biomarker_info = {}
        filepath = self.config.biomarker_sequence_file

        try:
            df = pd.read_csv(io.StringIO(self.config.read_asset("biomarker_sequence_file")))

            for _, row in df.iterrows():
                biomarker_name = str(row.iloc[0]).strip()
                amino_acid_seq = row.iloc[1]

                if pd.isna(amino_acid_seq):
                    biomarker_info[biomarker_name] = {
                        'type': 'legit_no_sequence',
                        'sequence': None
                    }
                elif str(amino_acid_seq).upper() == 'N/A':
                    biomarker_info[biomarker_name] = {
                        'type': 'not_legit',
                        'sequence': None
                    }
                else:
                    biomarker_info[biomarker_name] = {
                        'type': 'has_sequence',
                        'sequence': str(amino_acid_seq).strip()
                    }

            print(f"Loaded {len(biomarker_info)} biomarkers from {filepath}")

        except FileNotFoundError:
            print(f"ERROR: The file '{filepath}' was not found.")
            return {}
        except Exception as e:
            print(f"ERROR loading biomarker info: {e}")
            return {}

        return biomarker_info

    def _load_biomarker_mapping(self):
        mapping = self.config.biomarker_aliases()
        print(f"Loaded {len(mapping)} biomarker aliases")
        return mapping

    def _sanitize_name(self, name):
        return str(name).replace('.', '_').replace(' ', '_').replace('-', '_')

    def _initialize_method(self):
        if self.method == 'onehot':
            self._init_onehot()
        elif self.method == 'learnable':
            self._init_learnable()
        elif self.method == 'esm':
            self._init_esm()
        else:
            raise ValueError(f"Unsupported embedding method: {self.method}")

    def _init_onehot(self):
        self.biomarker_embeddings = nn.ParameterDict()

    def _init_learnable(self):
        self.embedding_dim = getattr(self.config, 'learnable_embedding_dim', 512)
        self.biomarker_embeddings = nn.ParameterDict()

    def _init_esm(self):
        self.biomarker_embeddings = nn.ParameterDict()
        self.embedding_dim = 1280
        if self.precomputed_esm:
            return
        import esm

        print("Initializing ESM model...")
        try:
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_model.eval()
            self.esm_model = self.esm_model.to(self.device)
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
            print("ESM model loaded successfully")
        except Exception as e:
            print(f"Failed to load ESM model: {e}")
            raise

    def build_biomarker_vocab(self, all_biomarkers: list = None):
        """Resolve aliases and initialize one embedding per canonical biomarker."""
        print(f"Building biomarker vocabulary using {self.method} method...")

        if all_biomarkers is None:
            all_biomarkers = self.valid_biomarkers
        else:
            filtered_biomarkers = []
            for biomarker in all_biomarkers:
                if biomarker in self.biomarker_mapping:
                    if self.biomarker_mapping[biomarker] != "EMPTY":
                        filtered_biomarkers.append(biomarker)
                else:
                    filtered_biomarkers.append(biomarker)
            all_biomarkers = filtered_biomarkers

        all_biomarkers = list(dict.fromkeys(self.biomarker_mapping.get(name, name) for name in all_biomarkers))
        safe_names = [self._sanitize_name(name) for name in all_biomarkers]
        if len(set(safe_names)) != len(safe_names):
            raise ValueError("Biomarker names collide after parameter-name sanitization")
        print(f"Processing {len(all_biomarkers)} valid biomarkers (EMPTY ones filtered out)")

        for i, biomarker_name in enumerate(all_biomarkers):
            self.biomarker_to_idx[biomarker_name] = i
            self.idx_to_biomarker[i] = biomarker_name

        if self.method == 'onehot':
            self._build_onehot_embeddings(all_biomarkers)
        elif self.method == 'learnable':
            self._build_learnable_embeddings(all_biomarkers)
        elif self.method == 'esm':
            self._build_esm_embeddings(all_biomarkers)
        return

    def _build_onehot_embeddings(self, all_biomarkers: list):
        print("Building one-hot embeddings...")
        self.embedding_dim = len(all_biomarkers)
        for i, biomarker_name in enumerate(all_biomarkers):
            embedding = torch.zeros(self.embedding_dim, requires_grad=False)
            embedding[i] = 1.0
            safe_name = self._sanitize_name(biomarker_name)
            self.biomarker_embeddings[safe_name] = Parameter(embedding, requires_grad=False)

    def _build_learnable_embeddings(self, all_biomarkers: list):
        print("Building learnable embeddings...")
        for biomarker_name in all_biomarkers:
            canonical_name = self.biomarker_mapping.get(biomarker_name, biomarker_name)
            if canonical_name == "EMPTY":
                continue
            safe_name = self._sanitize_name(canonical_name)
            if safe_name in self.biomarker_embeddings:
                continue
            embedding = self._create_learnable_fallback_embedding(canonical_name)
            # Retain both names for compatibility with existing checkpoints.
            self.register_parameter(f"emb__{safe_name}", embedding)
            self.biomarker_embeddings[safe_name] = embedding

    def _create_learnable_fallback_embedding(self, biomarker_name: str) -> torch.Tensor:
        """Initialize a unit-norm vector using a seed derived from the biomarker name."""

        import hashlib
        hash_object = hashlib.md5(biomarker_name.encode())
        seed = int(hash_object.hexdigest(), 16) % (2 ** 32)

        rng = np.random.RandomState(seed)

        embedding = rng.normal(0, 0.1, self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return Parameter(torch.tensor(embedding, dtype=torch.float32, requires_grad=True).to(self.device))

    def _build_esm_embeddings(self, all_biomarkers: list):
        print("Building ESM embeddings...")

        for biomarker_name in tqdm(
            all_biomarkers,
            desc="Computing ESM Embeddings",
            **_tqdm_options(self.config),
        ):

            if self.precomputed_esm:
                self.biomarker_embeddings[self._sanitize_name(biomarker_name)] = Parameter(
                    torch.zeros(self.embedding_dim, device=self.device), requires_grad=False)
                continue
            if biomarker_name in self.biomarker_info:
                biomarker_info = self.biomarker_info[biomarker_name]

                if biomarker_info['type'] == 'legit_no_sequence':
                    emb = self._create_learnable_fallback_embedding(biomarker_name)
                    safe_name = self._sanitize_name(biomarker_name)

                elif biomarker_info['type'] == 'has_sequence':
                    sequence = biomarker_info['sequence']
                    raw_emb = self._compute_esm_embedding(biomarker_name, sequence)
                    emb = raw_emb.detach()
                    emb.requires_grad = False
                else:
                    raise ValueError(f"Invalid protein sequence entry for {biomarker_name}")
            else:
                emb = self._create_learnable_fallback_embedding(biomarker_name)
                safe_name = self._sanitize_name(biomarker_name)

            safe_name = self._sanitize_name(biomarker_name)
            self.biomarker_embeddings[safe_name] = emb if isinstance(emb, Parameter) else Parameter(emb, requires_grad=False)

        if not self.precomputed_esm:
            self.esm_model.cpu()
            del self.esm_model
            torch.cuda.empty_cache()
        return

    def _compute_esm_embedding(self, biomarker_name: str, sequence: str) -> torch.Tensor:
        """Average ESM residue embeddings, excluding the start and end tokens."""
        try:
            data = [(biomarker_name, sequence)]
            batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

            token_representations = results["representations"][33]
            sequence_representations = token_representations[0, 1:-1].mean(0)

            return sequence_representations.to(self.device)

        except Exception as e:
            print(f"Error computing ESM embedding for {biomarker_name}: {e}")
            return self._create_learnable_fallback_embedding(biomarker_name)

    def forward(self, biomarker_names: list, intensities: torch.Tensor) -> torch.Tensor:
        """Map intensities of shape (cells, biomarkers) to cell embeddings."""
        valid_biomarkers = []
        valid_biomarker_indices = []
        for i, name in enumerate(biomarker_names):
            if name in self.biomarker_mapping:
                name = self.biomarker_mapping[name]
                if name == "EMPTY":
                    continue
            if name in self.biomarker_to_idx:
                valid_biomarkers.append(name)
                valid_biomarker_indices.append(i)

        if len(valid_biomarkers) < len(biomarker_names):
            warnings.warn(f"{len(biomarker_names) - len(valid_biomarkers)} biomarkers not found in embeddings and will be ignored.")

        if not valid_biomarkers:
            raise ValueError("No valid biomarkers found in the input.")
        valid_intensities = intensities[:, valid_biomarker_indices]
        valid_intensities = torch.tensor(valid_intensities, dtype=torch.float32, device=self.device)

        biomarker_embeddings = torch.stack([
            self.biomarker_embeddings[self._sanitize_name(name)].to(self.device)
            for name in valid_biomarkers])
        cell_profile = torch.matmul(valid_intensities, biomarker_embeddings)
        return cell_profile

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_single_biomarker_embedding(self, biomarker_name: str) -> torch.Tensor:
        if biomarker_name in self.biomarker_mapping:
            biomarker_name = self.biomarker_mapping[biomarker_name]
            if biomarker_name == "EMPTY":
                raise ValueError(f"Invalid biomarker name: {biomarker_name}")
        safe_name = self._sanitize_name(biomarker_name)
        if biomarker_name in self.biomarker_to_idx:
            return self.biomarker_embeddings[safe_name].to(self.device)
        else:
            raise ValueError(f"Invalid biomarker name: {biomarker_name}")

    def get_batched_embeddings(self, biomarker_names: list[str]) -> torch.Tensor:
        mapped_names = [self.biomarker_mapping.get(name, name) for name in biomarker_names]
        unknown = [name for name in mapped_names if name not in self.biomarker_to_idx]
        if unknown:
            raise ValueError(f"Biomarkers absent from the vocabulary: {unknown}")
        embeddings = torch.stack([
            self.biomarker_embeddings[self._sanitize_name(name)] for name in mapped_names
        ], dim=0).to(self.device)

        return embeddings
