"""
Antibody developability feature extraction.

Extracts sequence-based features from heavy and light chain variable domains
using a combination of external tools (abnumber, BioPhi, ScaLoP) and pure Python.
"""

import logging
import re
from collections import OrderedDict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AntibodyFeatures:
    """
    Extract developability features from antibody sequences.

    Combines outputs from:
    - abnumber (numbering, germline assignment, CDR extraction)
    - BioPhi/OASis (humanness scores)
    - ScaLoP (canonical structure classes)
    - Sequence-based analysis (liabilities, CDR properties, charge, pI)
    """

    def __init__(
        self,
        use_abnumber: bool = True,
        use_biophi: bool = True,
        use_scalop: bool = True,
        use_sequence_features: bool = True,
        cdr_definition: str = "north",
        cache_abnumber: bool = True,
    ):
        """
        Args:
            use_abnumber: Enable abnumber-based features (germline, CDR lengths)
            use_biophi: Enable BioPhi humanness scores
            use_scalop: Enable ScaLoP canonical classes
            use_sequence_features: Enable sequence-based features
            cdr_definition: CDR definition scheme (north, chothia, kabat, imgt)
            cache_abnumber: Cache abnumber results
        """
        self.use_abnumber = use_abnumber
        self.use_biophi = use_biophi
        self.use_scalop = use_scalop
        self.use_sequence_features = use_sequence_features
        self.numbering_scheme = "aho"
        self.cdr_definition = cdr_definition.lower()
        self.cache_abnumber = cache_abnumber
        self._abnumber_cache: Dict[str, dict] = {}

        # Lazy import of external tools
        self._abnumber_available = False

        # Validate CDR definition
        if self.cdr_definition not in ["north", "chothia", "kabat", "imgt"]:
            raise ValueError(f"Invalid CDR definition: {cdr_definition}")

        # Lazy import of external tools
        self._biophi_available = False
        self._scalop_available = False

        # Check tool availability
        if self.use_abnumber:
            try:
                import abnumber
                self._abnumber_available = True
            except ImportError:
                logger.warning("abnumber not available. Install with: pip install abnumber")

        if self.use_biophi:
            # BioPhi runs in separate conda env via subprocess
            self._biophi_available = True  # Assume available, will fail gracefully if not

        if self.use_scalop:
            try:
                import scalop
                self._scalop_available = True
            except ImportError:
                logger.warning("ScaLoP not available. Install with: conda install -c bioconda scalop")

    def extract_features(
        self,
        heavy_seq: str,
        light_seq: str,
    ) -> Dict[str, float]:
        """
        Extract all enabled features from heavy and light chain sequences.

        Args:
            heavy_seq: Heavy chain variable domain sequence
            light_seq: Light chain variable domain sequence

        Returns:
            Ordered dictionary of features (str -> float)
        """
        features = OrderedDict()

        # abnumber-based features (germline, CDR lengths)
        if self.use_abnumber and self._abnumber_available:
            abnumber_features = self._extract_abnumber_features(heavy_seq, light_seq)
            features.update(abnumber_features)
        elif self.use_abnumber:
            # Add sentinel values if tool not available
            features.update(self._get_abnumber_sentinel_features())

        # BioPhi humanness scores
        if self.use_biophi:
            biophi_features = self._extract_biophi_features(heavy_seq, light_seq)
            features.update(biophi_features)

        # ScaLoP canonical classes
        if self.use_scalop and self._scalop_available:
            scalop_features = self._extract_scalop_features(heavy_seq, light_seq)
            features.update(scalop_features)
        elif self.use_scalop:
            features.update(self._get_scalop_sentinel_features())

        # Sequence-based features (pure Python - always available)
        if self.use_sequence_features:
            seq_features = self._extract_sequence_features(heavy_seq, light_seq)
            features.update(seq_features)

        return features

    def extract_features_batch(
        self,
        heavy_sequences: List[str],
        light_sequences: List[str],
    ) -> List[Dict[str, float]]:
        """Extract features for multiple antibodies."""
        if len(heavy_sequences) != len(light_sequences):
            raise ValueError("Heavy and light sequence lists must have same length")

        return [
            self.extract_features(h, l)
            for h, l in zip(heavy_sequences, light_sequences)
        ]

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        dummy = self.extract_features(
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
            "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
        )
        return list(dummy.keys())

    def get_feature_dim(self) -> int:
        """
        Get dimension of feature vector (number of features) based on enabled modules.

        Calculates dimension from sentinel/template features without extracting.
        """
        features = OrderedDict()

        # Add sentinel features for each enabled module
        if self.use_abnumber:
            features.update(self._get_abnumber_sentinel_features())

        if self.use_biophi:
            features.update(self._get_biophi_sentinel_features())

        if self.use_scalop:
            features.update(self._get_scalop_sentinel_features())

        if self.use_sequence_features:
            features.update(self._get_sequence_sentinel_features())

        return len(features)

    def features_to_array(self, features: Dict[str, float]):
        """Convert feature dictionary to numpy array."""
        import numpy as np
        return np.array(list(features.values()), dtype=np.float32)

    # ========================================================================
    # abnumber-based features
    # ========================================================================

    def _extract_abnumber_features(self, heavy_seq: str, light_seq: str) -> Dict[str, float]:
        """Extract features using abnumber (germline, CDR lengths)."""
        from abnumber import Chain

        features = OrderedDict()

        # Process VH
        vh_result = self._run_abnumber(heavy_seq, chain_type="heavy")
        if vh_result:
            features["germline_identity_vh"] = vh_result.get("germline_identity", -1.0)
            features["n_mutations_vh"] = vh_result.get("n_mutations", -1)
            features["v_gene_family_vh"] = vh_result.get("v_gene_family", 0)
            features["cdr_h1_length"] = vh_result.get("cdr1_length", -1)
            features["cdr_h2_length"] = vh_result.get("cdr2_length", -1)
            features["cdr_h3_length"] = vh_result.get("cdr3_length", -1)
        else:
            features.update({
                "germline_identity_vh": -1.0,
                "n_mutations_vh": -1,
                "v_gene_family_vh": 0,
                "cdr_h1_length": -1,
                "cdr_h2_length": -1,
                "cdr_h3_length": -1,
            })

        # Process VL
        vl_result = self._run_abnumber(light_seq, chain_type="light")
        if vl_result:
            features["germline_identity_vl"] = vl_result.get("germline_identity", -1.0)
            features["n_mutations_vl"] = vl_result.get("n_mutations", -1)
            features["v_gene_family_vl"] = vl_result.get("v_gene_family", 0)
            features["cdr_l1_length"] = vl_result.get("cdr1_length", -1)
            features["cdr_l2_length"] = vl_result.get("cdr2_length", -1)
            features["cdr_l3_length"] = vl_result.get("cdr3_length", -1)
        else:
            features.update({
                "germline_identity_vl": -1.0,
                "n_mutations_vl": -1,
                "v_gene_family_vl": 0,
                "cdr_l1_length": -1,
                "cdr_l2_length": -1,
                "cdr_l3_length": -1,
            })

        # Total CDR length
        cdr_lengths = [
            features["cdr_h1_length"], features["cdr_h2_length"], features["cdr_h3_length"],
            features["cdr_l1_length"], features["cdr_l2_length"], features["cdr_l3_length"]
        ]
        valid_lengths = [l for l in cdr_lengths if l >= 0]
        features["total_cdr_length"] = sum(valid_lengths) if valid_lengths else -1

        return features

    def _run_abnumber(self, sequence: str, chain_type: str) -> Optional[dict]:
        """Run abnumber on a single sequence."""
        from abnumber import Chain

        # Check cache
        cache_key = f"{chain_type}:{sequence}"
        if self.cache_abnumber and cache_key in self._abnumber_cache:
            return self._abnumber_cache[cache_key]

        try:
            # Parse chain with abnumber (AHO numbering + configurable CDR definition)
            chain = Chain(sequence, scheme=self.numbering_scheme, cdr_definition=self.cdr_definition, assign_germline=True)

            result = {}

            # Germline assignment - use find_merged_human_germline()
            try:
                # Get V gene (assigned automatically with assign_germline=True)
                v_gene = chain.v_gene if hasattr(chain, 'v_gene') else None

                if v_gene:
                    # Get germline sequence and calculate identity
                    germline_chain = chain.find_merged_human_germline()
                    if germline_chain:
                        query_seq = chain.seq
                        germline_seq = germline_chain.seq

                        # Calculate identity (matches / min_length)
                        matches = sum(q == g for q, g in zip(query_seq, germline_seq))
                        min_len = min(len(query_seq), len(germline_seq))
                        identity = matches / min_len if min_len > 0 else 0.0

                        result["germline_identity"] = float(identity)
                        result["n_mutations"] = len(query_seq) - matches
                    else:
                        result["germline_identity"] = -1.0
                        result["n_mutations"] = -1

                    # V gene family from v_gene attribute
                    # Match IGHV, IGKV, or IGLV patterns
                    match = re.search(r'IG[HKL]V(\d+)', v_gene)
                    if match:
                        result["v_gene_family"] = int(match.group(1))
                    else:
                        result["v_gene_family"] = 0
                else:
                    result["germline_identity"] = -1.0
                    result["n_mutations"] = -1
                    result["v_gene_family"] = 0

            except Exception as e:
                logger.warning(f"Germline assignment failed: {e}")
                result["germline_identity"] = -1.0
                result["n_mutations"] = -1
                result["v_gene_family"] = 0

            # CDR lengths
            cdrs = chain.cdr_positions if hasattr(chain, 'cdr_positions') else None
            if cdrs:
                result["cdr1_length"] = len(cdrs.get('cdr1', []))
                result["cdr2_length"] = len(cdrs.get('cdr2', []))
                result["cdr3_length"] = len(cdrs.get('cdr3', []))
            else:
                # Fallback: extract CDRs manually
                cdr1 = chain.cdr1_seq if hasattr(chain, 'cdr1_seq') else None
                cdr2 = chain.cdr2_seq if hasattr(chain, 'cdr2_seq') else None
                cdr3 = chain.cdr3_seq if hasattr(chain, 'cdr3_seq') else None

                result["cdr1_length"] = len(cdr1) if cdr1 else -1
                result["cdr2_length"] = len(cdr2) if cdr2 else -1
                result["cdr3_length"] = len(cdr3) if cdr3 else -1

            # Cache result
            if self.cache_abnumber:
                self._abnumber_cache[cache_key] = result

            return result

        except Exception as e:
            logger.warning(f"abnumber failed for {chain_type}: {e}")
            return None

    def _get_abnumber_sentinel_features(self) -> Dict[str, float]:
        """Return sentinel values for abnumber features when tool unavailable."""
        return OrderedDict({
            "germline_identity_vh": -1.0,
            "n_mutations_vh": -1,
            "v_gene_family_vh": 0,
            "cdr_h1_length": -1,
            "cdr_h2_length": -1,
            "cdr_h3_length": -1,
            "germline_identity_vl": -1.0,
            "n_mutations_vl": -1,
            "v_gene_family_vl": 0,
            "cdr_l1_length": -1,
            "cdr_l2_length": -1,
            "cdr_l3_length": -1,
            "total_cdr_length": -1,
        })

    # ========================================================================
    # BioPhi features
    # ========================================================================

    def _extract_biophi_features(self, heavy_seq: str, light_seq: str) -> Dict[str, float]:
        """Extract humanness scores using BioPhi/OASis Python API via subprocess."""
        features = OrderedDict()

        try:
            # VH humanness
            vh_humanness = self._run_biophi_subprocess(heavy_seq)
            features["humanness_vh"] = float(vh_humanness) if vh_humanness is not None else -1.0

            # VL humanness
            vl_humanness = self._run_biophi_subprocess(light_seq)
            features["humanness_vl"] = float(vl_humanness) if vl_humanness is not None else -1.0

        except Exception as e:
            logger.warning(f"BioPhi failed: {e}")
            features["humanness_vh"] = -1.0
            features["humanness_vl"] = -1.0

        return features

    def _get_biophi_sentinel_features(self) -> Dict[str, float]:
        """Return sentinel values for BioPhi features when tool unavailable."""
        return OrderedDict({
            "humanness_vh": -1.0,
            "humanness_vl": -1.0,
        })

    def _run_biophi_subprocess(self, sequence: str) -> Optional[float]:
        """Run BioPhi humanness using Python API in biophi conda env."""
        import subprocess
        from pathlib import Path

        try:
            # BioPhi database path (relative to project root)
            db_path = Path(__file__).parent.parent.parent / "external" / "OASis_9mers_v1.db"

            # Python code to run in biophi env
            python_code = f"""
from biophi.humanization.methods.humanness import get_chain_humanness, OASisParams
from abnumber import Chain

seq = "{sequence}"
db_path = "{db_path}"

# Create Chain and get humanness
chain = Chain(seq, scheme='imgt')
params = OASisParams(oasis_db_path=db_path, min_fraction_subjects=0.01)
chain_humanness = get_chain_humanness(chain, params=params)
score = chain_humanness.get_oasis_identity(min_fraction_subjects=0.01)
print(score)
"""

            # Run in biophi conda env
            cmd = [
                "conda", "run", "-n", "biophi",
                "python", "-c", python_code
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.warning(f"BioPhi subprocess failed: {result.stderr}")
                return None

            # Parse output (should be a single number)
            score = float(result.stdout.strip())
            return score

        except Exception as e:
            logger.warning(f"BioPhi subprocess execution failed: {e}")
            return None

    # ========================================================================
    # ScaLoP features
    # ========================================================================

    def _extract_scalop_features(self, heavy_seq: str, light_seq: str) -> Dict[str, float]:
        """Extract canonical structure classes using ScaLoP."""
        features = OrderedDict()

        try:
            from scalop.predict import assign

            # ScaLoP expects heavy/light separated by '/'
            input_seq = f"{heavy_seq}/{light_seq}"

            # Run ScaLoP prediction
            result = assign(input_seq)

            # Result is a list of dicts: [heavy_result, light_result]
            # Each has 'outputs' dict with CDR names as keys
            # Each CDR value is [cdr_name, sequence, class_code, pdb]
            # class_code format: "H1-13-A" where 13 is the class number

            canonical_classes = {}
            for item in result:
                outputs = item.get('outputs', {})
                for cdr_name, cdr_data in outputs.items():
                    if len(cdr_data) >= 3:
                        class_code = cdr_data[2]  # e.g., "H1-13-A"
                        # Extract class number from code
                        match = re.search(r'-(\d+)', class_code)
                        if match:
                            canonical_classes[cdr_name] = int(match.group(1))
                        else:
                            canonical_classes[cdr_name] = 0

            features["canonical_class_l1"] = canonical_classes.get("L1", 0)
            features["canonical_class_l2"] = canonical_classes.get("L2", 0)
            features["canonical_class_l3"] = canonical_classes.get("L3", 0)
            features["canonical_class_h1"] = canonical_classes.get("H1", 0)
            features["canonical_class_h2"] = canonical_classes.get("H2", 0)

        except Exception as e:
            logger.warning(f"ScaLoP failed: {e}")
            features.update(self._get_scalop_sentinel_features())

        return features

    def _get_scalop_sentinel_features(self) -> Dict[str, float]:
        """Return sentinel values for ScaLoP features when tool unavailable."""
        return OrderedDict({
            "canonical_class_l1": 0,
            "canonical_class_l2": 0,
            "canonical_class_l3": 0,
            "canonical_class_h1": 0,
            "canonical_class_h2": 0,
        })

    # ========================================================================
    # Sequence-based features (pure Python)
    # ========================================================================

    def _extract_sequence_features(self, heavy_seq: str, light_seq: str) -> Dict[str, float]:
        """Extract sequence-based features without external tools."""
        features = OrderedDict()

        # 1. Sequence liabilities (count across both chains)
        vh_liabilities = self._count_liabilities(heavy_seq)
        vl_liabilities = self._count_liabilities(light_seq)

        # Deamidation sites (split NG and NS)
        features["n_deamidation_ng"] = vh_liabilities["ng"] + vl_liabilities["ng"]
        features["n_deamidation_ns"] = vh_liabilities["ns"] + vl_liabilities["ns"]

        # Glycosylation sites
        features["n_glycosylation"] = vh_liabilities["glycosylation"] + vl_liabilities["glycosylation"]

        # Oxidation sites (split M and W)
        features["n_oxidation_m"] = vh_liabilities["met"] + vl_liabilities["met"]
        features["n_oxidation_w"] = vh_liabilities["trp"] + vl_liabilities["trp"]

        # Unpaired cysteines
        features["n_unpaired_cys"] = vh_liabilities["unpaired_cys"] + vl_liabilities["unpaired_cys"]

        # 2. CDR-H3 properties (use simple extraction)
        cdr_h3_seq = self._extract_cdr_h3_simple(heavy_seq)
        if cdr_h3_seq and len(cdr_h3_seq) >= 5:
            features["cdr_h3_net_charge"] = self._calculate_net_charge(cdr_h3_seq)
            features["cdr_h3_hydrophobicity"] = self._calculate_hydrophobicity(cdr_h3_seq)
        else:
            # Use full sequence if CDR-H3 extraction fails
            features["cdr_h3_net_charge"] = self._calculate_net_charge(heavy_seq)
            features["cdr_h3_hydrophobicity"] = self._calculate_hydrophobicity(heavy_seq)

        # 3. Chain properties (pI and charge)
        features["pi_vh"] = self._calculate_pi(heavy_seq)
        features["pi_vl"] = self._calculate_pi(light_seq)

        vh_charge = self._calculate_net_charge(heavy_seq)
        vl_charge = self._calculate_net_charge(light_seq)

        features["net_charge_vh"] = vh_charge
        features["net_charge_vl"] = vl_charge
        features["charge_asymmetry"] = abs(vh_charge - vl_charge)

        return features

    def _get_sequence_sentinel_features(self) -> Dict[str, float]:
        """Return sentinel/template values for sequence-based features."""
        return OrderedDict({
            "n_deamidation_ng": 0,
            "n_deamidation_ns": 0,
            "n_glycosylation": 0,
            "n_oxidation_m": 0,
            "n_oxidation_w": 0,
            "n_unpaired_cys": 0,
            "cdr_h3_net_charge": 0.0,
            "cdr_h3_hydrophobicity": 0.0,
            "pi_vh": 0.0,
            "pi_vl": 0.0,
            "net_charge_vh": 0.0,
            "net_charge_vl": 0.0,
            "charge_asymmetry": 0.0,
        })

    def _count_liabilities(self, sequence: str) -> dict:
        """Count sequence liabilities."""
        liabilities = {
            "ng": 0,
            "ns": 0,
            "glycosylation": 0,
            "met": 0,
            "trp": 0,
            "unpaired_cys": 0,
        }

        # Deamidation: NG and NS separately
        liabilities["ng"] = len(re.findall(r'NG', sequence))
        liabilities["ns"] = len(re.findall(r'NS', sequence))

        # N-glycosylation: N[^P][ST]
        liabilities["glycosylation"] = len(re.findall(r'N[^P][ST]', sequence))

        # Oxidation: M and W separately
        liabilities["met"] = sequence.count('M')
        liabilities["trp"] = sequence.count('W')

        # Unpaired cysteines (odd number of C)
        cys_count = sequence.count('C')
        liabilities["unpaired_cys"] = cys_count % 2

        return liabilities

    def _extract_cdr_h3_simple(self, heavy_seq: str) -> Optional[str]:
        """
        Extract CDR-H3 using simple heuristic.

        CDR-H3 typically ends with conserved W (position 103 in IMGT).
        Look for pattern: C ... W (before final ~10-15 residues).
        """
        # Find last W (before C-terminal region)
        w_positions = [i for i, aa in enumerate(heavy_seq) if aa == 'W']
        if not w_positions:
            return None

        # Find last C before the W
        last_w = w_positions[-1]
        c_positions = [i for i, aa in enumerate(heavy_seq[:last_w]) if aa == 'C']
        if not c_positions:
            return None

        last_c = c_positions[-1]

        # CDR-H3: between C and W (exclusive)
        cdr_h3 = heavy_seq[last_c + 1:last_w]

        # Sanity check: typical CDR-H3 is 5-30 residues
        if 5 <= len(cdr_h3) <= 30:
            return cdr_h3

        return None

    def _calculate_net_charge(self, sequence: str) -> float:
        """Calculate net charge at pH 7.0 (K, R, H positive; D, E negative)."""
        positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
        negative = sequence.count('D') + sequence.count('E')
        return float(positive - negative)

    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate fraction of hydrophobic residues (AILMFWYV)."""
        if len(sequence) == 0:
            return 0.0
        hydrophobic = sum(sequence.count(aa) for aa in 'AILMFWYV')
        return float(hydrophobic) / len(sequence)

    def _calculate_pi(self, sequence: str) -> float:
        """Calculate isoelectric point using Biopython."""
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            pi = ProteinAnalysis(sequence).isoelectric_point()
            return float(pi)
        except ImportError:
            logger.warning("Biopython not available for pI calculation")
            return -1.0
        except Exception as e:
            logger.warning(f"pI calculation failed: {e}")
            return -1.0
