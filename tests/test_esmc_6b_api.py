#!/usr/bin/env python3
"""Test ESM-C 6B API connectivity."""

import os
import sys

def test_esmc_6b():
    """Test if ESM-C 6B API works."""

    # Check ESM package version
    try:
        import esm
        version = getattr(esm, '__version__', 'unknown')
        print(f"✓ ESM package installed (version: {version})")
    except ImportError:
        print("❌ ESM package not installed")
        print("   Install with: pip install esm")
        return False

    # Check for token
    token = os.getenv("FORGE_TOKEN")
    if not token:
        print("❌ FORGE_TOKEN environment variable not set")
        print("\nTo test ESM-C 6B API, you need to:")
        print("1. Get a Forge API token from EvolutionaryScale")
        print("   - Visit: https://forge.evolutionaryscale.ai")
        print("   - Sign up/login and get your API token")
        print("2. Set the token:")
        print("   export FORGE_TOKEN=<your_token>")
        print("3. Run this script again:")
        print("   python test_esmc_6b_api.py")
        return False

    print(f"✓ Found FORGE_TOKEN (length: {len(token)})")

    # Try importing ESM SDK
    try:
        from esm.sdk.forge import ESM3ForgeInferenceClient
        from esm.sdk.api import ESMProtein, LogitsConfig
        print("✓ ESM SDK imports successful")
    except ImportError as e:
        print(f"❌ Failed to import ESM SDK: {e}")
        print("\nInstall with: pip install esm")
        return False

    # Try connecting to Forge API
    try:
        print("\n🔄 Connecting to Forge API...")
        client = ESM3ForgeInferenceClient(
            model="esmc-6b-2024-12",
            url="https://forge.evolutionaryscale.ai",
            token=token,
        )
        print("✓ Successfully connected to Forge API")
    except Exception as e:
        print(f"❌ Failed to connect to Forge API: {e}")
        return False

    # Try encoding a simple sequence
    try:
        print("\n🔄 Testing encoding with a sample antibody sequence...")

        # Simple test sequence (first 20 residues of a VH chain)
        test_seq = "EVQLVESGGGLVQPGGSLRL"

        protein = ESMProtein(sequence=test_seq)
        print(f"✓ Created ESMProtein object for sequence: {test_seq[:30]}...")

        # Encode
        print("🔄 Encoding protein...")
        protein_tensor = client.encode(protein)
        print("✓ Encoding successful")

        # Get embeddings
        print("🔄 Getting embeddings...")
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        print("✓ Embeddings retrieved")

        # Check embedding shape
        embeddings = logits_output.embeddings
        print(f"✓ Embedding shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")

        print("\n✅ ESM-C 6B API test PASSED!")
        print("\nYou can now use the 6B model with:")
        print("  python train.py --config config/esmc_6b.yaml --property HIC")
        return True

    except Exception as e:
        print(f"❌ Failed to encode sequence: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("ESM-C 6B API Connectivity Test")
    print("="*60)

    success = test_esmc_6b()

    sys.exit(0 if success else 1)
